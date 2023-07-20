import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import torch.autograd as autograd
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torchvision.models as tvmodels
from datetime import datetime
from functools import partial 
from sklearn.utils import check_random_state
import sklearn.metrics 
import json
import os
import string 
from sklearn.datasets import make_classification, make_regression

import tensorflow as tf
import tensorflow_datasets as tfds

print("Tensorflow version ",tf.__version__)
print("Tensorflow Datasets version ",tfds.__version__)

def resetRandomStates(manualseed=47):
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)

def testModel(model, testX, testY):
  print("------------------ Testing model ------------------")
  results = model.evaluate(testX.numpy(), testY.numpy()) 
  
def export2cloud(model, datasetName, model2Explain):
  # export_path = "gs://exp_gg_001/{}/{}".format(datasetName, model2Explain)
  export_path = "{}/{}".format(datasetName, model2Explain)
  model.save(export_path)    
  print("Saving model to", export_path, "finished!")
  return export_path

resetRandomStates()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Create Credentials for the Access to gcloud storage
SERVICE_ACCOUNT_FILE = 'my-service-acct.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE


def extractDataTable(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x, y, w = next(iter(loader))
    return x, y, w 

def getSplittedDataset(expset, trainpart=0.6, testpart=0.2, predictpart=0.2):
    assert trainpart + testpart + predictpart == 1, 'Train + Test + Validation should be 1'
    x, y, _=expset[0]
    print("\n[FUNCTION]: Splitting dataset by getSplittedDataset()......")
    print("Display first (x, y) pair of dataset:\n %s, %s" % (x, y) )
    print("Shape of (x, y): %s %s" % (x.shape, y.shape) )
 
    train_len = int(len(expset) * trainpart)
    test_len = int(len(expset) * testpart)
    total_len = int(len(expset))
  
    trainset, remainset = torch.utils.data.random_split(expset, [train_len, total_len-train_len])
    print("len(trainset): %d" % (len(trainset)) )

    testset, predictset = torch.utils.data.random_split(remainset, [test_len, len(remainset)-test_len])
    print("len(testset): %d" % (len(testset)) )
    print("len(predictset): %d" % (len(predictset)) )
    return trainset, testset, predictset

class ExperimentDataset(Dataset):

    def __init__(self, datafilepath, classNum=1):
        full_data_table = np.genfromtxt(datafilepath, delimiter=',')
        data = torch.from_numpy(full_data_table).float()
        self.samples = data[:, :-1]
        self.labels = data[:, -1]
        
        self.feature_min, _ = self.samples.min(dim=0)
        self.feature_max, _ = self.samples.max(dim=0)
        self.samples = (self.samples - self.feature_min)/(self.feature_max-self.feature_min)
        
        self.classNum = classNum
        if classNum == 1:
            # regression dataset
            tmax = self.labels.max()
            tmin = self.labels.min()
            self.labels = (self.labels - tmin)/(tmax-tmin)
            self.classWeights = None
            self.sampleWeights =  torch.ones_like(self.labels)
        elif classNum > 1:
            self.labels = self.labels.long()
            # compute class weights
            bins = self.classNum * np.bincount(self.labels)
            assert bins.size == (self.labels.max().item()+1), "bins should be equal to class numbers"
            self.classWeights = torch.tensor(len(self.labels) / bins).float()
            self.sampleWeights = torch.tensor([self.classWeights[label] for label in self.labels]).float()
 
        self.featureNum = self.samples.size(1)

        print("Creating dataset from %s, len(samples): %d, feature number: %d" %(datafilepath, len(self.labels), self.samples.shape[1]) )
                      
    def __len__(self):
        return len(self.samples)
    
    def get_class_weights(self):
        return self.classWeights
            
    def get_sample_weights(self):
        return self.sampleWeights

    def __getitem__(self, index):
        return self.samples[index], self.labels[index], self.sampleWeights[index]  

def explainSamples(inputName, numpyArr):

    pred_list = []
    for idx in range(len(numpyArr)):
        pred_list.append({inputName: numpyArr[idx].tolist()})
        
    # Explainable AI supports generating explanations for multiple predictions
    explanations = model_artifact_with_metadata.explain(pred_list)
    shapArr = np.zeros((len(numpyArr), nFeatures))
    for i, exp in enumerate(explanations):
        shapArr[i] = exp.as_tensors()[inputName]
        
    shapArr = np.expand_dims(shapArr, axis=0)
    print("Shape of shapArr:", shapArr.shape)
    print("Sample of shapArr[0][0]:\n", shapArr[0][0])
    return shapArr 
    
def saveShapley(datasetName, dataSplitType, modelType, shapArr, dataArr, suffix):
    svName = "shaps/{}_Google_{}_{}_Ref1_{}.sv".format(datasetName, dataSplitType, modelType, suffix)
    gtName = "shaps/{}_Google_{}_{}_Ref1_{}.gt".format(datasetName, dataSplitType, modelType, suffix)
    torch.save(shapArr, svName)
    torch.save(dataArr, gtName)
    print("Save shaple values to", svName)
    print("Save ground truth features to", gtName)
    return 0
        
def saveModelOutput(datasetName, dataSplitType, modelType, dataArr):
    outName = "shaps/{}_Google_{}_{}_Ref1.out".format(datasetName, dataSplitType, modelType)
    torch.save(dataArr, outName)
    print("Save model outputs to", outName)
    return 0
        
classDatasetDict = {"adult" : 2, "bank" : 2, "credit" : 2, "diabetes" : 3, "energy" : 1, "news" : 5}
datasetName = "bank"

classNum = classDatasetDict[datasetName]

expset = ExperimentDataset(datasetName+".csv", classNum=classNum)
nFeatures = expset.featureNum
print("Number of features:", nFeatures)
usedSamples = 1000
trainset, testset, predictset = getSplittedDataset(expset) 
trainX, trainY, trainW = extractDataTable(trainset)
testX, testY, testW = extractDataTable(testset)
predictX, predictY, predictW = extractDataTable(predictset)
randomX = torch.rand(usedSamples, trainX.size(1))

for model2Explain in ("NN", "RF", "GBDT", "SVM"):

    # Step 1: train models 
    
    if model2Explain == "NN":
        if classNum > 1:    # classification
            assert classNum > 1, "For regression, classNum > 1 is required!"
            model2Explain = "NN"
            # Build your model
            model = tf.keras.Sequential(name="{}_{}_{}".format(datasetName, model2Explain, classNum))
            model.add(tf.keras.layers.Dense(nFeatures*2, input_dim=nFeatures, activation='sigmoid'))
            model.add(tf.keras.layers.Dense(nFeatures*2, activation='sigmoid'))
            model.add(tf.keras.layers.Dense(classNum, activation='softmax'))
            optimizer = tf.keras.optimizers.Adam(0.001)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            batch_size = 256
            epochs = 10

            model.fit(trainX.numpy(), trainY.numpy(), sample_weight=trainW.numpy(), batch_size=batch_size, epochs=epochs)

            testModel(model, testX, testY) 
            print("Sample outputs:\n", model(testX[:6].numpy()))
            export_path = export2cloud(model, datasetName, model2Explain)

        elif classNum == 1:     # regression
            model2Explain = "NN"

            assert classNum == 1, "For regression, classNum == 1 is required!"
            # Build your model
            model = tf.keras.Sequential(name="{}_{}_{}".format(datasetName, model2Explain, classNum))
            model.add(tf.keras.layers.Dense(nFeatures*3, input_dim=nFeatures, activation='sigmoid'))
            model.add(tf.keras.layers.Dense(classNum))
            optimizer = tf.keras.optimizers.Adam(0.001)
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
            batch_size = 256
            epochs = 2

            model.fit(trainX.numpy(), trainY.numpy(), sample_weight=trainW.numpy(), batch_size=batch_size, epochs=epochs)

            testModel(model, testX, testY) 
            print("Sample outputs:\n", model(testX[:6].numpy()))
            export_path = export2cloud(model, datasetName, model2Explain)
            
    elif model2Explain == "RF":
        if classNum > 1:    # classification
            import tensorflow_decision_forests as tfdf
            print("Found TensorFlow Decision Forests v" + tfdf.__version__)
            model2Explain = "RF"
            assert classNum > 1, "For regression, classNum > 1 is required!"
            model = tfdf.keras.RandomForestModel(num_trees=100, max_depth=5, task=tfdf.keras.Task.CLASSIFICATION)
            # model.add_loss(tf.keras.losses.SparseCategoricalCrossentropy())
            model.compile(metrics=['accuracy'])
            batch_size = 256

            model.fit(trainX.numpy(), trainY.numpy(), sample_weight=trainW.numpy(), batch_size=batch_size)

            testModel(model, testX, testY) 
            print("Sample outputs:\n", model(testX[:6].numpy()))
            export_path = export2cloud(model, datasetName, model2Explain)


        elif classNum == 1:     # regression
            import tensorflow_decision_forests as tfdf
            print("Found TensorFlow Decision Forests v" + tfdf.__version__)
            model2Explain = "RF"
            assert classNum == 1, "For regression, classNum == 1 is required!"
            model = tfdf.keras.RandomForestModel(num_trees=100, max_depth=5, task=tfdf.keras.Task.REGRESSION)
            model.compile(metrics=['mse'])
            batch_size = 256

            model.fit(trainX.numpy(), trainY.numpy(), sample_weight=trainW.numpy(), batch_size=batch_size)

            testModel(model, testX, testY) 
            print("Sample outputs:\n", model(testX[:6].numpy()))
            export_path = export2cloud(model, datasetName, model2Explain)


    elif model2Explain == "GBDT":
        if classNum > 1:    # classification
            import tensorflow_decision_forests as tfdf
            print("Found TensorFlow Decision Forests v" + tfdf.__version__)
            model2Explain = "GBDT"
            assert classNum > 1, "For regression, classNum > 1 is required!"
            model = tfdf.keras.GradientBoostedTreesModel(num_trees=100, max_depth=3, task=tfdf.keras.Task.CLASSIFICATION)
            # model.add_loss(tf.keras.losses.SparseCategoricalCrossentropy())
            model.compile(metrics=['accuracy'])
            batch_size = 256

            model.fit(trainX.numpy(), trainY.numpy(), sample_weight=trainW.numpy(), batch_size=batch_size)

            testModel(model, testX, testY) 
            print("Sample outputs:\n", model(testX[:6].numpy()))
            export_path = export2cloud(model, datasetName, model2Explain)


        elif classNum == 1:     # regression
            import tensorflow_decision_forests as tfdf
            print("Found TensorFlow Decision Forests v" + tfdf.__version__)
            model2Explain = "GBDT"
            assert classNum == 1, "For regression, classNum == 1 is required!"
            model = tfdf.keras.GradientBoostedTreesModel(num_trees=100, max_depth=3, task=tfdf.keras.Task.REGRESSION)
            model.compile(metrics=['mse'])
            batch_size = 256

            model.fit(trainX.numpy(), trainY.numpy(), sample_weight=trainW.numpy(), batch_size=batch_size)

            testModel(model, testX, testY) 
            print("Sample outputs:\n", model(testX[:6].numpy()))
            export_path = export2cloud(model, datasetName, model2Explain)

    elif model2Explain == "SVM":
        if classNum > 1:    # classification
            from tensorflow import keras
            from tensorflow.keras import layers
            from tensorflow.keras.layers.experimental import RandomFourierFeatures
            model2Explain = "SVM"
            assert classNum > 1, "For regression, classNum > 1 is required!"
            # Build your model
            model = keras.Sequential([
              keras.Input(shape=(nFeatures,)),
              RandomFourierFeatures(
                  output_dim=nFeatures*8,
                  scale=10.,
                  kernel_initializer='gaussian'),
              layers.Dense(units=classNum, activation='softmax'),
            ])

            optimizer = tf.keras.optimizers.Adam(0.001)
            model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            batch_size = 256
            epochs = 10

            model.fit(trainX.numpy(), trainY.numpy(), sample_weight=trainW.numpy(), batch_size=batch_size, epochs=epochs)

            testModel(model, testX, testY) 
            print("Sample outputs:\n", model(testX[:6].numpy()))
            export_path = export2cloud(model, datasetName, model2Explain)



        elif classNum == 1:     # regression
            from tensorflow import keras
            from tensorflow.keras import layers
            from tensorflow.keras.layers.experimental import RandomFourierFeatures
            model2Explain = "SVM"
            assert classNum == 1, "For regression, classNum == 1 is required!"
            # Build your model
            model = keras.Sequential([
              keras.Input(shape=(nFeatures,)),
              RandomFourierFeatures(
                  output_dim=nFeatures*8,
                  scale=10.,
                  kernel_initializer='gaussian'),
              layers.Dense(units=classNum),
            ])

            optimizer = tf.keras.optimizers.Adam(0.001)
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
            batch_size = 256
            epochs = 1

            model.fit(trainX.numpy(), trainY.numpy(), sample_weight=trainW.numpy(), batch_size=batch_size, epochs=epochs)

            testModel(model, testX, testY) 
            print("Sample outputs:\n", model(testX[:6].numpy()))
            export_path = export2cloud(model, datasetName, model2Explain)

    # Step 2: explain models 
    
    from explainable_ai_sdk.metadata.tf.v2 import SavedModelMetadataBuilder
    import explainable_ai_sdk

    print("export_path:", export_path)
    builder = SavedModelMetadataBuilder(export_path)
    # builder.set_numeric_metadata(
    #     model.input.name.split(':')[0],
    #     input_baselines=[trainX.numpy().mean(axis=0).tolist()],
    #     index_feature_mapping=[str(i) for i in range(nFeatures)]
    # )
    # builder.save_metadata(export_path)
    builder.save_model_with_metadata(export_path)
    metaData = builder.get_metadata()
    print(builder.get_metadata())
    inputName = list(metaData["inputs"].keys())[0]
    print("inputName in metaData:", inputName)

    print("\n------------------------------------------\n")

    # Load the model and adjust the configuration for Explainable AI parameters
    num_paths = 50
    model_artifact_with_metadata = explainable_ai_sdk.load_model_from_local_path(export_path, explainable_ai_sdk.SampledShapleyConfig(num_paths))

    for dataType in ("Test", "Validation", "Random"):
        print(datasetName, dataType, model2Explain)
        if dataType == "Test":
            data2explain = testX[:usedSamples, :] 
            
        elif dataType == "Validation":
            data2explain = predictX[:usedSamples, :]
            # saveModelOutput(datasetName.capitalize(), dataType, model2Explain, model(data2explain.numpy()))
            
        elif  dataType == "Random":
            data2explain = randomX
            
        shapArr = explainSamples(inputName, data2explain.numpy())
        suffix = ""
        saveShapley(datasetName.capitalize(), dataType, model2Explain, shapArr, data2explain.numpy(), suffix)
    























