import configparser
from queue import PriorityQueue
from datetime import datetime
import os 
import torch
import numpy as np
import random
import logging
import time
import math
from Datasets import extractDataTable
from torch.utils.data import Dataset, DataLoader

from Datasets import FeatureInferenceDataset
from Models import FeatureInference_NN

def resetRandomStates(manualseed=47):
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dm %ds' % (m, s) if h==0 else '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs)) 
    
def getSplittedDataset(trainpart, expset):
    # resetRandomStates()
    x, y=expset[0]
    logging.critical("\n[FUNCTION]: Splitting dataset by getSplittedDataset()......")
    logging.warning("Display first (x, y) pair of dataset:\n x = \n%s, y = \n%s", x, y)
    logging.warning("Shape of (x, y): (%s, %s)", x.shape, y.shape)
 
    train_len = int(len(expset) * trainpart)
    test_len = int(len(expset)) - train_len
    
    trainset, testset= torch.utils.data.random_split(expset, [train_len, test_len])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    logging.critical("len(trainset): %d", len(trainset))
    logging.critical("len(testset): %d", len(testset))

    return trainset, testset, trainloader, testloader
    
class AttackRegressor():
    def __init__(self, path, datasetName, APIType, modelType, sampleK=None, sample2use=-1, class2Use=0, quantization=-1, dropoutrate=-1):
        logging.critical("\n[FUNCTION]: Creating AttackRegressor()......")
        self.path = path 
        self.datasetName = datasetName
        self.APIType = APIType
        self.modelType = modelType
        self.class2Use = class2Use
        self.sample2use = sample2use
        self.sampleK = sampleK 
        self.quantization = quantization
        self.dropoutrate = dropoutrate
        self.regset = FeatureInferenceDataset(path, datasetName, APIType, "Test", modelType, sampleK, sample2use, class2Use, quantization=quantization, dropoutrate=dropoutrate)
        self.testSet = FeatureInferenceDataset(path, datasetName, APIType, "Validation", modelType, sampleK, sample2use=-1, class2Use=class2Use, smax = self.regset.smax, smin=self.regset.smin, quantization=quantization, dropoutrate=dropoutrate)
        
        _, _, self.trainloader, self.testloader = getSplittedDataset(0.8, self.regset)
    
        self.model = FeatureInference_NN(self.regset.get_feature_num(), self.regset.get_feature_num())
        logging.warning("Structure of the attack model: %s", self.model)


    def train(self, epochs=600):
        def check_test_accuracy(mymodel, dataloader, loss_fn):
            mymodel.eval()
            accur = 0.0
            base = 0
            with torch.no_grad():
                for x, y in dataloader:  
                    yhat = mymodel(x)
                    loss = loss_fn(yhat, y)
                    accur += loss.item()
                    base += 1
            return accur / base
        
        trainloader = self.trainloader
        testloader = self.testloader
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.L1Loss(reduction='mean')

        test_interval = 1 if epochs<5 else int(epochs / 5)
        
        self.model.train()

        for epoch in range(epochs):
            accurate = 0.0
            train_accur_base = 0.0
            for x, y in trainloader: 
                optimizer.zero_grad()
                yhat = self.model(x)
                loss = loss_fn(yhat, y)
                
                loss.backward()
                optimizer.step()
                accurate += loss.item()  
                train_accur_base += 1
                
            if (epoch % test_interval == 0 and epoch > 0) or epoch == epochs - 1:
                # for each epoch, print information
                train = accurate / train_accur_base
                test = check_test_accuracy(self.model, testloader, loss_fn)
                self.model.train()
                logging.critical("In epoch {}, train loss is {}, test loss is {}.".format(epoch, train, test))
                
        return self.model         
                
    def test(self, lossType="L1Loss", printSamples=4):
        def L1LossPerFeature(yhat, y):
            return torch.sqrt((yhat - y)**2)
        def MSELossPerFeature(yhat, y):
            return (yhat - y)**2
    
        logging.critical("\n[FUNCTION]: In AttackRegressor -> test()......")
        
        testset = self.testSet
        regmodel = self.model 
        if lossType == "L1Loss":
            lossfn = torch.nn.L1Loss(reduction='mean')
            lossfnPerFeature = L1LossPerFeature
        elif lossType == "MSELoss":
            lossfn = torch.nn.MSELoss(reduction='mean')
            lossfnPerFeature = MSELossPerFeature
        else:
            lossfn = None 
        no = len(testset)
        accur = 0.0
        randUniformAccur = 0.0
        randNormalAccur = 0.0
        randEmpericalAccur = 0.0
        x, y = testset[0]
        lossPerFeature = torch.zeros_like((x.unsqueeze(0)))
        with torch.no_grad(): 
            for i in range(no):
                x, y = testset[i]
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)
                
                randomUniform = torch.rand(y.shape)
                randomEmperical = torch.zeros(y.shape)
                randSampIdx = np.random.randint(0, len(self.regset.features), size=self.regset.get_feature_num())  
                randomEmperical = torch.zeros(y.shape)
                for i in range(self.regset.get_feature_num()):
                    randomEmperical[0][i] = self.regset.features[randSampIdx[i]][i]
                randomNorm = torch.normal(0.5, 0.25, size=y.shape)
                randomNorm = randomNorm.clamp(0, 1)
                yhat = regmodel(x)
                loss = lossfn(yhat, y)
                randUniformLoss = lossfn(randomUniform, y)
                randNormLoss = lossfn(randomNorm, y)
                randEmpericalLoss = lossfn(randomEmperical, y)
                accur += loss  
                randUniformAccur += randUniformLoss
                randNormalAccur += randNormLoss
                randEmpericalAccur += randEmpericalLoss
                lossPerFeature += lossfnPerFeature(yhat, y)
                if printSamples > 0:
                    logging.warning("-------- Sample %s --------", i)
                    logging.warning("Attack Results:\n%s", yhat.data)
                    logging.warning("Ground Truth:\n%s", y.data)
                    printSamples -= 1
                    
        logging.critical("\n\nFor AttackRegressor (attack 1) setting (%s, %s, %s, sample2use=%s, sampleK=%s, quantization=%s, dropout=%s):", self.datasetName, self.APIType, self.modelType, self.sample2use, self.sampleK, self.quantization, self.dropoutrate)
        logging.critical("The %s loss on the prediction set is: %s", lossType, (accur/no).item())
        logging.critical("The Emperical loss of Random Guess is: %s", randEmpericalAccur/no)
        
        logging.critical("The %s loss on the Uniform Random set is: %s", lossType, randUniformAccur/no)
        logging.critical("The %s loss on the Normal Random set is: %s", lossType, randNormalAccur/no)
        logging.critical("The %s loss per feature on the prediction set is:\n%s", lossType, lossPerFeature/no)
        logging.critical("\n\n<----------------- Finished for this AttackRegressor TEST ----------------->\n\n")
    
class AttackAverager():
    def __init__(self, path, datasetName, APIType, modelType, sampleK=None, sample2use=-1, class2Use=0, quantization=-1, dropoutrate=-1):
        logging.critical("\n[FUNCTION]: Creating AttackAverager()......")
        self.path = path 
        self.datasetName = datasetName
        self.APIType = APIType
        self.modelType = modelType
        self.sampleK = sampleK
        self.sample2use = sample2use
        self.class2Use = class2Use
        self.quantization = quantization
        self.dropoutrate = dropoutrate
        self.regRandomSet = FeatureInferenceDataset(path, datasetName, APIType, "Random", modelType, sampleK, sample2use, class2Use, quantization=quantization, dropoutrate=dropoutrate)
        self.testSet = FeatureInferenceDataset(path, datasetName, APIType, "Validation", modelType, sampleK, sample2use=-1, class2Use=class2Use, smax = self.regRandomSet.smax, smin=self.regRandomSet.smin, quantization=quantization, dropoutrate=dropoutrate)
      
    def test(self, loss="L1Loss", threshold=0.05):
        
        def near(tensorx, tensory, threshold):
            # L2 Norm 
            res = ((tensorx - tensory)**2).sum()
            res = torch.sqrt(res)
            if res < threshold:
                return True 
            return False
            
        def L2Norm(tensorx, tensory):
            #     print(tensorx.shape, tensory.shape)
            res = ((tensorx - tensory)**2).sum()
            return torch.sqrt(res)
        logging.critical("\n[FUNCTION]: In AttackAverager -> test()......")
  
        regRandSet = self.regRandomSet
        testSet = self.testSet
        CandidatesThreshold = 30
        inferedValues = torch.zeros_like(testSet.features)
        featureMask = torch.zeros_like(testSet.features)

        start = time.time()
        n_features = testSet.get_feature_num()

        for feature in range(n_features):   # for each feature
            logging.critical("\nFor feature {} / {}".format(feature, n_features-1)) 
         
            for index in range(len(testSet)):    # for each sample ID 
                phit, gtt = testSet[index]
                targetPhi = phit[feature]   # try to obtain a feature set with phis close to this target 

                inferedFeatureSum = 0 

                counter = 0 
                feasibleIndexes = PriorityQueue()  
                for i, randPhi in enumerate(regRandSet.shaps):
                    feasibleIndexes.put(( L2Norm(targetPhi, randPhi[feature]), i))
                    
        #######################################################################################
                tempList = []
                while not feasibleIndexes.empty():
                    l2norm, feasibleIdx = feasibleIndexes.get()
                    if counter > CandidatesThreshold: # l2norm > L2Threshold: # and 
                        break
                    else:
                        inferedFeatureSum += regRandSet.features[feasibleIdx][feature].data
                        tempList.append(regRandSet.features[feasibleIdx][feature].data)
                        counter += 1
        #######################################################################################            
                targetValue = gtt[feature]
                inferedValue = 0 if counter==0 else inferedFeatureSum / counter
                
                inferedValues[index][feature] = inferedValue
                # if loss == "L1Loss":
                    # deviate = abs(inferedValue-targetValue)
                # elif loss == "MSELoss":
                    # deviate = (inferedValue-targetValue)**2
                    
                predInterval = np.max(tempList) - np.min(tempList)
                if predInterval < 0.5:
                    featureMask[index][feature] = 1 
                
            logging.critical("Average loss for feature {} is: {}".format(feature, (torch.abs(inferedValues[:, feature] - testSet.features[:, feature]) * featureMask[:, feature]).sum() / featureMask[:, feature].sum()) )
            logging.critical("Success rate for feature {} is {}".format(feature, (featureMask[:, feature]).sum() / len(featureMask[:, feature])) )
            logging.critical('>>> %s (%d %d%%) finished for explaining this feature <<<\n' % (timeSince(start, (feature+1) / (n_features)), feature+1, (feature+1) / (n_features) * 100)) 
            # logging.critical("Loss for feature %s: %s", feature, avgLoss[feature])


        
        logging.critical("\n\nFor AttackAverager (attack 2) setting (%s, %s, %s, sample2use=%s, sampleK=%s, quantization=%s, dropout=%s):", self.datasetName, self.APIType, self.modelType, self.sample2use, self.sampleK, self.quantization, self.dropoutrate)
        logging.critical("\n\nThe real loss is: %s\n", ((torch.abs(inferedValues - testSet.features) * featureMask).sum() / featureMask.sum()).item() )
        
        logging.critical("\n\nThe total success rate is: %s\n", (featureMask.sum() / (featureMask.shape[0]*featureMask.shape[1])).item()  )
        
        logging.critical("\n\n<----------------- Finished for this AttackAverager TEST ----------------->\n\n") 
            
        
       
        