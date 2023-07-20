import torch 
import torch.nn as nn
import logging
from torch.utils.data import Dataset, DataLoader
from Datasets import extractDataTable
from Models import BlackModel_NN, FeatureInference_NN, ClassifierWrapper, RegressorWrapper
from sklearn.svm import SVC, SVR 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


        
class BlackBoxModelTrainer():
    def __init__(self, modeltype, trainset, testset, classNum, classWeights):
        super().__init__()
        logging.critical("\n[FUNCTION]: Creating BlackBoxModelTrainer()......")
        logging.critical("Creating a model for type %s", modeltype)

        if modeltype == "NN":
            if classNum == 1:
                self.trainer = NNRegressorTrainer(trainset, testset, classNum)
            else:
                self.trainer = NNClassifierTrainer(trainset, testset, classNum, classWeights) 
        else:
            if classNum == 1:
                self.trainer = SKLearnRegressorTrainer(trainset, testset, classNum, modeltype)
            else:
                self.trainer = SKLearnClassifierTrainer(trainset, testset, classNum, modeltype) 
      
        
    def train(self):
        self.model = self.trainer.train()
        return self.model 
        
class NNClassifierTrainer():
    def __init__(self, trainset, testset, classNum, classWeights):
        logging.critical("\n[FUNCTION]: Creating NNClassifierTrainer()......")
        x, _, _ = trainset[0]
        self.inputDim = x.shape[0]
        self.outputDim = classNum
        self.model = BlackModel_NN(self.inputDim, self.outputDim)
        logging.warning("Structure of NN Classifier: %s", self.model)
        self.trainset = trainset
        self.testset = testset
        self.classWeights = classWeights
       
    def train(self, epochs=6):
        def check_test_accuracy(mymodel, dataloader):
            mymodel.eval()
            accur = 0.0
            base = 0
            with torch.no_grad():
                for x, y, _ in dataloader:
                    yhat = mymodel(x)
                    accur += ( (yhat.argmax(dim=1)) == y ).sum()
                    base += x.shape[0]
            return accur / base
        
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.classWeights)
        
        test_interval = int(epochs / 5)
        self.model.train()

        for epoch in range(epochs):
            accurate = 0.0
            train_accur_base = 0.0
            for x, y, _ in trainloader: 
                optimizer.zero_grad()
                yhat = self.model(x)
                # print(yhat.dtype, y.dtype)
                loss = loss_fn(yhat, y)
                
                loss.backward()
                optimizer.step()
                accurate += ((yhat.argmax(dim=1))==y).sum()  
                train_accur_base += x.shape[0]
                
            if (epoch % test_interval == 0 and epoch > 0) or epoch == epochs - 1:
                # for each epoch, print information
                train = accurate / train_accur_base
                test = check_test_accuracy(self.model, testloader)
                self.model.train()
                logging.critical("In epoch {}, train accur is {}, test accur is {}.".format(epoch, train, test))
                
        return self.model 
        
class NNRegressorTrainer():
    def __init__(self, trainset, testset, classNum):
        logging.critical("\n[FUNCTION]: Creating NNRegressorTrainer()......")
        x, _, _ = trainset[0]
        self.inputDim = x.shape[0]
        self.outputDim = classNum
        self.model = BlackModel_NN(self.inputDim, self.outputDim)
        logging.warning("Structure of NN Regressor: %s", self.model)
        self.trainset = trainset
        self.testset = testset
       
    def train(self, epochs=1):
        def check_test_accuracy(mymodel, dataloader, loss_fn):
            mymodel.eval()
            accur = 0.0
            base = 0
            with torch.no_grad():
                for x, y, _ in dataloader:  
                    yhat = mymodel(x)
                    loss = loss_fn(yhat, y.float().unsqueeze(dim=1))
                    accur += loss.item()
                    base += 1
            return accur / base
        
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.L1Loss(reduction='mean')

        test_interval = 1 if epochs<5 else int(epochs / 5)
        
        self.model.train()

        for epoch in range(epochs):
            accurate = 0.0
            train_accur_base = 0.0
            for x, y, _ in trainloader: 
                optimizer.zero_grad()
                yhat = self.model(x)
                loss = loss_fn(yhat, y.float().unsqueeze(dim=1))
                
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
        
class SKLearnClassifierTrainer():
    def __init__(self, trainset, testset, classNum, modelType):
        assert classNum > 1, "classNum <= 1 in {}ClassifierTrainer".format(modelType)
        logging.critical("\n[FUNCTION]: Creating {}ClassifierTrainer()......".format(modelType))
        
        if modelType == "SVM":
            self.model = ClassifierWrapper(SVC(kernel='rbf', probability=True))
        elif modelType == "RF":
            self.model = ClassifierWrapper(RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5))
        elif modelType == "GBDT":
            self.model = ClassifierWrapper(GradientBoostingClassifier(n_estimators=100, max_depth=3))
        else:
            assert False, "Get a wrong model type {}".format(modelType)
            
        logging.critical("Creating model: {}".format(type(self.model.model)))
        self.modelType = modelType
        logging.warning("Parameters of %s: %s", type(self.model.model), self.model.get_params())
        
        self.trainset = trainset
        self.testset = testset
       
    def train(self):
        trainX, trainY, trainW = extractDataTable(self.trainset)
        
        if self.modelType == "SVM" and len(trainX) > 10000:
            trainX = trainX[:10000]
            trainY = trainY[:10000]
            trainW = trainW[:10000]
        self.model.fit(trainX, trainY, sample_weight=trainW)
        testX, testY, _ = extractDataTable(self.testset) 
        testAccuracy = self.model.score(testX, testY)
        logging.critical("The testing accuracy of the trained %s Classifier is %s", self.modelType, testAccuracy)
        return self.model        

class SKLearnRegressorTrainer():
    def __init__(self, trainset, testset, classNum, modelType):
        assert classNum == 1, "classNum != 1 in {}RegressorTrainer".format(modelType)
        logging.critical("\n[FUNCTION]: Creating {}RegressorTrainer()......".format(modelType))
        
        if modelType == "SVM":
            self.model = RegressorWrapper(SVR(kernel='rbf'))
        elif modelType == "RF":
            self.model = RegressorWrapper(RandomForestRegressor(n_estimators=100, max_depth=5))
        elif modelType == "GBDT":
            self.model = RegressorWrapper(GradientBoostingRegressor(n_estimators=100, max_depth=3))
        else:
            assert False, "Get a wrong model type {}".format(modelType)
        
        logging.critical("Creating model: {}".format(type(self.model.model)))
        self.modelType = modelType
        logging.warning("Parameters of %s: %s", type(self.model.model), self.model.get_params())
        
        self.trainset = trainset
        self.testset = testset
       
    def train(self):
        trainX, trainY, trainW = extractDataTable(self.trainset)
        if self.modelType == "SVM" and len(trainX) > 10000:
            trainX = trainX[:10000]
            trainY = trainY[:10000]
            trainW = trainW[:10000]
            
        self.model.fit(trainX, trainY, sample_weight=trainW)
        
        testX, testY, _ = extractDataTable(self.testset) 
        
        yhat = self.model.predict(testX)
        l1loss = ( np.sqrt((testY.numpy() - yhat)**2) ).sum() / len(yhat)
        logging.critical("The averaging L1 loss of the trained %s is %s", self.modelType, l1loss)
        return self.model  
