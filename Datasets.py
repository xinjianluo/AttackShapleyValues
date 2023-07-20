from torch.utils.data import Dataset
import numpy as np
import torch
import logging
import random 
import os 

def resetRandomStates(manualseed=47):
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)
        
class ExperimentDataset(Dataset):

    def __init__(self, datafilepath, classNum=1):
        full_data_table = np.genfromtxt(datafilepath, delimiter=',')
        data = torch.from_numpy(full_data_table).float()
        self.samples = data[:, :-1]
        self.labels = data[:, -1]
        
        self.feature_min, _ = self.samples.min(dim=0)
        self.feature_max, _ = self.samples.max(dim=0)
        denominator = self.feature_max - self.feature_min 
        denominator[denominator == 0] = 1 
        self.samples = (self.samples - self.feature_min)/denominator
        
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

        logging.critical("Creating dataset from %s, len(samples): %d, feature number: %d", datafilepath, len(self.labels), self.samples.shape[1])
                      
    def __len__(self):
        return len(self.samples)
    
    def get_class_weights(self):
        return self.classWeights
            
    def get_sample_weights(self):
        return self.sampleWeights

    def __getitem__(self, index):
        return self.samples[index], self.labels[index], self.sampleWeights[index]        
    
def extractDataTable(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x, y, w = next(iter(loader))
    return x, y, w 
    
def binizing(arr, binCount):
    maxPhi = 1
    minPhi = 0
    step = (maxPhi - minPhi) / binCount
    # print("bin step:", step)
    binNumbering = (arr // step) # reserve only int
    binedArray = binNumbering * step 
    binedArray = torch.clamp(binedArray, min=0, max=1)
    return binedArray
    
class FeatureInferenceDataset(Dataset):

    def __init__(self, path, datasetName, APIType, dataSplitType, modelType, sampleK=None, sample2use=-1, class2Use=0, smax=None, smin=None, quantization=-1, dropoutrate=-1):
        resetRandomStates()
        suffix = ""
        if sampleK is not None:
            suffix = "_SK{}".format(sampleK)
        if dropoutrate <= 0:
            svName = "{}{}_{}_{}_{}_Ref1{}.sv".format(path, datasetName, APIType, dataSplitType, modelType, suffix)
            gtName = "{}{}_{}_{}_{}_Ref1{}.gt".format(path, datasetName, APIType, dataSplitType, modelType, suffix)
        else:
            svName = "{}{}_{}_{}_{}_Ref1{}_dp{}.sv".format(path, datasetName, APIType, dataSplitType, modelType, suffix, dropoutrate)
            gtName = "{}{}_{}_{}_{}_Ref1{}_dp{}.gt".format(path, datasetName, APIType, dataSplitType, modelType, suffix, dropoutrate)
        shaps = torch.load(svName)
        features = torch.load(gtName)
        if sample2use>0:
            sample2use = sample2use if sample2use<len(features) else len(features)
        else: 
            sample2use = len(features)
        logging.critical("In %s, sample2use=%s, class2Use=%s", svName, sample2use, class2Use)
  
    
        if class2Use >=0:
            self.shaps = torch.tensor(shaps[class2Use])[:sample2use].float()
        else:
            self.shaps = torch.tensor(shaps).permute(1, 2, 0)[:sample2use].float()
        self.features = torch.tensor(features)[:sample2use].float()
#         smax, smin = self.shaps.max(), self.shaps.min()
    
        if smax is None:
            self.smax, _ = self.shaps.max(dim=0)
        else:
            self.smax = smax
        if smin is None:
            self.smin, _ = self.shaps.min(dim=0) 
        else:
            self.smin = smin 

        denominator = self.smax - self.smin 

        denominator[denominator == 0] = 1 

        self.shaps = (self.shaps - self.smin)/denominator
        self.featureNum = self.shaps.shape[1]
        
        if quantization > 0:
            self.shaps =  binizing(self.shaps, quantization)
       
    def __len__(self):
        return len(self.shaps)
    
    def get_feature_num(self):
        return self.featureNum
 
    def __getitem__(self, index):
        return self.shaps[index], self.features[index]    
    
