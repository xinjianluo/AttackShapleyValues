import logging
from aix360.algorithms.shap import KernelExplainer
from interpret.ext.blackbox import TabularExplainer
import torch 
import numpy as np
import time
import itertools
import math
import concurrent.futures
import random


import warnings
warnings.filterwarnings("ignore")

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
    
class ShapleyExplainer():
    def __init__(self, APIType, modelType, model, refDataset, refNum, classNum):
        # refDataset: torch.tensor
        logging.critical("\n[FUNCTION]: Creating ShapleyExplainer()......")
        
        logging.critical("Creating an explainer from API <%s>", APIType)
        self.APIType = APIType 

        refDataset = refDataset if refDataset.size(0) < refNum else refDataset[:refNum] 
        self.refNum = refNum
        self.shapleyValues = None 
        self.testDataset = None 

        if APIType == "Microsoft":
            self.explainer = MicrosoftExplainer(model, modelType, refDataset, classNum)
        elif APIType == "IBM":
            self.explainer = IBMExplainer(model, modelType, refDataset, classNum)
        elif APIType == "Vanilla":
            self.explainer = VanillaExplainer(model, modelType, refDataset, classNum)
        else:
            raise ValueError('Unsupported API type: {}.'.format(APIType))
            
    def saveShapley(self, datasetName, dataSplitType, modelType, suffix=""):
        svName = "shapley/{}_{}_{}_{}_Ref{}{}.sv".format(datasetName, self.APIType, dataSplitType, modelType, self.refNum, suffix)
        gtName = "shapley/{}_{}_{}_{}_Ref{}{}.gt".format(datasetName, self.APIType, dataSplitType, modelType, self.refNum, suffix)
        torch.save(self.shapleyValues, svName)
        torch.save(self.testDataset, gtName)
        logging.critical("Save shaple values to %s", svName)
        logging.critical("Save ground truth features to %s", gtName)
        return 0
        
    def saveModelOutput(self, datasetName, dataSplitType, modelType, dataArr):
        outName = "shapley/{}_{}_{}_{}_Ref1.out".format(datasetName, self.APIType, dataSplitType, modelType)
        torch.save(dataArr, outName)
        print("Save model outputs to", outName)
        return 0
        
    def computeShapley(self, testDataset, threads=4, sampleK=None):
        self.testDataset = testDataset.numpy()
        self.shapleyValues = self.explainer.computeShapley(testDataset, threads=threads, sampleK=sampleK)
        return self.shapleyValues
        
        
        
class IBMExplainer():
    def __init__(self, model, modelType, refDataset, classNum=None):
        # refDataset: torch.tensor
        logging.critical("\n[FUNCTION]: Creating IBMExplainer()......")
        
        refDataset = refDataset.numpy() 

        self.explainer = KernelExplainer(model.predict, refDataset)

        self.classNum = classNum
        self.modelType = modelType
      
    def computeShapley(self, testDataset, threads=4, sampleK=None):

        testDataset = testDataset.numpy()
        shapList = self.explainer.explain_instance(testDataset)
        
        if self.modelType != "NN" and self.classNum == 1:
            shapNdarray = np.expand_dims(shapList, axis=0)
        else:
            assert len(shapList) == self.classNum, "Mismatch between len(shap_values) and classNum"
            testNum, featureNum = shapList[0].shape
            
            shapNdarray = np.zeros((len(shapList), testNum, featureNum))
            for i, perClass in enumerate(shapList):
                shapNdarray[i] = perClass
                
        return shapNdarray
        
class MicrosoftExplainer():
    def __init__(self, model, modelType, refDataset, classNum=None):

        logging.critical("\n[FUNCTION]: Creating MicrosoftExplainer()......")

        if modelType in ("SVM", "GBDT", "RF"):
            self.explainer = TabularExplainer(model, refDataset.numpy())
        else:
            self.explainer = TabularExplainer(model, refDataset)
        self.classNum = classNum
        self.modelType = modelType
      
    def computeShapley(self, testDataset, threads=4, sampleK=None):

        local_explanation = self.explainer.explain_local(testDataset.numpy())
        
        sorted_local_importance_names = local_explanation.get_ranked_local_names()
        sorted_local_importance_values = local_explanation.get_ranked_local_values()
        rankedIndexes = np.array(sorted_local_importance_names)
        rankedShapleys = np.array(sorted_local_importance_values)
        if self.classNum == 1:
            rankedIndexes = np.expand_dims(rankedIndexes, axis=0)
            rankedShapleys = np.expand_dims(rankedShapleys, axis=0)
        
        rawShapleys = np.zeros_like(rankedShapleys)
        for i, perClass in enumerate(rankedIndexes):
            for j, perSample in enumerate(perClass):
                for k, perFeature in enumerate(perSample):
                    rawShapleys[i][j][perFeature] = rankedShapleys[i][j][k]

        return rawShapleys                
        
class VanillaExplainer():
    def __init__(self, model, modelType, refDataset, classNum=None):
        logging.critical("\n[FUNCTION]: Creating VanillaExplainer()......")
        
        self.refDataset = refDataset.numpy() 
        self.model = model 
        self.classNum = classNum
        self.modelType = modelType
      
    def computeShapley(self, testDataset, threads=4, sampleK=None):
        def getSampleRate(sampleK, featureNum):
            featureNum = featureNum - 1
            threshold = 2000
            if sampleK is None:
                ratio = threshold / (2 ** featureNum)   
                ratio = ratio if ratio < 1 else 1 
                return int(threshold), ratio 
            delta = 0.9
            m = np.log(2/delta) * (sampleK**2) / 2
            m = m if threshold > m else threshold
            ratio = m / (2 ** featureNum)   
            ratio = ratio if ratio < 1 else 1 
            return int(m), ratio 

        def getPrefixSubsets(n_features, feature2Exp, m):
            prefixSets = []
            for i in range(m):
                tempset  = np.random.permutation(n_features)
                tup = []

                for ele in tempset:
                    if ele == feature2Exp:
                        break
                    tup.append(ele)
         
                prefixSets.append(tuple(sorted(tup) ))  
            return prefixSets

        def getPrefixSubsetDict(n_features, m):
            subsetDict = {} 
            for feat in range(n_features):
                subsetDict[feat] = getPrefixSubsets(n_features, feat, m)
            return subsetDict

        def get_subset(n_features, feature_2_explain):
            subset = []
            for i in range(n_features):
                subset.append(i) if i != feature_2_explain else None
            return subset

        def replace_columns(matrix, vector, idx_lst):
            matrix[:, idx_lst] = vector[idx_lst]
        def get_replaced_data(training_data, data_row, combination, feature_2_explain):
            matrix_with_feature = np.copy(training_data)
            matrix_no_feature = np.copy(training_data)
            replace_columns(matrix_no_feature, data_row, list(combination))
            replace_columns(matrix_with_feature, data_row, list(combination + (feature_2_explain,)))
            return matrix_with_feature, matrix_no_feature

        def get_model_loss(refYhat, X, model):
                yhat = model(X)  
                return yhat - refYhat
        def get_diff_v(v1, v2):
            """ v1: (n_samples, n_outputs)
                v2: (n_samples, n_outputs)
            """
            # diff = (v1 - v2).sum(dim=0)
            diff = np.sum((v1 - v2), axis=0)
            # diff: torch.Size([n_outputs])
            diff = diff / len(v1)
            return diff  

        def compute_feature_shapley_per_sample(n_features, n_classes, training_data, refData, refYhat, mymodel, id2Explain, prefixSubsetDict):
            data_row =  training_data[id2Explain]    # the sample to be explained
            data_space = refData   # the data space for computing Shapley value

            n_samples = len(training_data)
                                                                                 
            phi_dict = np.zeros((n_features, n_classes))
            for feature_2_explain in range(n_features): 
                prefixSubsets = prefixSubsetDict[feature_2_explain]
                v_diff = 0 
                for pset in prefixSubsets:
                    matrix_with_feature, matrix_no_feature = get_replaced_data(data_space, data_row, pset, feature_2_explain) 
                    loss_with_feature = get_model_loss(refYhat, matrix_with_feature, mymodel) 
                    loss_no_feature = get_model_loss(refYhat, matrix_no_feature, mymodel) 
                    diff = get_diff_v(loss_with_feature, loss_no_feature)
                    v_diff += diff
               
                phi = v_diff / len(prefixSubsets)
     
                phi_dict[feature_2_explain] = phi

            if (id2Explain % 400 == 0 and id2Explain > 0) or id2Explain == (n_samples-1):
                # logging.critical("    -->> Explain Sample {}".format(id2Explain))
                logging.critical('>>> %s (%d %d%%) samples finished for explanation <<<' % (timeSince(start_outer, (id2Explain+1) / (n_samples+1)), id2Explain+1, (id2Explain+1) / (n_samples+1) * 100)) 
                                                                         
            return phi_dict
        
        if self.modelType in ("SVM", "GBDT"):
            if sampleK is None:
                sampleK = 50

        testDataset = testDataset.numpy()
        n_samples, n_features = testDataset.shape

        Phis = np.zeros((n_samples, n_features, self.classNum))
        
        
        
        m, samplingRatio = getSampleRate(sampleK, n_features)
        
        logging.critical("sampleK = %s", sampleK)
        logging.critical("m = %s", m)
        if len(self.refDataset) > 50:
            self.refDataset = self.refDataset[:50, :]
            
            
        logging.critical("self.modelType = %s", self.modelType)
        logging.critical("self.refDataset.shape = %s", self.refDataset.shape)
        logging.critical("testDataset.shape = %s", testDataset.shape)
        start_outer = time.time()
        refYhat = self.model.predict(self.refDataset)
        prefixSubsetDict = getPrefixSubsetDict(n_features, m)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # future_per_sample = {executor.submit(compute_feature_shapley_per_sample, n_features, self.classNum, testDataset, 
                                                 # self.refDataset, refYhat, self.model.predict, id2Explain, prefixSubsetDict): id2Explain for id2Explain in range(n_samples)}
            # for future in concurrent.futures.as_completed(future_per_sample):
                
                # id2Explain = future_per_sample[future]
                    
                # try:
                    # Phis[id2Explain] = future.result()
                # except Exception as exc:
                    # logging.critical('When explaining sample {} generated an exception: {}'.format(id2Explain, exc))
                    
        for id2Explain in range(n_samples):            
            Phis[id2Explain] = compute_feature_shapley_per_sample(n_features, self.classNum, testDataset, 
                                                 self.refDataset, refYhat, self.model.predict, id2Explain, prefixSubsetDict)
                                                 
            # logging.critical('>>> %s (%d %d%%) samples finished for explanation <<<' % (timeSince(start_outer, (id2Explain+1) / (n_samples+1)),
                                                                                 # id2Explain+1, (id2Explain+1) / (n_samples+1) * 100)) 
                    
        Phis = np.transpose(Phis, (2, 0, 1))  
                 
        logging.critical('<<<<<<<<<< %s finished for this program >>>>>>>>>>' % (timeSince(start_outer, 1)))


        return Phis        
              
