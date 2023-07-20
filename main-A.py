import configparser
from datetime import datetime
import os 
import torch
import numpy as np
import random
import logging
from Datasets import extractDataTable
from torch.utils.data import Dataset, DataLoader

from Datasets import FeatureInferenceDataset
from Models import FeatureInference_NN
from Attack import AttackRegressor, AttackAverager



def getTimeStamp():
    return datetime.now().strftime("_%Y%m%d_%H_%M_%S")
    
def currentDir():
    return os.path.dirname(os.path.realpath(__file__))
    
def resetRandomStates(manualseed=47):
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)
    
def initlogging(logfile):
    # debug, info, warning, error, critical
    # set up logging to file
    logging.shutdown()
    
    logger = logging.getLogger()
    logger.handlers = []
    
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=logfile,
                        filemode='w')
    
    # create console handler
    ch = logging.StreamHandler()
    # Sets the threshold for this handler. 
    # Logging messages which are less severe than this level will be ignored, i.e.,
    # logging messages with < critical levels will not be printed on screen
    # E.g., 
    # logging.info("This should be only in file") 
    # logging.critical("This shoud be in both file and console")
    
    ch.setLevel(logging.CRITICAL)
    # add formatter to ch
    ch.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(ch)  



    
def readConfigFile(configfile, suffix=None):
    parameters = {}
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(configfile)

    p_attack = config['ATTACK']
    parameters['Dataset'] = p_attack['Dataset']
    parameters['API'] = p_attack['API']
    parameters['Model'] = p_attack['Model']
    
    parameters['datasetpath'] = currentDir() + os.sep + "shapley" + os.sep
    
    # add timestamp and model info to the name of log file
    prefix = "Attacker_"
    logfile = (prefix + parameters['Dataset'] + "_{}_{}" + getTimeStamp() + "{}.log").format(parameters['API'], parameters['Model'], suffix)
    
    parameters['logpath'] =  currentDir() + os.sep + "log" + os.sep + logfile

    return parameters
    
if __name__=='__main__':  
    # read parameters from config file
    configfile = 'config.ini'
    
    class2UseDatasetDict = {"Adult" : 0, "Bank" : 0, "Credit" : 0, "Diabetes" : 2, "Energy" : 0, "News" : 1}
    sampleK = None  
    suffix = ""
    parameters = readConfigFile(configfile, suffix)
        
    # init logging
    initlogging(parameters['logpath'])
    
    class2Use = class2UseDatasetDict[parameters['Dataset']]
    # class2Use = 0
    logging.critical("<dataset = {}, class2Use = {}>".format(parameters['Dataset'], class2Use))

    sample2use = 1600 
    quantization = -1 
    dropoutrate = -1 

    if parameters['API'] == "Google":
        class2Use = 0 
    logging.critical("\n\n\n<<<<<<<-------------------------------For setting (%s, %s, %s, sample2use=%s, sampleK=%s)------------------------------->>>>>>>", parameters['Dataset'], parameters['API'], parameters['Model'], sample2use, sampleK)
    for attackType in (1, 2):
        if attackType == 1:
            # attack 1
            attackRegressor = AttackRegressor(parameters['datasetpath'], parameters['Dataset'], parameters['API'], parameters['Model'], sampleK=sampleK, sample2use=sample2use, class2Use=class2Use, quantization=quantization, dropoutrate=dropoutrate)
            attackRegressor.train(epochs=600)
            attackRegressor.test()
        elif attackType == 2:
            # attack 2
            attackAverager = AttackAverager(parameters['datasetpath'], parameters['Dataset'], parameters['API'], parameters['Model'], sampleK=sampleK, sample2use=sample2use, class2Use=class2Use, quantization=quantization, dropoutrate=dropoutrate)
            attackAverager.test()
    
    logging.critical("\n\n<----------------- ALL Finished ----------------->\n\n")
        
