import configparser
from datetime import datetime
import os 
import torch
import numpy as np
import random
import logging
from Datasets import extractDataTable
from torch.utils.data import Dataset, DataLoader

from Datasets import ExperimentDataset
from ModelTrainer import BlackBoxModelTrainer

from Explainer import ShapleyExplainer

def getTimeStamp():
    return datetime.now().strftime("_%Y%m%d_%H_%M_%S")
    
def currentDir():
    return os.path.dirname(os.path.realpath(__file__))
    
def resetRandomStates(manualseed=47):
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)
    
def getSplittedDataset(trainpart, testpart, predictpart, expset):
    assert trainpart + testpart + predictpart == 1, 'Train + Test + Validation should be 1'
    x, y, _=expset[0]
    logging.critical("\n[FUNCTION]: Splitting dataset by getSplittedDataset()......")
    logging.warning("Display first (x, y) pair of dataset:\n %s, %s", x, y)
    logging.warning("Shape of (x, y): %s %s", x.shape, y.shape)
 
    train_len = int(len(expset) * trainpart)
    test_len = int(len(expset) * testpart)
    total_len = int(len(expset))
  
    trainset, remainset = torch.utils.data.random_split(expset, [train_len, total_len-train_len])
    logging.critical("len(trainset): %d", len(trainset))

    testset, predictset = torch.utils.data.random_split(remainset, [test_len, len(remainset)-test_len])
    logging.critical("len(testset): %d", len(testset))
    logging.critical("len(predictset): %d", len(predictset))
    return trainset, testset, predictset
    
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
    
def readConfigFile(configfile):
    parameters = {}
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(configfile)

    p_dataset = config['DATASET']
    parameters['datasetpath'] = currentDir() + os.sep + "datasets" + os.sep + p_dataset['DataFile'] + ".csv"
    parameters['dataset'] = p_dataset['DataFile']
    parameters['DatasetName'] = p_dataset['DataFile'].split(".")[0].capitalize()
    parameters['TrainPortion'] = p_dataset.getfloat('TrainPortion')
    parameters['TestPortion'] = p_dataset.getfloat('TestPortion')
    parameters['ValidationPortion'] = p_dataset.getfloat('ValidationPortion')

    
    p_default = config['DEFAULT']
    parameters['Model2Explain'] = p_default['Model2Explain']
    parameters['ExplainAPI'] = p_default['ExplainAPI']
    parameters['ComputeShapley'] = p_default.getboolean('ComputeShapley')
    
    # add timestamp and model info to the name of log file
    logfile = p_dataset['DataFile']
    index = logfile.rfind('.')
    prefix = "Generator_"
    if index != -1:
        logfile = prefix + logfile[:index] + "_{}" + getTimeStamp()
    else:
        logfile = prefix + logfile + "_{}" + getTimeStamp()
    logfile = (logfile + ".log").format(parameters['Model2Explain'])
    parameters['logpath'] = currentDir() + os.sep + "log" + os.sep + logfile
    
    p_explainer = config['EXPLAINER']
    parameters['ReferenceSamples'] = p_explainer.getint('ReferenceSamples')
    
    return parameters
    
if __name__=='__main__':  
    resetRandomStates()
    # read parameters from config file
    configfile = 'config.ini'
    parameters = readConfigFile(configfile)
        
    # init logging
    initlogging(parameters['logpath'])
    usedSamples = 1000
    
    logging.critical("Writing log to file: %s", parameters['logpath'])
    classDatasetDict = {"adult" : 2, "bank" : 2, "credit" : 2, "diabetes" : 3, "energy" : 1, "news" : 5}
    
    sampleK = None 
    suffix = ""
    if sampleK is not None:
        suffix = "_SK{}".format(sampleK)
    
    resetRandomStates()
    logging.critical("\n\n\n<<<<<<<-------------------------------For model: {}------------------------------->>>>>>>".format(parameters['Model2Explain']))

    # log config.ini content
    logging.warning("\n------------{} Start------------".format(configfile))
    with open(configfile, 'r') as conf: 
        logging.warning(conf.read())
    logging.warning("\n------------{} End------------".format(configfile))
        
    parameters['ClassNum'] = classDatasetDict[parameters['dataset']]
    logging.critical("<dataset = %s, ClassNum = %s>",  parameters['dataset'], parameters['ClassNum'])
    
    
    expset = ExperimentDataset(parameters['datasetpath'], classNum=parameters['ClassNum'])

    
    # split dataset and create dataloader
    trainset, testset, predictset = getSplittedDataset(parameters['TrainPortion'], parameters['TestPortion'], parameters['ValidationPortion'], expset) 
    
    modelTrainer = BlackBoxModelTrainer(parameters['Model2Explain'], trainset, testset, parameters['ClassNum'], expset.get_class_weights())
    blackBoxModel = modelTrainer.train()
    
    trainX, _, _ = extractDataTable(trainset)
    testX, _, _ = extractDataTable(testset)
    predictX, _, _ = extractDataTable(predictset)
    resetRandomStates()
    randomX = torch.rand(usedSamples, trainX.size(1))
    
    
    # if parameters['ClassNum'] == 1:
        # print(blackBoxModel.predict(testX.numpy()))
    # else:
        # print(blackBoxModel.predict_proba(testX.numpy()))
    
    explainer = ShapleyExplainer(parameters['ExplainAPI'], parameters['Model2Explain'], blackBoxModel, trainX, parameters['ReferenceSamples'], parameters['ClassNum'])
    
    for dataType in ("Test", "Validation", "Random"):
        if dataType == "Test":
            data2explain = testX[:usedSamples, :] 
        elif dataType == "Validation":
            data2explain = predictX[:usedSamples, :]
            explainer.saveModelOutput(parameters['DatasetName'], dataType, parameters['Model2Explain'], blackBoxModel.predict(data2explain.numpy()))
        elif  dataType == "Random":
            data2explain = randomX
        shapleyVales = explainer.computeShapley(data2explain, threads=3, sampleK=sampleK)
        explainer.saveShapley(parameters['DatasetName'], dataType, parameters['Model2Explain'], suffix=suffix)
        logging.critical("Shape of shapley values: %s", shapleyVales.shape)
        logging.critical("shapleyVales[0][0] = %s", shapleyVales[0][0])
        
    logging.critical("\n\n<----------------- Finished to explain {} ----------------->\n\n".format(parameters['Model2Explain']))
        
    logging.critical("\n\n<----------------- ALL Finished ----------------->\n\n")
