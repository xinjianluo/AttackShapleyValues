# Feature Inference Attack on Shapley Values

## Overview
This repository contains the code to run the experiments present in this paper: [Feature Inference Attack on Shapley Values](https://dl.acm.org/doi/abs/10.1145/3548606.3560573). 
1. This code repository is for *off-line* running and suitable for three platforms:
Microsoft Azure Machine Learning [1], IBM Research Trusted AI [3], and the Vanilla method.
1. Google Cloud AI platform [2] only supports the on-line mode: the users need to manually create a Google Cloud account, train a model, deploy the model to the virtual machines of Google Cloud, send queries for predictions and explanations, which are trivial and thus excluded from this repository. Interested readers can refer to `google-generator.py` for an overview of the code performed on Google Cloud.
1. The experiments in the paper are performed in on-line mode on Microsoft and Google platforms and in off-line mode on IBM and Vanilla platforms.


## How to run
### Step 1: Install dependencies
Create a new virtual environment in Anaconda:

    conda create --name expenv python=3.6
    conda activate expenv

Install necessary libs:

    pip install aix360==0.2.1
    pip install azureml-interpret==1.39.0



### Step 2: Configure the runtime parameters
Before running these attacks, you can configure the parameters of datasets, model types, and platforms in `config.ini`. The key names in this configure file are self-explanatory.

  
### Step 3: Generate Shapley values

    python main-G.py


### Step 4: Perform attacks

    python main-A.py


### Note
1. The generated Shapley values are saved in the folder: `shapley/`.
1. The running logs are saved in the folder: `log/`.
1. We provide a toy dataset `bank.csv` in the folder `datasets/` for debugging purpose.

## Citation
If you use our results or this codebase in your research, then please cite this paper:
```
@inproceedings{luo2022shapley,
    title={Feature inference attack on shapley values},
    author={Luo, Xinjian and Jiang, Yangfan and Xiao, Xiaokui},
    booktitle={Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security},
    pages={2233--2247},
    year={2022}
}

```

## Reference
[1] Azure Machine Learning Documentation. 2023. Model Interpretability. [https://learn.microsoft.com/en-sg/azure/machine-learning/how-to-machine-learning-interpretability?view=azureml-api-2](https://learn.microsoft.com/en-sg/azure/machine-learning/how-to-machine-learning-interpretability?view=azureml-api-2). Online; accessed 20-July-2023. 

[2] Google Cloud AI Platform Documentation. 2023. Introduction to AI Explanations for AI Platform. [https://cloud.google.com/ai-platform/prediction/docs/ai-explanations/overview](https://cloud.google.com/ai-platform/prediction/docs/ai-explanations/overview). Online; accessed 20-July-2023. 

[3] Trusted AI. 2023. AI Explainability 360. [https://github.com/Trusted-AI/AIX360](https://github.com/Trusted-AI/AIX360). Online; accessed 20-July-2023. 



