import torch 
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier

class BlackModel_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BlackModel_NN, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            #nn.Dropout(),
            nn.Sigmoid(),
            nn.Linear(input_dim*2, output_dim*2),
            #nn.Dropout(),
            nn.Sigmoid(),
            nn.Linear(output_dim*2, output_dim),
            #nn.Dropout(),
        )
      
    def predict(self, x):
        # x: numpy.ndarray
  
        x = torch.tensor(x)
        with torch.no_grad():
            yhat = self.forward(x)
        return yhat.numpy()
        
    def predict_proba(self, x):
        # x: numpy.ndarray
        return self.predict(x)
        
    def forward(self, x):
        return self.dense(x)
        
        
class ClassifierWrapper():
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, Y, sample_weight=None):
        return self.model.fit(X, Y, sample_weight=sample_weight)
        
    def predict(self, X):
        return self.model.predict_proba(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)  

    def get_params(self):
        return self.model.get_params()
        
    def score(self, X, Y):
        return self.model.score(X, Y) 

class RegressorWrapper():
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, Y, sample_weight=None):
        return self.model.fit(X, Y, sample_weight=sample_weight)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict(X)  

    def get_params(self):
        return self.model.get_params()
        
    def score(self, X, Y):
        return self.model.score(X, Y)        
        
class FeatureInference_NN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeatureInference_NN, self).__init__()
        self.Linear1 = nn.Sequential(
            nn.Linear(in_dim, in_dim*4), 
            nn.Sigmoid()
        )
        self.Output = nn.Sequential(
            nn.Linear(in_dim*4, out_dim), 
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.Linear1(x)
        out = self.Output(x)
        return out      
 
