
import numpy as np, pandas as pd
import joblib
from scipy.stats import multivariate_normal
import sys
import os, warnings

warnings.filterwarnings('ignore')  

model_fname = "model.save"

MODEL_NAME = "clustering_base_gmm_numpy"


class ClusteringModel:
    """A Bayesian Gaussian mixture model.
    
    Assumes Gaussians' variances in each dimension are independent.
    
    Parameters
    ----------
    K : int > 0
        Number of mixture components.
    D : int > 0
        Number of dimensions.        
        
    Source code from here (slightly modified): https://github.com/lazyprogrammer/machine_learning_examples/blob/master/unsupervised_class/gmm.py
    """
    
    
    def __init__(self, D, K, **kwargs) -> None:
        self.D = D
        self.K = K  
        self.mu = np.zeros((self.K, self.D))
        self.cov = np.zeros((self.K, self.D, self.D))
        self.pi = np.ones(K) / K # uniform    
        
        
        
    def fit(self, X, max_epochs=100, verbose=False): 
        
        # for early stopping
        min_improvement_in_ll = 0.1
        max_epochs_without_improvement = 5
        
        N = X.shape[0]
        
        self.memberships = np.zeros((N, self.K))
        
        for k in range(self.K):
            self.mu[k] = X[np.random.choice(N)]
            self.cov[k] = np.eye(self.D) 
    
        log_likelihoods = []
        smoothing=1e-3
                
        weighted_pdfs = np.zeros((N, self.K)) # we'll use these to store the PDF value of sample n and Gaussian k
        
        num_epoch_since_last_improvement = 0
        for i in range(max_epochs):
                        
            # expectation
            for k in range(self.K):
                weighted_pdfs[:,k] = self.pi[k] * multivariate_normal.pdf(X, self.mu[k], self.cov[k])
            
            self.memberships = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)
                        
            # maximization
            for k in range(self.K):               
                
                Nk = self.memberships[:,k].sum()
                # maximization - mu
                self.pi[k] = Nk / N
                self.mu[k] = self.memberships[:,k].dot(X) / Nk

                ## maximization - cov
                delta = X - self.mu[k] # N x D
                Rdelta = np.expand_dims(self.memberships[:,k], -1) * delta # multiplies R[:,k] by each col. of delta - N x D
                self.cov[k] = Rdelta.T.dot(delta) / Nk + np.eye(self.D)*smoothing # D x D
            
            # expectation (log-likelihood)
            log_likelihood = np.log(weighted_pdfs.sum(axis=1)).sum()
            log_likelihoods.append(log_likelihood)
            if verbose: print(f"epoch: {i}, log_likelihood : {log_likelihood}")
            if i > 0:
                if np.abs(log_likelihoods[i-1] - log_likelihoods[i]) > min_improvement_in_ll:
                    num_epoch_since_last_improvement = 0
                else: 
                    num_epoch_since_last_improvement += 1
                    
                    if num_epoch_since_last_improvement >= max_epochs_without_improvement:
                        if verbose: print("Stopping due to no significant improvement")
                        break            
            
        return np.array(log_likelihoods)
    
    
    def predict_proba(self, X): 
        N = X.shape[0]
        weighted_pdfs = np.zeros((N, self.K))
        for k in range(self.K):
            weighted_pdfs[:,k] = self.pi[k] * multivariate_normal.pdf(X, self.mu[k], self.cov[k])

        memberships = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)
        return memberships
    
    
    def predict(self, X): 
        memberships = self.predict_proba(X)  
        predictions = np.argmax(memberships, axis=1)     
        return predictions
    
    
    def fit_predict(self, X): 
        self.fit(X) 
        preds = self.predict(X)
        return preds
        
    
    def evaluate(self, x_test): 
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))


    @classmethod
    def load(cls, model_path):         
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = ClusteringModel.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def get_data_based_model_params(data): 
    return { "D": data.shape[1] }
