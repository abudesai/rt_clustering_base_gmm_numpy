#!/usr/bin/env python

import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 
import numpy as np, pandas as pd

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
from algorithm.model.clustering import ClusteringModel as Model, get_data_based_model_params
from algorithm.utils import get_model_config


# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(train_data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()     
    
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg) 
    # pprint.pprint(pp_params) 
    
    # preprocess data; returns a dictionary of X values, ids, and indexes of ids
    print('Preprocessing data ...')  
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    preprocessed_data = preprocess_pipe.fit_transform(train_data)    
    train_X, ids = preprocessed_data['X'].values.astype(np.float), preprocessed_data['ids']
    # print('train_X shape:',  train_X.shape)      
    
    # suggested # of clusters in schema
    num_clusters = data_schema["datasetSpecs"]["suggestedNumClusters"]   
    
    data_based_params = get_data_based_model_params(train_X)
    model_params = {**hyper_params, **data_based_params, "K": num_clusters }  
    
             
    model = Model(  **model_params )   
    print('Fitting model ...')  
    predicted_clusters = model.fit_predict(train_X)
    
    
    # return the prediction df with the id and prediction fields
    id_field_name = data_schema["inputDatasets"]["clusteringBaseMainInput"]["idField"]  
    preds_df = pd.DataFrame(ids, columns=[id_field_name])
    preds_df['prediction'] = predicted_clusters
    
            
    return preprocess_pipe, model, preds_df



