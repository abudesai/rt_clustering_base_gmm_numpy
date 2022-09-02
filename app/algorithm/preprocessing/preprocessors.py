
import numpy as np, pandas as pd
import sys 
from sklearn.base import BaseEstimator, TransformerMixin


class TypeCaster(BaseEstimator, TransformerMixin):  
    def __init__(self, vars, cast_type):
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type
        
    def fit(self, X, y=None): return self
        

    def transform(self, data):  
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns] 
        for var in applied_cols: 
            data[var] = data[var].apply(self.cast_type)
        return data



class FloatTypeCaster(TypeCaster):  
    ''' Casts float features as object type if they are not already so.
    This is needed when some categorical features have values that can inferred as numerical.
    This causes an error when doing categorical feature engineering. 
    '''
    def __init__(self, num_vars):
        super(FloatTypeCaster, self).__init__(num_vars, float)



class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns, selector_type='keep'):
        self.columns = columns
        self.selector_type = selector_type
        
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):   
        
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]  
            X = X.drop(dropped_cols, axis=1)      
        else: 
            raise Exception(f'''
                Error: Invalid selector_type. 
                Allowed values ['keep', 'drop']
                Given type = {self.selector_type} ''')   
        return X
    
    

class XSplitter(BaseEstimator, TransformerMixin): 
    def __init__(self, id_col):
        self.id_col = id_col
        self.ids = None
        self.ids2idx = {}
        self.idx_col = "__idx__"
    
    def fit(self, data): 
        self.ids = data[[self.id_col]].drop_duplicates()   
        self.ids[self.idx_col] = np.arange(self.ids.shape[0])
        return self
    
    def transform(self, data):        
        X_cols = [ col for col in data.columns if col != self.id_col ]  
        data2 = data.merge(self.ids, on=[self.id_col])  
        return { 'X': data2[X_cols], 'ids': data2[self.id_col]   , "idxs": data2[self.idx_col] }
