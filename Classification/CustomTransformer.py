from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

def add_transformations(*funcs):
    def decorator(transformer_class):
        class EnhancedTransformer(transformer_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._transformations = []
                
                for func in funcs:
                    name = func.__name__
                    if name.startswith('transform_'):
                        name = name[len('transform_'):]
                    
                    self.add_transformation(name, func)
            
            def add_transformation(self, name, func):
                self._transformations.append((name, func))
                # Create a method that passes self to the function
                setattr(self, f'apply_{name}', 
                        lambda X, transformer=self: func(X, transformer))
                return self
                
            def transform(self, X):
                result = X.copy()
                for name, func in self._transformations:
                    # Pass self as transformer parameter
                    result = func(result, self)
                return result
                
        return EnhancedTransformer
    return decorator