# Understanding Decorators for Custom Transformers

This guide explains how to use Python decorators to create flexible and customizable transformer classes for data preprocessing in machine learning pipelines.

## What Are Decorators?

Decorators are a special feature in Python that allow you to modify functions or classes without changing their source code. They use the `@` symbol followed by the decorator name placed above the function or class definition.

Think of decorators like wrapping a gift. The original function or class is the gift, and the decorator is the wrapping paper that adds new features or behavior.

## The Advanced Approach: Creating a Transformer with Multiple Transformations

Let's break down the code piece by piece to understand how it works:

### Step 1: Define the Decorator Factory

```python
def add_transformations(*funcs):
    def decorator(transformer_class):
        class EnhancedTransformer(transformer_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._transformations = []
                
                # Add all transformation functions
                for func in funcs:
                    # Get the function name without the 'transform_' prefix if it exists
                    name = func.__name__
                    if name.startswith('transform_'):
                        name = name[len('transform_'):]
                    
                    # Add the transformation function
                    self.add_transformation(name, func)
            
            def add_transformation(self, name, func):
                # Store the transformation
                self._transformations.append((name, func))
                
                # Add a method to apply just this transformation
                setattr(self, f'apply_{name}', lambda X: func(X))
                
                return self
                
            def transform(self, X):
                # Apply all transformations in sequence
                result = X.copy()
                for name, func in self._transformations:
                    result = func(result)
                return result
                
        return EnhancedTransformer
    return decorator
```

This code creates a decorator factory that takes multiple transformation functions as arguments and returns a decorator. The decorator then enhances a transformer class with the ability to apply these transformations.

#### Key Components:

1. **Decorator Factory**: `add_transformations(*funcs)` is a function that returns a decorator. It takes any number of transformation functions as arguments.

2. **Decorator**: The inner `decorator(transformer_class)` function takes a class and returns an enhanced version of it.

3. **Enhanced Class**: `EnhancedTransformer` is a subclass of the original transformer class that adds new methods and behavior.

4. **Initialization**: During initialization, it adds all the provided transformation functions to the transformer.

5. **Dynamic Method Creation**: It creates new methods for each transformation using `setattr()`.

6. **Transformation Application**: The `transform()` method applies all transformations in sequence.

### Step 2: Define Transformation Functions

```python
def transform_log(X):
    X_copy = X.copy()
    numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (X_copy[col] > 0).all():
            X_copy[col] = np.log(X_copy[col])
    return X_copy

def transform_scale(X):
    X_copy = X.copy()
    numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X_copy[col] = (X_copy[col] - X_copy[col].mean()) / X_copy[col].std()
    return X_copy
```

These are simple transformation functions that perform specific operations on the data:
- `transform_log`: Applies log transformation to numeric columns
- `transform_scale`: Standardizes numeric columns

### Step 3: Create a Transformer Class With the Decorator

```python
@add_transformations(transform_log, transform_scale)
class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name="multi_transformer"):
        self.name = name
    
    def fit(self, X, y=None):
        return self
```

Here, we define a basic transformer class and enhance it with our decorator, which adds the log and scale transformations.

### Step 4: Use the Transformer

```python
# Create an instance
transformer = MyTransformer()

# Apply individual transformations
X_logged = transformer.apply_log(X)
X_scaled = transformer.apply_scale(X)

# Or apply all transformations in sequence
X_transformed = transformer.transform(X)
```

After creating an instance of the enhanced transformer, you can:
1. Apply individual transformations using the dynamically created methods
2. Apply all transformations in sequence using the `transform()` method

## Why This Approach Is Powerful

1. **Reusability**: You can define transformation functions once and reuse them across different transformers.

2. **Modularity**: Each transformation is a separate function, making your code more organized and easier to test.

3. **Flexibility**: You can easily add or remove transformations without changing the base transformer class.

4. **Pipeline Integration**: The resulting transformers are compatible with scikit-learn pipelines.

## Complete Example with scikit-learn Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Decorator factory
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
                setattr(self, f'apply_{name}', lambda X: func(X))
                return self
                
            def transform(self, X):
                result = X.copy()
                for name, func in self._transformations:
                    result = func(result)
                return result
                
        return EnhancedTransformer
    return decorator

# Transformation functions
def transform_log(X):
    X_copy = X.copy()
    numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (X_copy[col] > 0).all():
            X_copy[col] = np.log(X_copy[col])
    return X_copy

def transform_scale(X):
    X_copy = X.copy()
    numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X_copy[col] = (X_copy[col] - X_copy[col].mean()) / X_copy[col].std()
    return X_copy

def transform_outliers(X, threshold=3):
    X_copy = X.copy()
    numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = X_copy[col].mean()
        std = X_copy[col].std()
        X_copy.loc[X_copy[col] > mean + threshold * std, col] = mean + threshold * std
        X_copy.loc[X_copy[col] < mean - threshold * std, col] = mean - threshold * std
    return X_copy

# Enhanced transformer
@add_transformations(transform_log, transform_scale, transform_outliers)
class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name="multi_transformer"):
        self.name = name
    
    def fit(self, X, y=None):
        return self

# Create sample data
data = {
    'A': [1, 2, 3, 100, 5],  # Contains an outlier
    'B': [10, 20, 30, 40, 50]
}
X = pd.DataFrame(data)
y = np.array([1, 2, 3, 4, 5])

# Create pipeline
pipeline = Pipeline([
    ('preprocess', MyTransformer()),
    ('model', LinearRegression())
])

# Fit and predict
pipeline.fit(X, y)
predictions = pipeline.predict(X)
```

## Conclusion

Using decorators to enhance your transformer classes gives you a powerful way to create reusable, modular, and flexible data preprocessing pipelines. This approach allows you to:

1. Define a base transformer template
2. Add custom transformation methods dynamically
3. Reuse transformations across different transformers
4. Integrate seamlessly with scikit-learn pipelines

As you become more comfortable with this pattern, you can create your own library of transformation functions and mix and match them as needed for different machine learning tasks.
