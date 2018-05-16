# data-science

This repo contains Python libraries to aid in data science and machine learning tasks. Currently porting over TensorFlow deep learning modules to PyTorch.

To import the `helper` and `deeplearning` libraries along with other useful environmental objects, use the following:

```python
from helpers.imports_and_configs import *
import helpers as dsh
from deeplearning.imports_and_configs import *
import deeplearning as dl
```


### Deep Neural Net

Split your training data

```python
train, test, train_labels, test_labels, classes = dsh.train_test_data(df, 'label')
```

Train a deep neural network with two hidden layers with 25 neurons for the first layer and 12 for the second

```python
train_neural_net = dl.deep_neural_net(train, train_labels, test, test_labels, 
                                     [25, 12], num_epochs=1500)
```

Then test it

```python
dl.test_deep_neural_net(train_neural_net, test, test_labels)
```


### Functions depending on Jupyter Notebook


```python
%matplotlib inline
def visualize_missing_data(df):
    """ returns visualization of completeness of each row of the data from most incomplete to most complete"""
    %matplotlib inline
    sorted_df = msno.nullity_sort(df, sort='ascending')
    # diplay table for missing percentages
    num_rows = df.shape[0]
    percent_missing = {}
    for column_name in df.columns.values:
        num_missing = df[column_name].isnull().sum()
        percent_missing[column_name] = (num_missing / num_rows) * 100
    percent_missing_df = pd.DataFrame({'% missing': pd.Series(percent_missing)})
    display(percent_missing_df)
    # return visualization
    return msno.matrix(sorted_df)
```