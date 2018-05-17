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

### Transfer Learning on ImageNet Winning Conv Nets

Below are the ImageNet 1-crop error rates (224x224) for winning convolutional neural nets. I've implemented all but the SqueezeNet models in a transfer learning context.

| Network |	Top-1 error | Top-5 error |
| ------- | ----------- | ----------- |
| AlexNet |	43.45 |	20.91 |
| VGG-11 |	30.98 |	11.37 |
| VGG-13 |	30.07 |	10.75 |
| VGG-16 |	28.41 |	9.62 |
| VGG-19 |	27.62 |	9.12 |
| VGG-11 with batch normalization |	29.62 |	10.19 |
| VGG-13 with batch normalization |	28.45 |	9.63 |
| VGG-16 with batch normalization |	26.63 |	8.50 |
| VGG-19 with batch normalization |	25.76 |	8.15 |
| ResNet-18 |	30.24 |	10.92 |
| ResNet-34 |	26.70 |	8.58 |
| ResNet-50 |	23.85 |	7.13 |
| ResNet-101 |	22.63 |	6.44 |
| ResNet-152 |	21.69 |	5.94 |
| SqueezeNet 1.0 |	41.90 |	19.58 |
| SqueezeNet 1.1 |	41.81 |	19.38 |
| Densenet-121 |	25.35 |	7.83 |
| Densenet-169 |	24.00 |	7.00 |
| Densenet-201 |	22.80 |	6.43 |
| Densenet-161 |	22.35 |	6.20 |
| Inception v3 |	22.55 |	6.44 |


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