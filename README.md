# data-science

This repo contains Python libraries to aid in data science and machine learning tasks. Currently porting over TensorFlow deep learning modules to PyTorch.

To import the `helper` and `deeplearning` libraries along with other useful environmental objects, use the following:

```python
from helpers import *
from deeplearning.imports_and_configs import *
import deeplearning as dl
```

### Preprocessing

You can preprocess your pandas dataframe in a pipeline fashion:

```python
all_variable_names = df.columns.values
call_categorical_names = get_categorical_names(df)

tasks = [
    call(replace_nan_with_string(''), all_variable_names),
    call(strip_whitespace(), all_categorical_variable_names),
    call(remove_string(','), 'some_variable_name'),
    call(string_to_float(), ['other_variable_1', 'other_variable_2']),
    call(replace_string_with_nan(''), 'variable_with_nans')
]

df = df_pipeline(df, tasks)
```

### Feature Engineering

If you'd like to do a binary comparison between two variables and use the result as a new variable:

```python
variable_match = variable_match(df['variable_1'], df['variable_2'])
df = add_variable_to_df(variable_match, 'variable_match', df)
```

Or you could calculate the cosine similarity between two varaibles:

```python
cosine_sim_variable = cosine_similarity(df['variable_1'], df['variable_2'])
df = add_variable_to_df(cosine_sim_variable, 'variable_cosine_sim', df)
```

If you have a text field, you can use text embeddings like a continuous bag of words model:

```python
# fine tune continuous bag of words model
ngram = 3
min_word_count = 10
workers = 20
epochs = list(range(2,3))
model_type = 'cbow'
hidden_layer_size = 300
initial_learning_rate = .9

cbow_model = word_embedding([df['variable_1'], df['variable_2']], ngram, min_word_count, epochs,
                                initial_learning_rate, workers, model_type=model_type)

# now calculate cosine similarity between the learned vector representations of the words
variable_cos_sim_cbow = cosine_similarity_text_embedding([df['variable_1'], df['variable_2']], cbow_model)
df = add_variable_to_df(variable_cos_sim_cbow, 'variable_cos_sim_cbow', df)
```

Or a skip gram model:

```python
# fine tune skip gram model
ngram = 3
min_word_count = 10
workers = 20
epochs = list(range(2,3))
model_type = 'skipgram'
hidden_layer_size = 300
initial_learning_rate = .9

sg_model = word_embedding([df['variable_1'], df['variable_2']], ngram, min_word_count, epochs,
                                initial_learning_rate, workers, model_type=model_type)

# now calculate cosine similarity between the learned vector representations of the words
variable_cos_sim_skipgram = cosine_similarity_text_embedding(df['variable_1'], df['variable_2'], sg_model)
df = add_variable_to_df(variable_cos_sim_skipgram, 'variable_cos_sim_skipgram', df)
```

### Deep Neural Net

Split your training data

```python
train, test, train_labels, test_labels, classes = train_test_data(df, 'label')
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


First structure the data as such:

```
data/
  - train/
      - class_1 folder/
          - img1.png
          - img2.png
      - class_2 folder/
      .....
      - class_n folder/
  - dev/
      - class_1 folder/
      - class_2 folder/
      ......
      - class_n folder/
```

Show transformed images

```python
data_directory = 'image_directory'
data_transforms = dl.get_image_transform_rules(224)
dl.show_transformed_images(data_transforms, 5)
```

Specify parameters of the model you'd like to use. Choose any model from the following list:

Specify the model you'd like to use

```python
models_to_choose_from = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19']
model_name = 'vgg19'
```

Specify the parameters of the model

```python
freeze_params = {'freeze_all_layers': True, 'freeze_first_n_layrs': 0}
# decay learning rate every step_size epochs at a factor gamma
optimizer_params = {'step_size': 7, 'gamma': 0.1}
learning_rate_params = {'learning_rate': 0.001, 'momentum': 0.9}
num_epochs = 25
```

Train the model 

```python
dl.transfer_learn(model_name, data_directory, freeze_params, optimizer_params,
               learning_rate_params, num_epochs)
```

### More Traditional Models

#### Random Forest

```python
# split your data
train, test, train_labels, test_labels, classes = train_test_data(df, 'label',
                                                                  train_size=0.8,
                                                                  random_state=52)

# some hyperparamters to try
num_estimators = list(range(5, 10))
min_samples_leaves = list(range(1, 5))
max_depths = list(range(9, 10))

# create the parameter grid for a random forest to grid search for best parameters
param_grid = create_random_forest_param_grid(num_estimators, max_depths,
                                             min_samples_leaves, num_workers=42)

# autotune the model
rf = random_forest(train, test, train_labels, test_labels, param_grid)
```
