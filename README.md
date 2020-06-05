# ZenML

This repo contains Python libraries to aid in data science and machine learning tasks.

### Install

`pip install git+https://github.com/bobbywlindsey/zenml`

### Run Unit Tests

To test, run `pytest zenml/` or `python -m pytest zenml/` from the root directory.

### Preprocessing

Many functions support transformations of multiple columns at once.

```python
from zenml.preprocessing import *

df['my feature'] = replace_string(df['my feature'], 'value', 'new value')
df[['first feature', 'second feature']] = add_suffix(df[['first feature', 'second feature']], '_mysuffix')
df['cat feature'] = count_encode(df['cat feature'])
```

### Preprocessing with Pipelines

But you can also use these preprocessing functions in scikit-learn's pipeline.

```python
from zenml.utils           import pipelinize
from sklearn.pipeline      import Pipeline
from sklearn.impute        import SimpleImputer
from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import RobustScaler

# Pipeline for a particular feature
my_mapping = {' d1rty_valu3': 'value', 'enginEer': 'engineer'}
feature_pipeline = Pipeline([
    ('replace',    pipelinize(replace_string, 'value', 'new value')),
    ('removal',    pipelinize(remove_string, 'string to remove')),
    ('replacenan', pipelinize(replace_nan_with_string, 'missing')),
    ('prefix',     pipelinize(add_prefix, 'myprefix_')),
    ('suffix',     pipelinize(add_suffix, '_mysuffix')),
    ('mapping',    pipelinize(apply_map, my_mapping)),
])

# A more general pipeline for categorical variables
ordinal_mapping = {'small': 1, 'medium': 2, 'large': 3}
categorical_pipeline = Pipeline([
    ('replace nan with mean', SimpleImputer(strategy='constant', fill_value='missing')),
    # Choose thy encoding
    ('encode nominal', OneHotEncoder(handle_unknown='ignore')),
    ('encode ordinal', pipelinize(apply_map, ordinal_mapping)),
    ('count encoder', pipelinize(count_encode))
])

# A more general pipeline for numeric variables
numeric_pipeline = Pipeline([
    ('standardize variable', RobustScalar())
])

preprocess = ColumnTransformer([
    ('my feature pipeline', feature_pipeline, ['my feature']),
    ('categorical pipeline', categorical_pipeline, ['cat feature 1', 'cat feature 2'])
])

# Preprocess your data and with the option of 
# returning a data frame for additional inspection
print(fit_column_transformer(preprocess, df, to_df=True))
```

### Hypothesis Testing

Only two hypothesis tests are available at the moment: t-test (Student's, Welch's, and paired) and fisher's test.

As an example t-test, say you're looking at the Boston Housing dataset.

```python
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
boston_df = pd.DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df['PRICE'] = boston.target
boston_df.head()
```

Your hypothesis is that house prices in areas with lower student-teacher ratios are greater than house prices in an area with a higher student-teacher ratio.

```python
from zenml.hypothesis_tests import t_test

low_st_ratio = boston_df[boston_df.PTRATIO < 18].PRICE
low_st_ratio.reset_index(drop=True, inplace=True)
high_st_ratio = boston_df[boston_df.PTRATIO > 18].PRICE

alt_hypothesis = '>'
no_intervention_data = high_st_ratio
intervention_data = low_st_ratio

test_results = t_test(alt_hypothesis, intervention_data, no_intervention_data)
print(test_results)
```

```
{'Sample Size 1': 185, 'Sample Size 2': 316, 'Alt. Hypothesis': '>', 'Test Name': 'Welchs t-test', 'p-value': 5.19826947410151e-23, 'Test Statistic': 10.642892435040595, 'Effect Size': 1.0606056235342178, 'Power': 0.8641386288870567}
```

Or say you want to know if there's a significant difference in the clicks between site A and site B.

```python
from zenml.hypothesis_tests import fisher

site_a = pd.Series([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0])
site_b = pd.Series([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1])
no_intervention_data = site_a
intervention_data = site_b
alt_hypothesis = '!='

test_results = fisher(alt_hypothesis, intervention_data, no_intervention_data)
print(test_results)
```

```
{'Sample Size 1': 15, 'Sample Size 2': 17, 'Alt. Hypothesis': '!=', 'Test Name': 'Fishers Exact Test', 'p-value': 0.03195238826540317, 'Effect Size': 0.7951807228915662, 'Power': None}
```

### Feature Engineering

If you'd like to do a binary comparison between two variables and use the result as a new variable:

```python
from zenml.features import variable_match

variable_match = variable_match(df.variable_1, df.variable_2)
df.variable_match = variable_match
```

Or you could calculate the cosine similarity between two text variables:

```python
from zenml.features import cosine_similarity

cosine_sim_variable = cosine_similarity(df.variable_1, df.variable_2)
df.variable_cosine_sim = cosine_sim_variable
```

To create a ngram feature for a variable:

```python
from zenml.features import ngram_tf, ngram_idf_sum

ngram_tf_df = ngram_tf(2, .0025, .5, [df.variable])
bigram_idf_sum_variable = ngram_idf_sum(df.variable, ngram_tf_df, 2)
```

If you have a text field, you can use text embeddings like a continuous bag of words model:

```python
from zenml.features import word_embedding, cosine_similarity_text_embedding

# fine tune continuous bag of words model
ngram = 3
min_word_count = 10
workers = 20
epochs = list(range(2,3))
model_type = 'cbow'
hidden_layer_size = 300
initial_learning_rate = .9

cbow_model = word_embedding([df.variable_1, df.variable_2], ngram, min_word_count, epochs,
                             initial_learning_rate, workers, model_type=model_type)

# now calculate cosine similarity between the learned vector representations of the words
variable_cos_sim_cbow = cosine_similarity_text_embedding([df.variable_1, df.variable_2], cbow_model)
df.variable_cos_sim_cbow = variable_cos_sim_cbow
```

Or a skip gram model:

```python
from zenml.features import word_embedding, cosine_similarity_text_embedding

# fine tune skip gram model
ngram = 3
min_word_count = 10
workers = 20
epochs = list(range(2,3))
model_type = 'skipgram'
hidden_layer_size = 300
initial_learning_rate = .9

sg_model = word_embedding([df.variable_1, df.variable_2], ngram, min_word_count, epochs,
                           initial_learning_rate, workers, model_type=model_type)

# now calculate cosine similarity between the learned vector representations of the words
variable_cos_sim_skipgram = cosine_similarity_text_embedding(df.variable_1, df.variable_2, sg_model)
df.variable_cos_sim_skipgram = variable_cos_sim_skipgram
```

