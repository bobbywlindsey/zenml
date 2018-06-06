from helpers import *

# iris dataframe
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
             columns=iris['feature_names'] + ['target'])


# test
all_columns = df.columns.values
preprocess = df_pipeline(df, [call(add_prefix('prefix'), 'sepal length (cm)'),
                              call(strip_whitespace, get_categorical_variable_names(df)),
                              call(replace_string_with_nan('prefix.1'), 'sepal length (cm)')])
print(preprocess.head())
