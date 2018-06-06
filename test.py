from helpers import *

# iris dataframe
from sklearn.datasets import load_iris
import pandas as pd
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)


# test
all_columns = df.columns.values
preprocess = df_pipeline(df, [call(add_prefix('poop'), 'sepal length (cm)'),
                              call(strip_whitespace, all_columns),
                              call(replace_string_with_nan('poop5.1'), 'sepal length (cm)')])
print(preprocess.head())
