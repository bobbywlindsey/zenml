# TODO

* Find out how to read just a random sample from a CSV file into memory and put in function like `read_csv_random_sample(filename, sample_size)`

```python
import pandas as pd
import random
p = 0.01  # 1% of the lines
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
df = pd.read_csv(
         filename,
         header=0, 
         skiprows=lambda i: i>0 and random.random() > p
)
```

* Apply dictionary of values to pandas series. Something like:

```python
my_mapping = {'word': 'other word', 'word2': 'other word2'}

df.column.map(my_mapping)
```

* Parse date columns (code exists)

* Call something like `description(df)` which returns dataframe of descriptive statistics, nan value count, unique value count, etc...

* Correlation plot

* Create plotting functions in dash/plotly
* Handle missing values
