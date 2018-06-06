# suppress warnings

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", module="theano")

# configure pandas

import pandas as pd
idx = pd.IndexSlice
digits = 4
pd.options.display.chop_threshold = 10**-(digits+1)
pd.options.display.float_format = lambda x: '{0:.{1}f}'.format(x, digits)
pd.options.display.show_dimensions = False

# configure numpy

import numpy as np
np.set_printoptions(precision=4, linewidth=100)

# configure seaborn

import seaborn as sns
sns.set(style="ticks", color_codes=True)

# configure matplotlib

import matplotlib.pyplot as plt
# %matplotlib notebook
# list all available styles: print(plt.style.available)
# plt.style.use('fivethirtyeight')
# plt.style.use('ggplot')
plt.style.use('seaborn-pastel')
plt.rc('figure', figsize=(10, 6))
