# suppress annoying warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
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

# import others
import IPython.display as ipd
import datarobot as dr
import itertools as it
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import requests
import hypertools as hyp
from collections import defaultdict
import operator
from sklearn.preprocessing import LabelEncoder

# import packages that deal with file formats
import os
import re
import shutil
import json
import json_tricks
from glob import glob
import bcolz
import itertools
import pickle

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

# configure datarobot API
current_directory = os.path.dirname(os.path.abspath(__file__))
dr.Client(config_path=current_directory + '/api_config.yaml')

# import database libraries
import sqlalchemy
from elasticsearch import Elasticsearch
import pymssql
