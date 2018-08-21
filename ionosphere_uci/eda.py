import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

data = pd.read_csv('ionosphere.csv', index_col=0)
n_rows, n_cols = data.shape

data_descript = data.describe()

data_descript.loc[:, ~(data_descript.sum(axis=0) > 351.0)]  ## checking for column wise sum to be equal to counts
data.drop('1', axis=1, inplace=True)

