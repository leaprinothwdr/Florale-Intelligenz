# %%
import pandas as pd
from pandas import read_csv

import seaborn as sb
import matplotlib as plt
import scipy


# %%
df = read_csv("iris.csv", index_col=0)

# %%
df

# %% [markdown]
# # Explorative Datenanalyse

# %% [markdown]
# # Unsupervised Learning

# %%
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster import hierarchy

# %%
num_data = df.select_dtypes(include= "number") # numerische Daten selektieren

# %%
Z = hierarchy.linkage(num_data)  
hierarchy.dendrogram(Z) # Dendogram erstellen

# %% [markdown]
# # Supervised Learning

# %% [markdown]
# test test test


