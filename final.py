# %%
import pandas as pd
from pandas import read_csv

import seaborn as sns
import matplotlib.pyplot as plt
import scipy


# %%
df = read_csv("iris.csv", index_col=0)

# %%
df

# %% [markdown]
# # Explorative Datenanalyse

# %% 
df.describe()

# %%
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['sepal_length_cm'], kde=True)
plt.title('Verteilung der Sepal-Länge')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['sepal_length_cm'])
plt.title('Boxplot der Sepal-Länge')

plt.show()
# %% [markdown]
# # Interpretation von 2 
# Die Hälfte der Sepal-Länge (aus beiden Diagrammen ablesbar) befindet sich zwischen ca. 5,2 und 6,5, der Median liegt bei ca. 5,8.\n",
# Der kleinste Wert liegt bei ca. 4,3 und der größte Wert bei ca. 7,9.
# Die wenigsten Blumen haben eine Länge von 7,1-7,5, wie man aus dem linken Diagramm ablesen kann.
# %%
print("Varianz:", df['sepal_length_cm'].var())
print("Standardabweichung:", df['sepal_length_cm'].std())

# %%
sns.scatterplot(data=df, x='sepal_length_cm', y='petal_length_cm', hue='species')
plt.title('Zusammenhang zwischen Sepal- und Petal-Länge')
plt.show()
# %% [markdown]
# # Interpretation von 3.
# Setosas haben eine kleine Sepal- als auch eine kleine Petal-Länge.
# Virginicas haben eine lange Sepal- als auch eine lange Petal-Länge. Es gibt jedoch einen Ausreißer, bei dem die Sepal-Länge kürzer ist als bei den anderen Virginicas.
# Die Versicolors liegen zwischen den Setosas und Verginicas, grenzen jedoch mehr an die Virginicas als an die Setosas.
# %%
corr = df.drop(columns=['species']).corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Korrelationsmatrix')
plt.show()
# %% [markdown]
## Interpretation von 4.
# Korrelation:
# r = +1: perfekte positive Korrelation
# r = -1: perfekt negative Korrelation
# r = 0: kein linearer Zusammenhang
# Sepal width und Sepal Length haben nichts mit einander zu tun. 
# Hingegen korrelieren Petal length und Petal width stark.
# %%
sns.pairplot(df, hue='species')
plt.show()
# %%

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

# %%
# KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster_kmeans'] = kmeans.fit_predict(num_data)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
df['cluster_agg'] = agg.fit_predict(num_data)

# Visualisierung
sns.scatterplot(data=df, x='sepal_length_cm', y='petal_length_cm', hue='cluster_kmeans', palette='Set2')
plt.title('KMeans Clustering')
plt.show()

sns.scatterplot(data=df, x='sepal_length_cm', y='petal_length_cm', hue='cluster_agg', palette='Set1')
plt.title('Agglomerative Clustering')
plt.show()
# %% [markdown]
# Bei 2 Clustern wäre es underfitted. Bei 4 Cluster wäre es overfitted.
# Deswegen haben wir uns bewusst für 3 Cluster entschieden.
# %% [markdown]
# # Supervised Learning

# %% [markdown]
# ## 1. Daten für Machine Learning vorbereiten (X/y, train/test)
# Zuerst laden wir die notwendigen Bibliotheken und bereiten unsere Daten vor. Wir trennen die Features (X) von der Zielvariable (y) und teilen diese dann in Trainings- und Testsets auf, um unsere Modelle später bewerten zu können.

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Features (X) und Zielvariable (y) definieren
# Wir verwenden alle numerischen Spalten außer 'species' als Features
X = df[['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']]
y = df['species']

# Daten in Trainings- und Testsets aufteilen
# test_size=0.3 bedeutet, dass 30% der Daten für das Testset verwendet werden
# random_state sorgt für reproduzierbare Ergebnisse
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# ## 2. Zwei unterschiedliche Machine Learning Modelle trainieren
