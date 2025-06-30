# %% [markdown]
# Gruppenarbeit von:
# Lea Prinoth-Widauer, Annika Stampfl, Johanna Schindl, Linda Marie Potakowskyj und Max Zimmermann

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
# # Interpretation
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
# # Interpretation
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
## Interpretation
# Korrelation:
# r = +1: perfekte positive Korrelation
# r = -1: perfekt negative Korrelation
# r = 0: kein linearer Zusammenhang
# Sepal width und Sepal Length haben nichts mit einander zu tun. 
# Hingegen korrelieren Petal length und Petal width stark.
# %%
sns.pairplot(df, hue='species')
plt.show()

# %% [markdown]
## Interpretation
# Die Paarplot-Matrix zeigt die Verteilung und Beziehungen zwischen den verschiedenen Merkmalen.
# Es ist deutlich, dass die verschiedenen Iris-Spezies unterschiedliche morphologische Merkmale aufweisen.
# Setosa ist klar von den anderen beiden Arten getrennt!, während Versicolor und Virginica näher beieinander liegen und teilweise überlappen.

# %% [markdown]
# # Unsupervised Learning

# %%
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster import hierarchy

# %%
num_data = df.select_dtypes(include= "number")

# %%
Z = hierarchy.linkage(num_data)  
hierarchy.dendrogram(Z)

# %% [markdown]
# Das Dendrogramm zeigt die hierarchische Struktur der Cluster im Iris-Datensatz.
# Die Höhe der Verzweigungen repräsentiert die Distanz zwischen den Clustern.
# Je höher die Verzweigung, desto weiter entfernt sind die Cluster voneinander.
# Durch das Dendrogramm können wir die Anzahl der Cluster bestimmen, indem wir eine horizontale Linie ziehen, die die Cluster trennt.
# In diesem Fall können wir drei Cluster identifizieren, was mit der bekannten Verteilung der Iris-Spezies übereinstimmt.
# Setosa ist klar von den anderen beiden Arten getrennt, während Versicolor und Virginica näher beieinander

# %%
plt.figure(figsize=(10, 15))  # Größere Figur für bessere Lesbarkeit
hierarchy.dendrogram(
    Z,
    orientation='right',         # Blätter rechts
    labels=df['species'].values,  # Species als Labels
    leaf_font_size=8             # Kleinere Schriftgröße
)
plt.title('Dendrogramm des Iris-Datensatzes')
plt.xlabel('Distanz')
plt.ylabel('Proben')
plt.show()

# %% [markdown]
# Das Dendrogramm versucht schöner darzustellen bzw Lesbarkeit zu verbessern.


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
# Die Wahl von 3 Clustern, unterstützt durch das Dendrogramm, führt zu einer sinnvollen Gruppierung der Iris-Daten.
# Beide Algorithmen, KMeans und Agglomerative Clustering, erfassen erfolgreich die natürlichen Strukturen im Datensatz,
# wie die klare Trennung von Setosa und die teilweise Überlappung von Versicolor und Virginica zeigen. Die nahezu identischen
# Ergebnisse der beiden Methoden (bis auf drei Ausnahmen) unterstreichen die Robustheit der Clusterstruktur. Diese Interpretation bestätigt,
# dass die unüberwachte Clusteranalyse die inhärenten Muster in den Daten effektiv aufdeckt und mit der bekannten Verteilung der drei Spezies übereinstimmt.

# %% [markdown]
# # Supervised Learning

# %% [markdown]
# ## Daten für's Machine Learning vorbereiten (X/y, train/test)
# Zuerst laden wir die notwendigen Bibliotheken und bereiten unsere Daten vor. Wir trennen die Features (X), also die unabhängigen Variablen, von der Zielvariable (y), der Iris-Spezies. Anschließend teilen wir diese Daten in Trainings- und Testsets auf. Das Trainingsset wird zum Trainieren unserer Modelle verwendet, und das Testset dient zur unabhängigen Bewertung der Modellleistung.

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X = df[['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']]
y = df['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# %% [markdown]
# ## Zwei unterschiedliche Machine Learning Modelle trainieren
# Wir werden zwei verschiedene Klassifikationsalgorithmen trainieren, um ihre Leistung auf dem Iris-Datensatz zu vergleichen.

# %% [markdown]
# ### Modell 1: Logistische Regression
# Die Logistische Regression ist ein einfaches, aber leistungsstarkes und gut interpretierbares lineares Modell für Klassifikationsaufgaben.

# %%
log_reg_model = LogisticRegression(random_state=42, max_iter=200)
log_reg_model.fit(X_train, y_train)


# %% [markdown]
# ### Modell 2: Random Forest Classifier
# Der Random Forest ist ein Ensemble-Modell, das aus vielen einzelnen Entscheidungsbäumen besteht. Er ist oft leistungsfähiger als einzelne Modelle und weniger anfällig für Überanpassung (Overfitting).

# %%
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)


# %% [markdown]
# ## Evaluiert eure Modelle mit geeigneten Metriken
# Nachdem wir beide Modelle trainiert haben, machen wir Vorhersagen auf dem ungesehenen Testset, um ihre Leistung objektiv zu bewerten. Wir verwenden drei Metriken:
# 1.  **Accuracy (Genauigkeit):** Der prozentuale Anteil der korrekten Vorhersagen.
# 2.  **Classification Report:** Bietet eine detaillierte Aufschlüsselung nach Klassen, einschließlich Precision, Recall und F1-Score.
# 3.  **Confusion Matrix (Konfusionsmatrix):** Visualisiert die Leistung, indem sie korrekte und inkorrekte Vorhersagen für jede Klasse gegenüberstellt.

# %%
y_pred_log_reg = log_reg_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)


# %% [markdown]
# #### Auswertung der Logistischen Regression

# %%
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
report_log_reg = classification_report(y_test, y_pred_log_reg)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)

print("--- Metriken: Logistische Regression ---")
print(f"Accuracy: {accuracy_log_reg:.4f}")
print("\nClassification Report:\n", report_log_reg)


# %% [markdown]
# Visualisierung der Confusion Matrix für die Logistische Regression.

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues', 
            xticklabels=log_reg_model.classes_, yticklabels=log_reg_model.classes_)
plt.title('Confusion Matrix - Logistische Regression')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Tatsächliche Klasse')
plt.show()


# %% [markdown]
# #### Auswertung des Random Forest Classifiers

# %%
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print("--- Metriken: Random Forest Classifier ---")
print(f"Accuracy: {accuracy_rf:.4f}")
print("\nClassification Report:\n", report_rf)


# %% [markdown]
# Visualisierung der Confusion Matrix für den Random Forest Classifier.

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Tatsächliche Klasse')
plt.show()


# %% [markdown]
# ## Interpreattion
# Für diese spezifische Aufteilung der Daten hat das einfachere Modell der Logistischen Regression das komplexere Random-Forest-Modell knapp übertroffen.
# Dies kann in der Praxis vorkommen, insbesondere bei Datensätzen, die wie der Iris-Datensatz linear gut trennbar sind. Es unterstreicht die Wichtigkeit, 
# immer verschiedene Modelle zu testen und nicht anzunehmen, dass ein komplexeres Modell automatisch besser ist. Beide Modelle liefern jedoch sehr gute 
# Ergebnisse und sind für die Klassifikation der Iris-Pflanzen gut geeignet. Für den Botanischen Garten in Innsbruck bietet die Logistische Regression ein 
# einfaches und leistungsstarkes Werkzeug zur automatischen Identifikation.

# %%
