import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

# Citire date din csv
date_sol = pd.read_csv("Parametrii_sol2.csv", sep=";", low_memory=False)
print(date_sol.info())

# Verificare existenta valori NaN fiecare coloana
print(date_sol.isnull().sum())

# Eliminare valori NaN
date_sol.dropna(inplace=True)
print(date_sol.isnull().sum())

print(date_sol.describe())

# Pastram datele din perioada 03.08.2022 - 30.09.2022
date_sol = date_sol[date_sol.index.isin(range(0, 453))]

# Detectie valori aberante
date_sol.boxplot(['Soil temp. NPK:','Soil Potassium:'])
plt.title("Outlier detection")
plt.savefig("Outlier detection.png")
plt.show()

# Inlocuire valori aberante cu valori NULL
for x in ['Soil temp. NPK:','Soil Potassium:']:
    q75, q25 = np.percentile(date_sol.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    date_sol.loc[date_sol[x] < min,x] = np.nan
    date_sol.loc[date_sol[x] > max,x] = np.nan

# Verificare numar valori NULL pe fiecare coloana
print(date_sol.isnull().sum())

# Stergere valori NULL
date_sol = date_sol.dropna(axis = 0)

# Verificare numar valori NULL pe fiecare coloana
print(date_sol.isnull().sum())

## Clustering cu K-Means ##

X = date_sol[['Soil temp. NPK:', 'Soil Potassium:']]

# Determinarea numarului optim k de clustere, folosind 'parametrul silhouette'
range_n_clusters = range(2,11)
silhouette_avg = []
    
for nr_clusters in range_n_clusters:
    # Initializare Kmeans
    kmeans = KMeans(n_clusters= nr_clusters, n_init=10)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
 
    # Silhouette score
    silhouette_avg.append(silhouette_score(X, cluster_labels))
    
plt.plot(range_n_clusters,silhouette_avg)
plt.xlabel("Number of clusters") 
plt.ylabel("Silhouette score") 
plt.title("Silhouette analysis - Soil temp. NPK and Soil Potassium")
plt.savefig("Silhouette score - Soil temp. NPK and Soil Potassium.png")
plt.show()

fig, ax = plt.subplots(3, 3, figsize=(15,8))
a=1
for i in range(2,11):
    
    # Crearea cate unui caz KMeans pentru fiecare numar de clustere
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    
    # Crearea de cazuri SilhouetteVisualizer folosind cazurile KMeans
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=plt.subplot(3,3,a))
        
    # Fituire visualizer
    visualizer.fit(X)
    a=a+1

plt.suptitle("Silhouette visualizer - Soil temp. NPK and Soil Potassium")    
plt.savefig("Silhouette visualizer - Soil temp. NPK and Soil Potassium.png")
plt.show()
plt.clf()

# Aplicare PCA
pca = PCA()
X = pca.fit_transform(X)

# Rulare algoritm Kmeans cu 3 clustere
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=100,n_init=10, random_state=0)
ykmeans=kmeans.fit_predict(X) 

# Analiza performantelor algoritmului
print()
print("               # MODEL PERFORMANCE ANALYSIS #")
print()
print("Inertia: ", kmeans.inertia_)  
print("Silhouette score: ",silhouette_score(X, ykmeans)) 
print("Calinski-Harabasz Index", calinski_harabasz_score(X, ykmeans))
print("Davies-Bouldin Index", davies_bouldin_score(X, ykmeans))
print()
   
# Vizualizare clustere
plt.scatter(X[ykmeans==0, 0], X[ykmeans==0, 1], s=70, c='red', alpha=0.9)
plt.scatter(X[ykmeans==1, 0], X[ykmeans==1, 1], s=70, c='blue', alpha=0.9)
plt.scatter(X[ykmeans==2, 0], X[ykmeans==2, 1], s=70, c='green', alpha=0.9)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black',label='Centroizi')
plt.title("Clustering with K-Means - Soil temp. NPK and Soil Potassium")
plt.xlabel("Soil Potassium") 
plt.ylabel("Soil temp. NPK") 
plt.legend()
plt.savefig("Clustering with K-Means - Soil temp. NPK and Soil Potassium.png")
plt.show()
 


