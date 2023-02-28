import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Citire date din csv
date_senzori = pd.read_csv("output_en29.11.2022.csv", low_memory=False)

print(date_senzori.info())

# Verificare existenta valori NaN fiecare coloana
print(date_senzori.isnull().sum())

# Eliminare valori NaN
date_senzori.dropna(inplace=True)
print(date_senzori.isnull().sum())

print(date_senzori.describe())

# Extragere coloana 'date'
col_date = date_senzori['date']

# Stergerea coloanelor 'date', 'Voltage', 'Voltage2'
date_senzori2 = date_senzori.drop(date_senzori.columns[[0, 23, 42]], axis=1)
print(date_senzori2.info())

# Salvarea DataFrame-ului sub forma de fisier csv
date_senzori2.to_csv('Date_senzori_noi.csv')

                       ## MATRICE DE CORELATIE PE TOT SETUL DE DATE ## 

# Construire matrice de corelatie
correlation_matrix = date_senzori2.corr()

# Obtinere masca pentru matrice de corelatie diagonala
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up pentru matplotlib figure
f, ax = plt.subplots(figsize=(30, 20))

# Desenare heatmap cu masca and aspect corect
sns_plot = sns.heatmap(correlation_matrix,
                       mask=mask,
                       annot=True,
                       fmt='.2f')
f.set_tight_layout(True)
f.suptitle("Correlation matrix - sensor data", fontsize=18)
f.savefig("Matrice de corelatie - date senzori.png")
f.show()

                ## MATRICE DE CORELATIE PARAMETRII HIPERSPECTRALI ##

# Matrice de corelatie 
figure(figsize=(16, 8), dpi=80)       
matr_corelatie= date_senzori2.iloc[:,18:22].corr()
sns.heatmap(matr_corelatie, annot=True)
plt.title("Correlation matrix - soil parameteres")
plt.savefig("Matrice de corelație - parametrii sol.png", bbox_inches="tight")
plt.show()

# Scalare valori
scaler = MinMaxScaler()
date_senzori2 = scaler.fit_transform(date_senzori2)
date_senzori2 = pd.DataFrame(date_senzori2)

# Adaugare coloana 'date'
date_senzori2.insert(0, 'date', date_senzori['date'].values)

# Readaugare nume coloane
date_senzori = date_senzori.drop(date_senzori.columns[[23, 42]], axis=1)
date_senzori2.columns = date_senzori.columns

                            ## CLUSTERING CU KMeans ##

# Selectare set date clustering
columns = ['Soil humidity NPK:', 'Soil temp. NPK:', 'Soil el. conductivity NPK:', 'Soil Potassium:']
X = pd.DataFrame(date_senzori2, columns=columns)
X.to_csv('Parametrii_sol.csv', encoding='utf-8')
X.to_excel('Parametrii_sol2.xlsx')



