import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
import scipy


# Citire date din csv
param_sol = pd.read_csv("Parametrii_sol_complet.csv", low_memory=False)
labels = pd.read_csv("Labels (1 col.) Plasmopara and Botrytis.csv", sep=';', low_memory=False)
      
param_sol.drop('Unnamed: 0', axis=1, inplace=True)

print(param_sol.info())

# Setare coloana 'Date' ca DatetimeIndex
param_sol.set_index(pd.to_datetime(param_sol['Date']), inplace=True)
labels.set_index(pd.to_datetime(labels['Date']), inplace=True)

param_sol = pd.merge(param_sol,labels, how='inner', left_index=True, right_index=True)

# Stergere coloana 'Date'
param_sol.drop(labels=['Date_x', 'Date_y'], axis=1, inplace=True)

# Matrice de corelatie ce include si etichete 
figure(figsize=(16, 8), dpi=80)       
matr_corelatie= param_sol.corr()
sns.set(font_scale=1.4)
sns.heatmap(matr_corelatie, annot=True, annot_kws={"size":13})
plt.xticks( rotation=45, ha='right') 
plt.title("Correlation matrix - soil parameteres and labels", fontsize=20)
plt.savefig("Correlation matrix - soil parameteres and labels.png", bbox_inches="tight")
plt.show()

# # Plot corelatii intre variabile
# sns.set()
# sns.pairplot(param_sol, kind='reg')

plt.figure(figsize=(15, 12))

params = {'axes.labelsize': 13,
          'axes.titlesize': 16,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}

pylab.rcParams.update(params)

for i, c in enumerate(param_sol.select_dtypes(include='number').columns):
    plt.subplot(5,2,i+1)
    sns.distplot(param_sol[c])
    plt.title('Distribution plot for field: ' + c)
    plt.xlabel('')   
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig("Distribution plots.png", bbox_inches="tight")

                        ## FEATURE SELECTION CU RANDOM FOREST ##
                                
X = param_sol.iloc[:, 0:6]
y = param_sol.iloc[:, 6:8]

# Impartire date in seturi de antrenare si testare
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.30)

# SelectFromModel - selectare predictori cu importanta mai mare decat 
# media importantelor tuturor predictorilor
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100)) # numar arbori
sel.fit(X_train, y_train)

print(pd.DataFrame(X.columns))

# Afisare predictori selectati
selected_feat= X_train.columns[(sel.get_support())]
print(selected_feat)

# Plot importanta predictori
importances = sel.estimator_.feature_importances_
indices = np.argsort(importances)[::-1] 
figure(figsize=(14, 5), dpi=80)
plt.figure()
plt.title("Feature importance", fontsize=20)
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], fontsize=16, rotation=45, ha='right')
plt.yticks(fontsize=16)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Feature importance", fontsize=18)
plt.savefig("Feature importance.png", bbox_inches="tight")
plt.show()

                                ## RESAMPLING ##
                                
from sklearn.utils import resample
healthy = param_sol[param_sol['Plasmopara viticola & Botrytis cinerea'] == 0]
plasmopara = param_sol[param_sol['Plasmopara viticola & Botrytis cinerea'] == 1]
botrytis = param_sol[param_sol['Plasmopara viticola & Botrytis cinerea'] == 2]

# Upsample clase minoritare
plasmopara_upsampled = resample(plasmopara, 
                                replace=True,     
                                n_samples=1160)

botrytis_upsampled = resample(botrytis, 
                              replace=True,     
                              n_samples=1160)

# Combine minority class with downsampled majority class
date = pd.concat([healthy, plasmopara_upsampled, botrytis_upsampled])

                            ## CLASSIFICATION ##
                        
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

X = date.iloc[:, [1,2]]
y = date.iloc[:, 6:8] 

# Afisarea distributiei variabilei target
print(y.value_counts())  

columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=12)

regex = re.compile(r"\[|\]|<", re.IGNORECASE)

X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]
X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

clf = []
cm = []
y_proba = []
nume_clf= ["GradientBoostingClassifier", "LGBMClassifier", "XGBClassifier", "RandomForestClassifier", "KNeighborsClassifier"]
# Definire etichete pentru clase
class_names = ['Healthy', 'Plasmopara', 'Botrytis']

for i in range(0,5):
    
    #Creare model
    if i==0: 
        classifier = GradientBoostingClassifier(n_estimators=90, learning_rate=1, max_features=2, max_depth=4, random_state=0)
        
    elif i==1:
        params = {'n_estimators':[60],
                  'learning_rate': [0.02],
                  'max_depth': [15],
                  'num_leaves': [50],
                  'is_unbalance':["+"],
                  'boosting':['dart']}

        params = list(ParameterSampler(params, n_iter=2000))

        store = []
        for j in range(len(params)):
            p = params[j]
            classifier = LGBMClassifier(**p, objective='F1')
            score=cross_val_score(classifier, X_train, y_train, cv=10)

            store.append({'parameters':p,'Score':score})
            print(score)
        
        store=pd.DataFrame(store)
        
    elif i==2:
        classifier= XGBClassifier() 
        
    elif i==3:
        classifier= RandomForestClassifier()
        
    else:
        classifier= KNeighborsClassifier()
        
    # Antrenare model
    classifier.fit(X_train, y_train.values.ravel())
    
    # Predictie valoare clasa
    y_pred= classifier.predict(X_test)
    
    # Predict probabilities of each class for test set
    y_proba.append(classifier.predict_proba(X_test))
    
    # Calculare matrice de confuzie si adaugare in lista
    cm.append(confusion_matrix(y_test, y_pred))
    
    # Afisare raport de clasificare
    print()
    print("Classification report: "+ nume_clf[i])
    print(classification_report(y_test, y_pred)) 

    
# Generare matrice de confuzie
# Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(17, 22))

axs = axs.flatten()

# Plot each confusion matrix in a subplot
i = 0
for ax in axs[:-1]:
    im = ax.imshow(cm[i], cmap='Blues')
    for j in range(cm[i].shape[0]):
        for k in range(cm[i].shape[1]):
            color = 'black' if cm[i][j, k] < cm[i].max() / 2 else 'white'
            text = ax.text(k, j, cm[i][j, k], ha='center', va='center', color=color, fontsize=16)
    ax.set_title(nume_clf[i], fontsize=20)
    ax.set_xlabel('Predicted label', fontsize=19)
    ax.set_ylabel('True label', fontsize=19)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(class_names, fontsize=19)
    ax.set_yticklabels(class_names, fontsize=19)
    ax.grid(False)
    i += 1
    
# Add a title and show the plot
fig.suptitle('Confusion matrices for multiple classifiers', fontsize=21, y=0.95)
fig.tight_layout(pad=2.2)
plt.subplots_adjust(top=0.90)
plt.savefig("Confusion matrices for multiple classifiers.png", bbox_inches="tight")
plt.show()

# Create a 3x2 grid of subplots
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(17, 22))

axs = axs.flatten()

# Plot each confusion matrix in a subplot
i = 0
for ax in axs[:-1]:
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(3):
        fpr[j], tpr[j], _ = roc_curve(y_test == j, y_proba[i][:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])

    # Plot ROC curve for each class
    ax.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='Healthy (area = %0.2f)' % roc_auc[0])
    ax.plot(fpr[1], tpr[1], color='blue', lw=2, label='Plasmopara (area = %0.2f)' % roc_auc[1])
    ax.plot(fpr[2], tpr[2], color='green', lw=2, label='Botrytis (area = %0.2f)' % roc_auc[2])
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate', fontsize=19)
    ax.set_ylabel('True Positive Rate', fontsize=19)
    ax.set_title(nume_clf[i], fontsize=20)
    ax.legend(loc="lower right", fontsize=16)
    ax.grid(False)
    i += 1
      
# Add a title and show the plot
fig.suptitle('Receiver Operating Characteristic (ROC) Curve for multiple classifiers', fontsize=21, y=0.95)
fig.tight_layout(pad=2.2)
plt.subplots_adjust(top=0.90)
plt.savefig("Receiver Operating Characteristic (ROC) Curve for multiple classifiers.png", bbox_inches="tight")
plt.show()

