# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 21:13:10 2020

@author: Yohana Delgado Ramos
"""

import pandas as pd
import numpy as np
import profiling_data as profiling
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno #
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import calendar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score


dataset = pd.read_csv('./dataset/online_shoppers_intention.csv')

info=dataset.info()
describe=dataset.describe()
head=dataset.head()
print(info)

#Explorando los datos
#Revision de datos faltantes

msno.matrix(dataset) # 

dataset.dropna(inplace=True)#Eliminar NAN del dataset. 

msno.matrix(dataset) # 

#Converimos los valores booleanos a enteros. 
dataset.Weekend = dataset.Weekend.astype(int)
dataset.Revenue = dataset.Revenue.astype(int)
print('No:',len(dataset[dataset.Revenue == 0]))
print('Yes:',len(dataset[dataset.Revenue == 1]))


#transformar los datos categoricos mes y tipo de visitante

# Replacing visitor_type to int
print(dataset['VisitorType'].unique())
dataset = dataset.replace({'VisitorType' : { 'New_Visitor' : 0, 'Returning_Visitor' : 1, 'Other' : 2 }})

# Coverting month column to numeric numeric values
df = dataset.copy()
monthlist = dataset['Month'].replace('June', 'Jun')
mlist = []
m = np.array(monthlist)
for mi in m:
    a = list(calendar.month_abbr).index(mi)
    mlist.append(a)
df['Month'] =  mlist
dataset=df

plt.figure(figsize=(18,8))
   
y = len(dataset[dataset.Revenue == 0]),len(dataset[dataset.Revenue == 1])

cat = ['No','Si']
plt.bar(cat,y,color = '#7bbdee')
   
   
plt.title('Valores de frecuencia Revenue')  


#Diagrama de bigotes, no se tendra en cuenta revenue 
color = '#0f4b78'
dataset.plot(kind='box', subplots=True, layout=(5,4), sharex=False, sharey=False, figsize=(14,14), 
                                        title='Gráfico de bigotes para cada atributo')
plt.savefig('shopping_box')
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(dataset.describe()[1:].transpose(),
            annot=True,linecolor="#0f4b78",
            linewidth=2,cmap=sns.color_palette("muted"))
plt.title("Resumen de atributos")
plt.show()

# Histogramas 
dataset.drop(['Revenue', 'Weekend'], axis=1).hist(bins=30, figsize=(14, 14), color='blue')
plt.suptitle("Histograma para cada atributo", fontsize=10)
plt.savefig('shopping_hist')
plt.show()


#Matrix de correlacion

#MODELOS DE CLASIFICACION


#EJEMPLO DE MODELO UNO  CON REGRESION LOGISTICA 



#Luego de familliarizarse con el conjunto de datos, se dividen en pruebas y entrenamiento 

# Variables a tener en cuenta para los datos de prueba 
atributos = ['Administrative', 'Administrative_Duration', 'Informational', 
        'Informational_Duration',  'ProductRelated', 
        'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 
        'Month', 'Region', 'TrafficType', 'VisitorType'] #Se excluye special day, no se muestra informacion relevante nu 
X = dataset[atributos] #Se toman unicamente las variables anteriores
y = dataset['Revenue'] #Obtengo el valor de revenue para poderlo predecir


#dividir datos entre de prueba y entrenamiento

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
scaler = MinMaxScaler() #Se usa minmaxScarler para tener las caracteristicas entre 0 y 1 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Uso de regresion logistica
logreg = LogisticRegression(solver='lbfgs',random_state = 0)
score = logreg.fit(X_train, y_train).decision_function(X_test)

#Evaluacion del modelo. 

y_pred = logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

# create heatmap

class_names=[False, True] # name  of classes
fig, ax = plt.subplots(figsize=(7, 6))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matrix de confusion')
plt.ylabel('Real')
plt.xlabel('Prediccion')
  
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
 
#Revision de metricasetricas individuales del modelo
from sklearn.metrics import precision_score, accuracy_score, f1_score,  recall_score 
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
print('f1_score: {:.2f}'.format(f1_score(y_test, y_pred)))


#import pickle
#pickle.dump(logreg, open("./model/linearmodel.pkl","wb"))
from joblib import dump

dump(logreg, filename="./model/text_classification.joblib")


