# Busqueda de los mejores Hiperparametros para los metodos MLP, DT y SVM en Batallas Grupales

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Lectura de Datos
tabla_combates_grupales = np.loadtxt('combates_grup_completo.csv',delimiter=',')

X_grupales = tabla_combates_grupales[:,:-1]
Y_grupales = tabla_combates_grupales[:,-1]

X_grup_train, X_grup_test, Y_grup_train, Y_grup_test = train_test_split(X_grupales, Y_grupales, test_size=0.25, random_state=42)

##### duplicamos los datos del train e invertimos los equipos de la segunda mitad por si hay sesgo
# 0:144 es el primer el equipo (el rango llega hasta 143) y 144:288 es el segundo equipo
X_grup_train_invertido= np.zeros(X_grup_train.shape)

X_grup_train_invertido[:,0:144] = X_grup_train[:,144:288]
X_grup_train_invertido[:,144:288] = X_grup_train[:,0:144]

X_grup_train= np.concatenate((X_grup_train,X_grup_train_invertido),axis=0)#concatena en por filas

Y_grup_train_invertido= 1 - Y_grup_train # si Y=0 queda 1 y si Y=1 queda 0

Y_grup_train = np.concatenate((Y_grup_train,Y_grup_train_invertido),axis=0)

X_grup_train, Y_grup_train = shuffle(X_grup_train, Y_grup_train, random_state=42) #desordenamos los datos

fk = KFold(n_splits=3)


##########################################################################
####################### Busqueda usando gridsearch #######################

parametros = [
    {
        'kernel': ['poly'],
        'C': [0.01,0.1,5],
        'gamma': [0.1, 0.5, 0.01],
        'degree': [2, 3]
    },
    {
        'kernel': ['sigmoid', 'rbf'],
        'C': [0.01,0.1,5],
        'gamma': [0.1, 0.5, 0.01],
    }
]

busqueda = GridSearchCV(
    estimator=SVC(),
    param_grid=parametros,
    cv=fk,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

busqueda.fit(X_grup_train, Y_grup_train)

print("Mejores hiperparámetros:")
print(busqueda.best_params_)

print("Mejor accuracy:")
print(busqueda.best_score_)

################### Busqueda de hiperparametros en MLPC ################################

parametros = [
    {
        'hidden_layer_sizes': [(20,40), (100, 20)],
        'activation': ['relu', 'tanh', 'logistic'],
        'learning_rate_init':[0.001,0.01,0.1,0.05,0.5],
        'learning_rate':['adaptive'],
        'solver':['adam','sgd'],
        'max_iter':[500,200,750,250,1000]
        # 'n_iter_no_change':[100] 
    }
]

busqueda = GridSearchCV(
    estimator=MLPClassifier(),
    param_grid=parametros,
    cv=fk,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

busqueda.fit(X_grup_train, Y_grup_train)

print("Mejores hiperparámetros:")
print(busqueda.best_params_)

print("Mejor accuracy:")
print(busqueda.best_score_)

################################################## Arbol de decisión ###############################################################
parametros = [
    {
        'criterion':['gini','entropy', 'log_loss'],
        'min_impurity_decrease': [0.0,0.01,0.1,0.05],
        
    }
]

busqueda = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=parametros,
    cv=fk,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

busqueda.fit(X_grup_train, Y_grup_train)

print("Mejores hiperparámetros:")
print(busqueda.best_params_)

print("Mejor accuracy:")
print(busqueda.best_score_)

arbol=DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0.0)
arbol.fit(X_grup_train,Y_grup_train)
print(arbol.get_depth())