# Implementación de metodos de aprendizaje MLP, DT y SVM con k-fold de 3
# + Prueba de los mejores Hiperparametros encontrados Para Batallas Individuales

import numpy as np
import time as tm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Lectura de Datos
tabla_combates_individuales = np.loadtxt('combates_indv_completo.csv',delimiter=',')

X_indviduales = tabla_combates_individuales[:,:-1]
Y_indviduales = tabla_combates_individuales[:,-1]

X_indv_train, X_indv_test, Y_indv_train, Y_indv_test = train_test_split(X_indviduales, Y_indviduales, test_size=0.25, random_state=42)

#invertimos y concatenamos los datos del train
# [0:24] primer pokemon, [24:48] segundo pokemon
X_indv_train_invertido= np.zeros(X_indv_train.shape)

X_indv_train_invertido[:,0:24] = X_indv_train[:,24:48]
X_indv_train_invertido[:,24:48] = X_indv_train[:,0:24]

X_indv_train= np.concatenate((X_indv_train,X_indv_train_invertido),axis=0)#concatena en por filas


Y_indv_train_invertido= 1 - Y_indv_train # si Y=0 queda 1 y si Y=1 queda 0
Y_indv_train = np.concatenate((Y_indv_train,Y_indv_train_invertido),axis=0)

X_indv_train, Y_indv_train = shuffle(X_indv_train, Y_indv_train, random_state=42) #desordenamos los datos


fk = KFold(n_splits=3)

# # Constantes MLP
eta = 0.05
epocas = 1000
tol = 0.95
capas=(64,32)


vector_aciertos_MLP = []
vector_aciertos_AD = []
vector_aciertos_SVC = []

for i,(index_train,index_test) in enumerate(fk.split(X_indv_train)):
    print(f'Kfol = {i}')

    # --- 1. Perceptrón Multicapa (MLP) ---
    tini = tm.perf_counter()
    modelo = MLPClassifier(hidden_layer_sizes=capas,activation='logistic',solver='sgd',learning_rate_init=eta,max_iter=epocas,
                            tol=tol, verbose=False) #para que no muestre en consola todo el proceso
    
    modelo.fit(X_indv_train[index_train],Y_indv_train[index_train])
    vector_aciertos_MLP.append(modelo.score(X_indv_train[index_test],Y_indv_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion MLP = {tfin - tini} s')

    # --- 2. Arbol de decisión (AD) ---
    tini = tm.perf_counter()
    tree = DecisionTreeClassifier(criterion='gini',splitter='best')

    tree.fit(X_indv_train[index_train],Y_indv_train[index_train])
    vector_aciertos_AD.append(tree.score(X_indv_train[index_test],Y_indv_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion Árbol de Decisión = {tfin - tini} s')

    
    # --- 3. Máquina de soporte vectorial (SVC) ---
    tini = tm.perf_counter()
    clasificador_SVC = SVC(kernel='rbf', C=1.0)

    clasificador_SVC.fit(X_indv_train[index_train], Y_indv_train[index_train])
    vector_aciertos_SVC.append(clasificador_SVC.score(X_indv_train[index_test], Y_indv_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion SVC = {tfin - tini} s')

N = len(vector_aciertos_MLP)

print('\n--- Perceptrón Multicapa (MLP) ---')
media_mlp = np.sum(vector_aciertos_MLP)/N
varianza_mlp = np.sum((vector_aciertos_MLP - media_mlp)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_MLP}")
print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

print('\n--- Arbol de Decisión (AD) ---')
media_mlp = np.sum(vector_aciertos_AD)/N
varianza_mlp = np.sum((vector_aciertos_AD - media_mlp)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_AD}")
print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")

print('\n--- Máquina de soporte vectorial (SVC) ---')
media_mlp = np.sum(vector_aciertos_SVC)/N
varianza_mlp = np.sum((vector_aciertos_SVC - media_mlp)**2)/N
print(f"Ratios obtenidos = {vector_aciertos_SVC}")
print(f"Media = {media_mlp:.4f} | Varianza = {varianza_mlp:.6f}")



################## MEJORES HIPERPARAMETROS #################################
capas = (80,10)
activation = 'relu'
eta = 0.01
epocas = 1000
solver = 'sgd'
n_iter_no_change = 10
mlp = MLPClassifier(hidden_layer_sizes=capas,activation=activation,solver=solver,learning_rate='adaptive',learning_rate_init=eta,max_iter=epocas, n_iter_no_change=n_iter_no_change)
mlp.fit(X_indv_train,Y_indv_train)
print(mlp.score(X_indv_test,Y_indv_test))


svm = SVC(kernel='rbf', C=15, gamma=0.05)
svm.fit(X_indv_train, Y_indv_train)
print(svm.score(X_indv_train,Y_indv_train))
print(svm.score(X_indv_test,Y_indv_test))

dt = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=0.0)
dt.fit(X_indv_train, Y_indv_train)
print(dt.score(X_indv_test,Y_indv_test))