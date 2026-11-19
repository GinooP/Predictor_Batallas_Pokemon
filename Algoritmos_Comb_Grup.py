# Implementación de metodos de aprendizaje MLP, DT y SVM con k-fold de 3
# + Prueba de los mejores Hiperparametros encontrados Para Batallas Grupales


import numpy as np
import time as tm
import matplotlib.pyplot as plt
import matriz_de_confusion as mc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
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


######################################################################
# # Constantes MLP
capas = (20,40)
activation = 'relu'
solver='adam'
eta = 0.05
epocas = 1000
tol = 0.95
n_iter_no_change = epocas

kernel='rbf'
C=5
gamma=0.01

criterion='gini'
min_impurity_decrease=0.0

vector_aciertos_MLP = []
vector_aciertos_AD = []
vector_aciertos_SVC = []

for i,(index_train,index_test) in enumerate(fk.split(X_grup_train)):
    print(f'Kfol = {i}')

    # --- 1. Perceptrón Multicapa (MLP) ---
    tini = tm.perf_counter()
    modelo = MLPClassifier(hidden_layer_sizes=capas,activation=activation,solver=solver,learning_rate_init=eta,max_iter=epocas,
                            tol=tol, verbose=False,n_iter_no_change=n_iter_no_change,learning_rate='adaptive') 
    
    modelo.fit(X_grup_train[index_train],Y_grup_train[index_train])
    vector_aciertos_MLP.append(modelo.score(X_grup_train[index_test],Y_grup_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion MLP = {tfin - tini} s')

    # --- 2. Arbol de decisión (AD) ---
    tini = tm.perf_counter()
    tree = DecisionTreeClassifier(criterion=criterion,min_impurity_decrease=min_impurity_decrease)

    tree.fit(X_grup_train[index_train],Y_grup_train[index_train])
    vector_aciertos_AD.append(tree.score(X_grup_train[index_test],Y_grup_train[index_test]))
    tfin = tm.perf_counter()
    print(f'Duracion Árbol de Decisión = {tfin - tini} s')

    
    # --- 3. Máquina de soporte vectorial (SVC) ---
    tini = tm.perf_counter()
    clasificador_SVC = SVC(kernel=kernel, C=C, gamma=gamma)

    clasificador_SVC.fit(X_grup_train[index_train], Y_grup_train[index_train])
    vector_aciertos_SVC.append(clasificador_SVC.score(X_grup_train[index_test], Y_grup_train[index_test]))
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
capas = (20,40)
activation = 'relu'
solver='adam'
eta = 0.05
epocas = 1000

# ----- Otra Prueba -----
# capas = (20,40)
# solver = 'sgd'
# activation = 'logistic'
# eta = 0.01
# epocas = 250

mlp = MLPClassifier(hidden_layer_sizes=capas,activation=activation,solver=solver,learning_rate_init=eta,learning_rate='adaptive',max_iter=epocas, n_iter_no_change=epocas)
mlp.fit(X_grup_train,Y_grup_train)
print(mlp.score(X_grup_test,Y_grup_test))


svm = SVC(kernel='rbf', C=5, gamma=0.01)
svm.fit(X_grup_train, Y_grup_train)
print(svm.score(X_grup_test,Y_grup_test))


dt = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=0.0)
dt.fit(X_grup_train, Y_grup_train)
print(dt.score(X_grup_test,Y_grup_test))






###################### Determinar las matrices de confuncion #########################

# MLP: capas = (20,40), activation=”relu”, learning rate=”adaptive”, learning rate init=0.05, maxit=1000, solver=’adam’, n iter no change=10
# SVM: ’C’ = 5, gamma = 0.01, kernel = ”rbf”
# DT: criterion=”gini”, min impurity decrease=0.0

multicapa= MLPClassifier(hidden_layer_sizes= (20,40), activation='relu', learning_rate='adaptive', learning_rate_init=0.05, max_iter=1000, solver='adam', n_iter_no_change=1000)

clasificador_SVC=SVC(C= 5,gamma=0.01,kernel='rbf')

tree = DecisionTreeClassifier(criterion='gini',min_impurity_decrease=0)

multicapa.fit(X_grup_train,Y_grup_train)
clasificador_SVC.fit(X_grup_train,Y_grup_train)
tree.fit(X_grup_train,Y_grup_train)


#------------- multicapa -----------------------------------
Y_predicha=multicapa.predict(X_grup_test)#obtenemos los valores predichos para el test

matriz_confucion_MLP= mc.obtener_matriz_de_confusion(Y_grup_test,Y_predicha)

sensibilidad_eq1= matriz_confucion_MLP[0,0]/(matriz_confucion_MLP[0,0]+matriz_confucion_MLP[0,1])
precision_eq1= matriz_confucion_MLP[0,0]/(matriz_confucion_MLP[0,0] + matriz_confucion_MLP[1,0])
F1_eq1= 2*sensibilidad_eq1*precision_eq1/(sensibilidad_eq1 + precision_eq1)

sensibilidad_eq2= matriz_confucion_MLP[1,1]/(matriz_confucion_MLP[1,1] + matriz_confucion_MLP[1,0])
precision_eq2= matriz_confucion_MLP[1,1]/(matriz_confucion_MLP[1,1] + matriz_confucion_MLP[0,1])
F1_eq2= 2*sensibilidad_eq2*precision_eq2/(sensibilidad_eq2 + precision_eq2)

print("--------------- multicapa grupal mejores parametros -----------------------")
print("-------- matriz de confusión")
print(matriz_confucion_MLP)
print(f'ratio_aciertos={multicapa.score(X_grup_test,Y_grup_test)}')
print("-------medidas de desempeño haciendo foco en el equipo 1")
print(f"sensibilidad equipo 1 = {sensibilidad_eq1}")
print(f"precision equipo 1 = {precision_eq1}")
print(f"F1 equipo 1 = {F1_eq1}")
print(f"------medidas de desempeño haciendo foco en equipo 2")
print(f"sensibilidad equipo 2 = {sensibilidad_eq2}")
print(f"precision equipo 2 = {precision_eq2}")
print(f"F1 equipo 2 = {F1_eq2}")


# ------------ SVM ------------------------------------------
Y_predicha=clasificador_SVC.predict(X_grup_test)#obtenemos los valores predichos para el test

matriz_confucion_SVM= mc.obtener_matriz_de_confusion(Y_grup_test,Y_predicha)

sensibilidad_eq1= matriz_confucion_SVM[0,0]/(matriz_confucion_SVM[0,0]+matriz_confucion_SVM[0,1])
precision_eq1= matriz_confucion_SVM[0,0]/(matriz_confucion_SVM[0,0] + matriz_confucion_SVM[1,0])
F1_eq1= 2*sensibilidad_eq1*precision_eq1/(sensibilidad_eq1 + precision_eq1)

sensibilidad_eq2= matriz_confucion_SVM[1,1]/(matriz_confucion_SVM[1,1] + matriz_confucion_SVM[1,0])
precision_eq2= matriz_confucion_SVM[1,1]/(matriz_confucion_SVM[1,1] + matriz_confucion_SVM[0,1])
F1_eq2= 2*sensibilidad_eq2*precision_eq2/(sensibilidad_eq2 + precision_eq2)

print("--------------- SVM grupal mejores parametros -----------------------")
print("-------- matriz de confusión")
print(matriz_confucion_SVM)
print(f'ratio_aciertos={clasificador_SVC.score(X_grup_test,Y_grup_test)}')
print("-------medidas de desempeño haciendo foco en el equipo 1")
print(f"sensibilidad equipo 1 = {sensibilidad_eq1}")
print(f"precision equipo 1 = {precision_eq1}")
print(f"F1 equipo 1 = {F1_eq1}")
print(f"------medidas de desempeño haciendo foco en equipo 2")
print(f"sensibilidad equipo 2 = {sensibilidad_eq2}")
print(f"precision equipo 2 = {precision_eq2}")
print(f"F1 equipo 2 = {F1_eq2}")
#------------ Árbol ----------------------------------------
Y_predicha=tree.predict(X_grup_test)#obtenemos los valores predichos para el test

matriz_confucion_tree= mc.obtener_matriz_de_confusion(Y_grup_test,Y_predicha)

sensibilidad_eq1= matriz_confucion_tree[0,0]/(matriz_confucion_tree[0,0]+matriz_confucion_tree[0,1])
precision_eq1= matriz_confucion_tree[0,0]/(matriz_confucion_tree[0,0] + matriz_confucion_tree[1,0])
F1_eq1= 2*sensibilidad_eq1*precision_eq1/(sensibilidad_eq1 + precision_eq1)

sensibilidad_eq2= matriz_confucion_tree[1,1]/(matriz_confucion_tree[1,1] + matriz_confucion_tree[1,0])
precision_eq2= matriz_confucion_tree[1,1]/(matriz_confucion_tree[1,1] + matriz_confucion_tree[0,1])
F1_eq2= 2*sensibilidad_eq2*precision_eq2/(sensibilidad_eq2 + precision_eq2)

print("--------------- Árbol grupal mejores parametros -----------------------")
print("-------- matriz de confusión")
print(matriz_confucion_tree)
print(f'ratio_aciertos={tree.score(X_grup_test,Y_grup_test)}')
print("-------medidas de desempeño haciendo foco en el equipo 1")
print(f"sensibilidad equipo 1 = {sensibilidad_eq1}")
print(f"precision equipo 1 = {precision_eq1}")
print(f"F1 equipo 1 = {F1_eq1}")
print(f"------medidas de desempeño haciendo foco en equipo 2")
print(f"sensibilidad equipo 2 = {sensibilidad_eq2}")
print(f"precision equipo 2 = {precision_eq2}")
print(f"F1 equipo 2 = {F1_eq2}")

