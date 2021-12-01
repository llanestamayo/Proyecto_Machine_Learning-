### Importamos nuestra librerías y funciones
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import os

#Cambiamos nuestra ruta al directorio actual
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

###
#Importamos funciones
from utils import functions as func

###
## Cargamos los datos
X_train, X_test, y_train, y_test = func.load_train_sets(path = r'data\processed\train_set.npz')
X_train_corr, X_test_corr, y_train_corr, y_test_corr = func.load_train_sets(path = r'data\processed\train_set_corr.npz')
#
## Creamos nuestro primer modelo --- Regresión Lineal
#
#
#Entrenamos con nuestros datos completos
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
pred_lin = lin_reg.predict(X_test)
#Obtenemos las métricas de nuestro modelo
tupla_met_lin = func.obtiene_metricas(y_test, pred_lin)
## Creamos dataframe en el que iremos incluyendo las métricas de nuestros modelos
indices = ['Lineal_reg']
data_metricas = pd.DataFrame({'MAE':tupla_met_lin[0],'MSE':tupla_met_lin[1],'RMSE':tupla_met_lin[2],'R2_Score':tupla_met_lin[3]}, index = indices)

#
#Entrenamos con nuestros cuatro columnas más correladas
lin_reg_corr = LinearRegression()
lin_reg_corr.fit(X_train_corr, y_train_corr)
pred_lin_corr = lin_reg_corr.predict(X_test_corr)
#Obtenemos las métricas de nuestro modelo
tupla_met_lin_corr = func.obtiene_metricas(y_test_corr, pred_lin_corr)
## Creamos dataframe en el que iremos incluyendo las métricas de nuestros modelos
indices_corr = ['Lineal_reg']
data_metricas_corr = pd.DataFrame({'MAE':tupla_met_lin_corr[0],'MSE':tupla_met_lin_corr[1],'RMSE':tupla_met_lin_corr[2],'R2_Score':tupla_met_lin_corr[3]}, index = indices)
score_lin = tupla_met_lin_corr[3]
#
#
# Creamos y entrenamos la regresión polinómica
#
#
# Primero con todos nuestros datos
poly_feats = PolynomialFeatures(degree = 3)
train_set, test_set = func.escala_estandar(train = X_train, test = X_test) #Función para escalar los datos
poly_feats.fit(train_set)
X_poly = poly_feats.transform(train_set)
X_poly_test = poly_feats.transform(test_set)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)
#Predecimos
pred_poly = pol_reg.predict(X_poly_test)
#Obtenemos las métricas de nuestro modelo
tupla_met_poly = func.obtiene_metricas(y_test, pred_poly)
## Añadimos al dataframe general
data_metricas = func.concatena_metricas(data = data_metricas, tupla = tupla_met_poly , indices = ['Poly_reg'])
#
# Entrenamos con las columnas más correladas
poly_feats = PolynomialFeatures(degree = 3)
train_set_corr, test_set_corr = func.escala_estandar(train = X_train_corr, test = X_test_corr) #Función para escalar los datos
poly_feats.fit(train_set_corr)
X_poly_corr = poly_feats.transform(train_set_corr)
X_poly_test_corr = poly_feats.transform(test_set_corr)
pol_reg_corr = LinearRegression()
pol_reg_corr.fit(X_poly_corr, y_train_corr)
#Predecimos
pred_poly_corr = pol_reg_corr.predict(X_poly_test_corr)
#Obtenemos las métricas de nuestro modelo
tupla_met_poly_corr = func.obtiene_metricas(y_test_corr, pred_poly_corr)
## Añadimos al dataframe general
data_metricas_corr = func.concatena_metricas(data = data_metricas_corr, tupla = tupla_met_poly_corr , indices = ['Poly_reg'])
score_poly = tupla_met_poly_corr[3]
#
#
#Pasamos a entrenar un regresor Random Forest
#
#
# Entrenamos con todos nuestros datos. Utilizamos los parámetros que en las pruebas anteriores nos han dado un mejor resultado.
rand_forest_reg = RandomForestRegressor(max_depth= 10, max_features = 4, n_estimators = 100)
rand_forest_reg.fit(X_train,y_train)
#Predecimos
pred_rand = rand_forest_reg.predict(X_test)
#Obtenemos las métricas de nuestro modelo
tupla_met_rand = func.obtiene_metricas(y_test, pred_rand)
## Añadimos al dataframe general
data_metricas = func.concatena_metricas(data = data_metricas, tupla = tupla_met_rand , indices = ['Random_Forest_reg'])
#
# Entrenamos con los datos más correlados. 
rand_forest_reg_corr = RandomForestRegressor(max_depth=10, max_features=4, n_estimators=500)
rand_forest_reg_corr.fit(X_train_corr,y_train_corr)
#Predecimos
pred_rand_corr = rand_forest_reg_corr.predict(X_test_corr)
#Obtenemos las métricas de nuestro modelo
tupla_met_rand_corr = func.obtiene_metricas(y_test_corr, pred_rand_corr)
## Añadimos al dataframe general
data_metricas_corr = func.concatena_metricas(data = data_metricas_corr, tupla = tupla_met_rand_corr , indices = ['Random_Forest_reg'])
score_rand = tupla_met_rand_corr[3]
#
#Pasamos a entrenar un regresor SVM
#
#
#Entrenamos con todos nuestros datos
svr_reg = SVR(kernel = 'rbf', gamma = 'auto', C = 10)
svr_reg.fit(train_set, y_train)
#Predecimos
svr_pred = svr_reg.predict(test_set)
#Obtenemos las métricas de nuestro modelo
tupla_met_svr = func.obtiene_metricas(y_test, svr_pred)
## Añadimos al dataframe general
data_metricas = func.concatena_metricas(data = data_metricas, tupla = tupla_met_svr , indices = ['SVM_reg'])
#
#Entrenamos con los más correlados
svr_reg_corr = SVR(kernel = 'rbf', gamma = 'auto', C = 10)
svr_reg_corr.fit(train_set_corr, y_train_corr)
#Predecimos
svr_pred_corr = svr_reg_corr.predict(test_set_corr)
#Obtenemos las métricas de nuestro modelo
tupla_met_svr_corr = func.obtiene_metricas(y_test_corr, svr_pred_corr)
## Añadimos al dataframe general
data_metricas_corr = func.concatena_metricas(data = data_metricas_corr, tupla = tupla_met_svr_corr , indices = ['SVM_reg'])
score_svr = tupla_met_svr_corr[3]
#
#
#Pasamos a entrenar un regresor con Gradient Boosting
#
#
#Entrenamos con todos nuestros datos
gbt_reg = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 3, n_estimators = 100, random_state=42)
gbt_reg.fit(train_set, y_train)
#Predecimos
gbt_pred = gbt_reg.predict(test_set)
#Obtenemos las métricas de nuestro modelo
tupla_met_gbt = func.obtiene_metricas(y_test, gbt_pred)
## Añadimos al dataframe general
data_metricas = func.concatena_metricas(data = data_metricas, tupla = tupla_met_gbt , indices = ['Gradient_Boosting_reg'])
#
#Entrenamos con los más correlados
gbt_reg_corr = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 3, n_estimators = 100, random_state=42)
gbt_reg_corr.fit(train_set_corr, y_train_corr)
#Predecimos
gbt_pred_corr = gbt_reg_corr.predict(test_set_corr)
#Obtenemos las métricas de nuestro modelo
tupla_met_gbt_corr = func.obtiene_metricas(y_test_corr, gbt_pred_corr)
## Añadimos al dataframe general
data_metricas_corr = func.concatena_metricas(data = data_metricas_corr, tupla = tupla_met_gbt_corr , indices = ['Gradient_Boosting_reg'])
score_gbt = tupla_met_gbt_corr[3]
#
#
#
#Pasamos a entrenar un regresor con un modelo de red neuronal
#
#
#Entrenamos con todos nuestros datos
model = keras.models.Sequential([
    keras.layers.Dense(200, activation = 'relu',
                      input_shape = train_set.shape[1:]),
    keras.layers.Dense(100, activation = 'relu'),

    keras.layers.Dense(100, activation = 'relu'),
                                      
    keras.layers.Dense(1, activation = 'relu')
])

model.compile(loss = "mean_squared_error",
             optimizer = "adam",
             metrics = ["mae","mse"])

history = model.fit(train_set,
                   y_train,
                   epochs = 1000,
                   validation_split = 0.2,
                   batch_size = 64,
                   callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])             
#Predecimos
pred_rn = model.predict(test_set)
#Obtenemos las métricas de nuestro modelo
tupla_met_rn = func.obtiene_metricas(y_test, pred_rn)
## Añadimos al dataframe general
data_metricas = func.concatena_metricas(data = data_metricas, tupla = tupla_met_rn , indices = ['Redes_Neur_reg'])
#
#Entrenamos con los más correlados
model_corr = keras.models.Sequential([
    keras.layers.Dense(200, activation = 'relu',
                      input_shape = train_set_corr.shape[1:]),
    keras.layers.Dense(100, activation = 'relu'),

    keras.layers.Dense(100, activation = 'relu'),
                                      
    keras.layers.Dense(1, activation = 'relu')
])

model_corr.compile(loss = "mean_squared_error",
             optimizer = "adam",
             metrics = ["mae","mse"])

history_corr = model_corr.fit(train_set_corr,
                   y_train_corr,
                   epochs = 1000,
                   validation_split = 0.2,
                   batch_size = 64,
                   callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])  
#Predecimos
pred_rn_corr = model_corr.predict(test_set_corr)
#Obtenemos las métricas de nuestro modelo
tupla_met_rn_corr = func.obtiene_metricas(y_test_corr, pred_rn_corr)
## Añadimos al dataframe general
data_metricas_corr = func.concatena_metricas(data = data_metricas_corr, tupla = tupla_met_rn_corr , indices = ['Redes_Neur_reg'])
score_rn = tupla_met_rn_corr[3]
#
#Imprimimos para ver nuestros resultados
print('\nMétricas para las predicciones efectuadas con todos los datos:\n')
print('\n',data_metricas)
print('\nMétricas para las predicciones efectuadas con los datos más correlados:\n')
print('\n',data_metricas_corr)
#Creamos dataframe con nuestros modelos y sus scores
modelos_pred = pd.DataFrame({'model':[lin_reg_corr, pol_reg_corr, rand_forest_reg_corr, svr_reg_corr, gbt_reg_corr, model_corr],
                                'score': [score_lin, score_poly, score_rand, score_svr, score_gbt, score_rn]})
#Seleccionamos el mejor modelo                               
final_model = modelos_pred[modelos_pred['score'] == modelos_pred['score'].max()]['model']   
#Guardamos nuestro modelo final (el de mejor score)                             
func.guarda_modelo(modelo = final_model, path = "models/final_model.model")                   

