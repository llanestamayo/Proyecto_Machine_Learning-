#Importamos todas las librerías que podriamos necesitar
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

#Función para la carga de los datasets
def load_dataset(path, separador, codif):
    new_data = pd.read_csv(path, sep= separador, encoding = codif)
    return new_data

#Función para guardar nuevos datasets
def save_dataset(data, path, separador):
    data.to_csv(path, separador)

#Función para guardar sets de entrenamiento y test
def save_train_sets(path, tupla_datos):
    np.savez(path,
        X_train = tupla_datos[0],
        y_train = tupla_datos[2],
        X_test = tupla_datos[1],
        y_test = tupla_datos[3])   

#Función para cargar sets de entrenamiento y test
def load_train_sets(path):
    data = np.load(path)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    return X_train, X_test, y_train, y_test

#Función para remplazar valores de columnas partiendo de un diccionario
def cambia_valores(data, nombre_columna, dicc_valores):
    data = data.replace({nombre_columna:dicc_valores}) 
    return data

#Función para hacer el split de los datos
def split_train_set(data, test_prop, rand_st, target_name):
    X = (data.drop(target_name, axis = 'columns'))
    y= (data[target_name])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=rand_st)
    return X_train, X_test, y_train, y_test

#Función para mostrar las predicciones contra valor real y guardar la figura
def muestra_pred(data, data_pred, long, path):
    plt.figure(figsize= (30,15))
    plt.plot(range(long), (data_pred[:long]), color = 'red',linestyle='-', marker='o', label='Predicción')
    plt.plot(range(long),(data[:long]),linestyle='--', marker='o',color = 'blue', label='Valor Real')
    plt.xlabel('Índices', fontsize = 18)
    plt.ylabel('Predicciones y Valores Reales de Test',fontsize = 18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(path)

#Función para imprimir los errores y el score obtenidos
def muestra_metricas(data,data_pred):
    print('MAE:', metrics.mean_absolute_error(data, data_pred))
    print('MSE:', metrics.mean_squared_error(data, data_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(data, data_pred)))
    print('R2 score:', metrics.r2_score(data, data_pred))  

#Función para guardar los modelos
def guarda_modelo(path, modelo):
    with open(path, "wb") as archivo_salida:
        pickle.dump(modelo, archivo_salida)

#Función para escalado estándar
def escala_estandar(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    train_scal = scaler.transform(train)
    test_scal = scaler.transform(test)
    return train_scal, test_scal

#Función para escalado MinMax
def escala_minmax(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    train_scal = scaler.transform(train)
    test_scal = scaler.transform(test)
    return train_scal, test_scal

#Función para obtener métricas de regresión
def obtiene_metricas(data,data_pred):
    MAE_Modelo =  metrics.mean_absolute_error(data, data_pred)
    MSE_Modelo = metrics.mean_squared_error(data, data_pred)
    RMSE_Modelo = np.sqrt(metrics.mean_squared_error(data, data_pred))
    R2_Score_Modelo = metrics.r2_score(data, data_pred)
    return MAE_Modelo, MSE_Modelo, RMSE_Modelo, R2_Score_Modelo

#Función para añadir métricas a un dataframe
def concatena_metricas(data, tupla, indices):
    indices_poly = ['Poly_reg']
    data_new = pd.DataFrame({'MAE':tupla[0],'MSE':tupla[1],'RMSE':tupla[2],'R2_Score':tupla[3]}, index = indices)
    data = pd.concat([data,data_new])        
    return data

#Función para mostrar las tres mejores predicciones contra valor real y guardar la figura
def muestra_top(data, tupla_pred, tupla_labels, long, path):
        
    plt.figure(figsize=(30,20))
    plt.subplot(221)
    plt.plot(range(long), (tupla_pred[0][-long:]), color = 'blue',linestyle='--', marker='o', label= tupla_labels[0])
    plt.plot(range(long),(data[-long:]),color = 'red',linestyle='-', marker='o', label= tupla_labels[4])
    plt.xlabel("Índices", fontsize=14)
    plt.ylabel("Predicción y valor real", fontsize=14)
    plt.legend(fontsize=12)
    
    plt.grid()

    plt.subplot(222)
    plt.plot(range(long), (tupla_pred[1][-long:]), color = 'green',linestyle='--', marker='o', label= tupla_labels[1])
    plt.plot(range(long),(data[-long:]),color = 'red',linestyle='-', marker='o', label= tupla_labels[4])
    plt.xlabel("Índices", fontsize=14)
    plt.legend(fontsize=12)
    plt.tick_params(labelleft=False)
    plt.grid()
    
    plt.subplot(223)
    plt.plot(range(long), (tupla_pred[2][-long:]), color = 'orange',linestyle='--', marker='o', label= tupla_labels[2])
    plt.plot(range(long),(data[-long:]),color = 'red',linestyle='-', marker='o', label= tupla_labels[4])
    plt.xlabel("Índices", fontsize=14)
    plt.ylabel("Predicción y valor real", fontsize=14)
    plt.legend(fontsize=12)
    
    plt.grid()

    plt.subplot(224)
    plt.plot(range(long), (tupla_pred[3][-long:]), color = 'purple',linestyle='--', marker='o', label= tupla_labels[3])
    plt.plot(range(long),(data[-long:]),color = 'red',linestyle='-', marker='o', label= tupla_labels[4])
    plt.xlabel("Índices", fontsize=14)
    plt.legend(fontsize=12)
     
    plt.tick_params(labelleft=False)
    plt.grid()
    plt.savefig(path)