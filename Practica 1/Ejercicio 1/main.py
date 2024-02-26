import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_entradas, tasa_aprendizaje=0.01, max_epocas=1000, umbral=0.0):
        self.pesos = np.zeros(num_entradas + 1)  # +1 para el sesgo
        self.tasa_aprendizaje = tasa_aprendizaje
        self.max_epocas = max_epocas
        self.umbral = umbral

    def predecir(self, entradas):
        suma = np.dot(entradas, self.pesos[1:]) + self.pesos[0]  # producto punto + sesgo
        return 1 if suma > self.umbral else -1

    def entrenar(self, entradas_entrenamiento, etiquetas):
        for _ in range(self.max_epocas):
            for entradas, etiqueta in zip(entradas_entrenamiento, etiquetas):
                prediccion = self.predecir(entradas)
                self.pesos[1:] += self.tasa_aprendizaje * (etiqueta - prediccion) * entradas
                self.pesos[0] += self.tasa_aprendizaje * (etiqueta - prediccion)

            if all(self.predecir(entradas) == etiqueta for entradas, etiqueta in zip(entradas_entrenamiento, etiquetas)):
                break

def mostrar_datos_y_linea(X, y, pesos):
    plt.scatter(X[:,0], X[:,1], c=y)
    x_vals = np.array([np.min(X[:,0]), np.max(X[:,0])])
    y_vals = -(pesos[0] + pesos[1] * x_vals) / pesos[2]
    plt.plot(x_vals, y_vals, '--')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Datos y Línea de Decisión')
    plt.show()

# Lectura de los datos de entrenamiento y prueba desde archivos CSV
df_entrenamiento = pd.read_csv('OR_trn.csv', header=None)
X_entrenamiento = df_entrenamiento.iloc[:, :-1].values
y_entrenamiento = df_entrenamiento.iloc[:, -1].values

df_prueba = pd.read_csv('OR_tst.csv', header=None)
X_prueba = df_prueba.iloc[:, :-1].values
y_prueba = df_prueba.iloc[:, -1].values

# Entrenamiento del perceptrón
perceptron = Perceptron(num_entradas=2)
perceptron.entrenar(X_entrenamiento, y_entrenamiento)

# Prueba del perceptrón
precision = np.mean([1 if perceptron.predecir(x) == y else 0 for x, y in zip(X_prueba, y_prueba)])
print("Precisión:", precision)

# Visualización de los datos y la línea de decisión
mostrar_datos_y_linea(X_entrenamiento, y_entrenamiento, perceptron.pesos)
