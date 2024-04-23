import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

class PerceptronMulticapa:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=2000, random_state=None):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                   solver=solver, max_iter=max_iter, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

def leave_k_out(X, y, k):
    kf = KFold(n_splits=k)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        perceptron_multicapa = PerceptronMulticapa(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=2000, random_state=42)
        perceptron_multicapa.train(X_train, y_train)
        errors.append(1 - perceptron_multicapa.evaluate(X_test, y_test))
    return errors

def leave_one_out(X, y):
    loo = LeaveOneOut()
    errors = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        perceptron_multicapa = PerceptronMulticapa(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=2000, random_state=42)
        perceptron_multicapa.train(X_train, y_train)
        errors.append(1 - perceptron_multicapa.evaluate(X_test, y_test))
    return errors

# Función para cargar y preprocesar los datos del archivo irisbin.csv
def load_iris_data(file_path):
    data = pd.read_csv(file_path)
    species = []
    for i in range(len(data)):
        if (data.iloc[i][-3:] == [-1, -1, 1]).all():
            species.append('setosa')
        elif (data.iloc[i][-3:] == [-1, 1, -1]).all():
            species.append('versicolor')
        elif (data.iloc[i][-3:] == [1, -1, -1]).all():
            species.append('virginica')
    X = data.iloc[:, :-3].values
    y = np.array(species)
    return X, y

# Cargar datos desde el archivo irisbin.csv
X_iris, y_iris = load_iris_data('irisbin.csv')

# Leave-k-out
k_out_errors = leave_k_out(X_iris, y_iris, k=10)
mean_k_out = np.mean(k_out_errors)
std_k_out = np.std(k_out_errors)

# Leave-one-out
one_out_errors = leave_one_out(X_iris, y_iris)
mean_one_out = np.mean(one_out_errors)
std_one_out = np.std(one_out_errors)

print("Leave-k-out: Mean Error =", mean_k_out, "Standard Deviation =", std_k_out)
print("Leave-one-out: Mean Error =", mean_one_out, "Standard Deviation =", std_one_out)

# Reducción de dimensionalidad con PCA para la proyección en dos dimensiones
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_iris)

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_2d, y_iris, test_size=0.2, random_state=42)

# Entrenar el perceptrón multicapa
perceptron_multicapa = PerceptronMulticapa(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=2000, random_state=42)
perceptron_multicapa.train(X_train_iris, y_train_iris)

# Evaluar el modelo
accuracy_iris = perceptron_multicapa.evaluate(X_test_iris, y_test_iris)
print(f'Accuracy (irisbin.csv): {accuracy_iris}')

# Graficar la proyección en dos dimensiones de la distribución de clases
plt.figure(figsize=(8, 6))
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

for species in colors:
    subset = X_2d[y_iris == species]
    plt.scatter(subset[:, 0], subset[:, 1], color=colors[species], label=species)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Proyección en Dos Dimensiones de la Distribución de Clases para el Dataset Iris')
plt.legend()
plt.grid(True)
plt.show()

# Graficar la distribución de clases en dos dimensiones
plt.figure(figsize=(8, 6))

for species in colors:
    subset = X_iris[y_iris == species]  
    plt.scatter(subset[:, 0], subset[:, 1], color=colors[species], label=species)

plt.xlabel('Sepal Length (cm)')  
plt.ylabel('Sepal Width (cm)')
plt.title('Distribución de Clases para el Dataset Iris')  
plt.legend()
plt.grid(True)

# Agregar también los datos de "petal"
for species in colors:
    subset = X_iris[y_iris == species]  
    plt.scatter(subset[:, 2], subset[:, 3], color=colors[species], marker='x')

plt.xlabel('Length (cm)')  
plt.ylabel('Width (cm)')
plt.title('Distribución de Clases para el Dataset Iris')  
plt.legend()
plt.grid(True)
plt.show()