import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

class PerceptronSimple:
    def __init__(self, num_features, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(num_features + 1)  # +1 for the bias

    def predict(self, x):
        activation = np.dot(self.weights[1:], x) + self.weights[0]
        return 1 if activation >= 0 else -1

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                prediction = self.predict(X[i])
                self.weights[1:] += self.learning_rate * (y[i] - prediction) * X[i]
                self.weights[0] += self.learning_rate * (y[i] - prediction)

    def evaluate(self, X_test, y_test):
        correct = 0
        for i in range(X_test.shape[0]):
            if self.predict(X_test[i]) == y_test[i]:
                correct += 1
        return correct / X_test.shape[0]

# Función para cargar datos desde un archivo CSV
def load_csv(filename):
    data = pd.read_csv(filename, header=None)
    return data.values

# Cargar datos originales
original_data = load_csv('spheres1d10.csv')

# Función para perturbar los datos con un cierto porcentaje
def perturb_data(data, percentage):
    noise = np.random.normal(0, 0.1, size=data.shape)
    return data + percentage * noise

# Cargar datos perturbados con diferentes porcentajes
perturbed_10 = perturb_data(load_csv('spheres2d10.csv'), 0.1)
perturbed_50 = perturb_data(load_csv('spheres2d50.csv'), 0.5)
perturbed_70 = perturb_data(load_csv('spheres2d70.csv'), 0.7)

# Gráficas en 3D
fig = plt.figure(figsize=(15, 15))

# Datos originales
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(original_data[:,0], original_data[:,1], original_data[:,2])
ax1.set_title('Datos originales')

# Datos perturbados < 10%
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(perturbed_10[:,0], perturbed_10[:,1], perturbed_10[:,2])
ax2.set_title('Datos perturbados < 10%')

# Datos perturbados < 50%
ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(perturbed_50[:,0], perturbed_50[:,1], perturbed_50[:,2])
ax3.set_title('Datos perturbados < 50%')

# Datos perturbados < 70%
ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(perturbed_70[:,0], perturbed_70[:,1], perturbed_70[:,2])
ax4.set_title('Datos perturbados < 70%')

plt.show()