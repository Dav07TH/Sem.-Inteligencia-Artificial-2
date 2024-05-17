import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Cargar el conjunto de datos Zoo desde el archivo CSV
zoo_data = pd.read_csv("ZOO.csv")

# Verificar la distribución de las clases
print(zoo_data["class_type"].value_counts())

# Separar las características (X) y las etiquetas (y)
X = zoo_data.drop(["animal_name", "class_type"], axis=1)
y = zoo_data["class_type"]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sobremuestreo de las clases minoritarias
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Inicializar los modelos con el número de iteraciones aumentado para Logistic Regression
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machines": SVC(),
    "Naive Bayes": GaussianNB()
}

# Función para calcular Specificity
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm.diagonal().sum() - cm.sum(axis=0)
    fp = cm.sum(axis=0) - cm.diagonal()
    specificity = tn / (tn + fp)
    return specificity.mean()

# Evaluar los modelos usando validación cruzada
results = {}
kfold = StratifiedKFold(n_splits=10)
for name, model in models.items():
    y_pred = cross_val_predict(model, X_resampled, y_resampled, cv=kfold)
    accuracy = accuracy_score(y_resampled, y_pred)
    precision = precision_score(y_resampled, y_pred, average='weighted', zero_division=0)
    sensitivity = recall_score(y_resampled, y_pred, average='weighted', zero_division=0)  # Sensitivity es igual a Recall
    f1 = f1_score(y_resampled, y_pred, average='weighted', zero_division=0)

    # Entrenar el modelo para calcular Specificity en el conjunto de prueba
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    specificity = calculate_specificity(y_test, y_test_pred)
    
    results[name] = {"Accuracy": accuracy, "Precision": precision, "Sensitivity": sensitivity, "Specificity": specificity, "F1 Score": f1}

# Agregar red neuronal
def create_neural_network(input_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # 7 clases
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Convertir las etiquetas a una codificación one-hot
y_resampled_categorical = to_categorical(y_resampled - 1)  # -1 porque las clases empiezan desde 1

# Crear y entrenar la red neuronal
nn_model = create_neural_network(X_train.shape[1])
nn_model.fit(X_train, to_categorical(y_train - 1), epochs=50, batch_size=10, verbose=0)

# Evaluar la red neuronal
y_test_pred_nn = nn_model.predict(X_test)
y_test_pred_nn_classes = y_test_pred_nn.argmax(axis=1) + 1  # +1 para ajustar el índice de clase

accuracy_nn = accuracy_score(y_test, y_test_pred_nn_classes)
precision_nn = precision_score(y_test, y_test_pred_nn_classes, average='weighted', zero_division=0)
sensitivity_nn = recall_score(y_test, y_test_pred_nn_classes, average='weighted', zero_division=0)
f1_nn = f1_score(y_test, y_test_pred_nn_classes, average='weighted', zero_division=0)
specificity_nn = calculate_specificity(y_test, y_test_pred_nn_classes)

results["Neural Network"] = {
    "Accuracy": accuracy_nn,
    "Precision": precision_nn,
    "Sensitivity": sensitivity_nn,
    "Specificity": specificity_nn,
    "F1 Score": f1_nn
}

# Imprimir los resultados
print("Resultados de la evaluación:")
for name, metrics in results.items():
    print(f"Modelo: {name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()

# Crear un DataFrame a partir de los resultados
results_df = pd.DataFrame(results).T  # Transponer para mejor visualización en la gráfica

# Graficar los resultados
results_df.plot(kind='bar', rot=0)
plt.title('Comparación de Métricas de Evaluación')
plt.xlabel('Métrica')
plt.ylabel('Valor')
plt.xticks(rotation=45)
plt.legend(title='Modelo')
plt.tight_layout()
plt.show()
