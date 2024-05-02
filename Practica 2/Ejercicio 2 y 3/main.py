import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

# Desactivar las advertencias de Sklearn
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Cargar los datasets desde los archivos CSV
swedish_auto_data = pd.read_csv('AutoInsurSweden.csv')
wine_quality_data = pd.read_csv('wine-Quality.csv')
pima_diabetes_data = pd.read_csv('pima-indians-diabetes.csv')

# Categorizar los valores de Y en Swedish Auto Insurance Dataset
quartiles = swedish_auto_data['Y'].quantile([0.25, 0.5, 0.75])
low_limit = quartiles.iloc[0]
medium_limit = quartiles.iloc[1]
high_limit = quartiles.iloc[2]

def categorize_y(y):
    if y <= low_limit:
        return 'bajo'
    elif y <= medium_limit:
        return 'medio'
    else:
        return 'alto'

swedish_auto_data['Y_category'] = swedish_auto_data['Y'].apply(categorize_y)

# Definir las variables X_swedish, X_wine y X_pima
X_swedish = swedish_auto_data[['X']]
X_wine = wine_quality_data.drop('quality', axis=1)
X_pima = pima_diabetes_data.drop('Class variable (0 or 1)', axis=1)

# Definir una función para implementar clasificadores y calcular métricas de evaluación
def evaluate_classifier(X, y, classifier):
    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el clasificador
    if classifier == LogisticRegression:
        model = LogisticRegression(max_iter=1000)
    else:
        model = classifier()
    model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division='warn')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1

# Función para graficar las métricas
def plot_metrics(metrics_dict, title):
    labels = list(metrics_dict.keys())
    metrics = list(metrics_dict.values())

    # Obtener las métricas
    accuracy = [metric[0] for metric in metrics]
    precision = [metric[1] for metric in metrics]
    recall = [metric[2] for metric in metrics]
    f1 = [metric[3] for metric in metrics]

    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x, accuracy, width=0.15, label='Accuracy')
    plt.bar([i + 0.15 for i in x], precision, width=0.15, label='Precision')
    plt.bar([i + 0.3 for i in x], recall, width=0.15, label='Recall')
    plt.bar([i + 0.45 for i in x], f1, width=0.15, label='F1 Score')

    plt.xlabel('Clasificador')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks([i + 0.3 for i in x], labels)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Función para evaluar y graficar las métricas de un conjunto de datos
def evaluate_and_plot(X, y, title):
    classifiers = [LogisticRegression, KNeighborsClassifier, SVC, GaussianNB, MLPClassifier]
    metrics_dict = {}

    for classifier in classifiers:
        accuracy, precision, recall, f1 = evaluate_classifier(X, y, classifier)
        metrics_dict[classifier.__name__] = [accuracy, precision, recall, f1]

    plot_metrics(metrics_dict, title)

# Swedish Auto Insurance Dataset
y_swedish = swedish_auto_data['Y_category']
evaluate_and_plot(X_swedish, y_swedish, "Swedish Auto Insurance Dataset")

# Wine Quality Dataset
y_wine = wine_quality_data['quality']
evaluate_and_plot(X_wine, y_wine, "Wine Quality Dataset")

# Pima Indians Diabetes Dataset
y_pima = pima_diabetes_data['Class variable (0 or 1)']
evaluate_and_plot(X_pima, y_pima, "Pima Indians Diabetes Dataset")
