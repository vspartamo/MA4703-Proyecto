import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from get_data import get_data_donut_2d, get_data_spiral_2d, get_data_squares_2d, get_data_donut_1d

# Función para generar los datos
def generate_data(dataset_type="donut_2d", n_samples=1000):
    if dataset_type == "donut_1d":
        features, labels = get_data_donut_1d(n_samples)
    elif dataset_type == "donut_2d":
        features, labels = get_data_donut_2d(n_samples)
    elif dataset_type == "spiral_2d":
        features, labels = get_data_spiral_2d(n_samples)
    elif dataset_type == "squares_2d":
        features, labels = get_data_squares_2d(n_samples)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    features = features.T
    labels = labels.reshape(-1, 1)

    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


# Función para graficar el límite de decisión en un subplot
def plot_decision_boundary_subplot(ax, model, X, y, title="Decision Boundary"):
    X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX, YY = np.meshgrid(np.arange(X_min, X_max, 0.1),
                         np.arange(Y_min, Y_max, 0.1))
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)

    ax.contourf(XX, YY, Z, alpha=0.75, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=ListedColormap(['#FF0000', '#0000FF']), alpha=0.1)
    ax.set_title(title)


# Función para evaluar modelos de clasificación
def evaluate_models(X, y):
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Definir y entrenar modelos
    models = {
        "SVM": SVC(kernel='rbf', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        results[name] = {"accuracy": acc, "precision": precision}
        print(f"{name} Accuracy: {acc:.4f}, Precision: {precision:.4f}")

    # Crear subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    # Graficar los límites de decisión en los subplots
    for ax, (name, model) in zip(axes, models.items()):
        plot_decision_boundary_subplot(ax, model, X_train, y_train, title=name)

    # Ajustar diseño
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    # Elegir dataset y generar datos
    dataset_type = input("Enter dataset type (donut_1d, donut_2d, spiral_2d, squares_2d): ").strip()
    X, y = generate_data(dataset_type=dataset_type, n_samples=1000)

    # Evaluar modelos
    print(f"Evaluating models on dataset '{dataset_type}'...")
    results = evaluate_models(X.numpy(), y.numpy().flatten())  # Convertir a numpy para sklearn

    # Mostrar resultados
    print("\nPerformance Summary:")
    for model, scores in results.items():
        print(f"{model} - Accuracy: {scores['accuracy']:.4f}, Precision: {scores['precision']:.4f}")