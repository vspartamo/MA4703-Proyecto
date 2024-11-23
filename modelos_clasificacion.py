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
from sklearn.metrics import accuracy_score, adjusted_rand_score
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

# Función para graficar el límite de decisión
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX, YY = np.meshgrid(np.arange(X_min, X_max, 0.1),
                         np.arange(Y_min, Y_max, 0.1))
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)

    plt.contourf(XX, YY, Z, alpha=0.75, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.title(title)
    plt.show()

# Función para evaluar modelos de clasificación
def evaluate_models(X, y):
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 1. Regresión logística
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    log_reg_preds = log_reg.predict(X_test)
    log_reg_acc = accuracy_score(y_test, log_reg_preds)
    print(f"Logistic Regression Accuracy: {log_reg_acc:.4f}")

    # 2. Árboles de decisión
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    tree_preds = tree_clf.predict(X_test)
    tree_acc = accuracy_score(y_test, tree_preds)
    print(f"Decision Tree Accuracy: {tree_acc:.4f}")

    # 3. K-Vecinos más cercanos (KNN)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_preds)
    print(f"KNN Accuracy: {knn_acc:.4f}")

    # 4. Máquinas de soporte vectorial (SVM)
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    print(f"SVM Accuracy: {svm_acc:.4f}")

    # 5. Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # 6. Gradient Boosting
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_clf.fit(X_train, y_train)
    gb_preds = gb_clf.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_preds)
    print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

    # 7. Naive Bayes
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    nb_preds = nb_clf.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_preds)
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")

    # 8. QDA
    qda_clf = QuadraticDiscriminantAnalysis()
    qda_clf.fit(X_train, y_train)
    qda_preds = qda_clf.predict(X_test)
    qda_acc = accuracy_score(y_test, qda_preds)
    print(f"QDA Accuracy: {qda_acc:.4f}")

    # Graficar los límites de decisión para los modelos
    models = [log_reg, tree_clf, knn, svm, rf_clf, gb_clf, nb_clf, qda_clf]
    model_names = ["Logistic Regression", "Decision Tree", "KNN", "SVM", "Random Forest", "Gradient Boosting", "Naive Bayes", "QDA"]

    for model, name in zip(models, model_names):
        print(f"Plotting decision boundary for {name}...")
        plot_decision_boundary(model, X_train, y_train, title=f"{name} Decision Boundary")

    # 9. Comparar resultados
    results = {
        "Logistic Regression": log_reg_acc,
        "Decision Tree": tree_acc,
        "KNN": knn_acc,
        "SVM": svm_acc,
        "Random Forest": rf_acc,
        "Gradient Boosting": gb_acc,
        "Naive Bayes": nb_acc,
        "QDA": qda_acc,
    }
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
    for model, score in results.items():
        print(f"{model}: {score:.4f}")
