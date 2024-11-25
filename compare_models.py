from resnet import ResNet, generate_data, train_model, evaluate_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import time

def plot_decision_boundary_subplot(ax, model, X, y, title=""):
    """
    Plot decision boundary for a given model and dataset in a subplot.
    Supports both sklearn models and PyTorch ResNet.
    """
    X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX, YY = np.meshgrid(np.arange(X_min, X_max, 0.1),
                         np.arange(Y_min, Y_max, 0.1))
    grid_points = np.c_[XX.ravel(), YY.ravel()]

    if isinstance(model, ResNet):
        # Usar el método forward de ResNet
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        with torch.no_grad():
            preds = model(grid_tensor).squeeze().numpy()
        Z = preds.reshape(XX.shape)
    else:
        # Usar predict para modelos sklearn
        Z = model.predict(grid_points).reshape(XX.shape)

    # Crear el gráfico
    ax.contourf(XX, YY, Z, alpha=0.75, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=ListedColormap(['#FF0000', '#0000FF']), alpha=0.1)
    ax.set_title(title)


def compare_models_and_plot(dataset_types, num_samples=1000, num_epochs=20):
    from matplotlib.ticker import MaxNLocator

    # Configuración de los modelos
    resnet_rk = lambda input_dim: ResNet(input_dim, num_layers=20, hidden_dim=50, integration="RK")
    resnet_euler = lambda input_dim: ResNet(input_dim, num_layers=20, hidden_dim=50, integration="Euler")
    classical_models = {
        "SVM": SVC(kernel="rbf", random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    model_names = ["ResNet RK", "ResNet Euler"] + list(classical_models.keys())  # 6 métodos en total

    # Crear subplots de 6x4
    fig, axes = plt.subplots(6, 4, figsize=(20, 25))  # 6 filas, 4 columnas
    axes = axes.reshape(6, 4)  # Matriz para organizar filas y columnas

    # Diccionario para almacenar tiempos de ejecución y modelos entrenados
    execution_times = {model: [] for model in model_names}
    trained_models = {model: [] for model in model_names}

    # Etiquetas de las columnas (datasets)
    for col_idx, dataset_type in enumerate(dataset_types):
        axes[0, col_idx].set_title(dataset_type, fontsize=14, pad=20)

    # Etiquetas de las filas (modelos)
    for row_idx, model_name in enumerate(model_names):
        axes[row_idx, 0].set_ylabel(
            model_name, fontsize=12, rotation=0, labelpad=10, va="center", ha="right"
        )

    # Eliminar ticks de ejes internos para evitar superposición
    for i in range(6):  # 6 filas
        for j in range(4):  # 4 columnas
            if i < 5:  # No eliminar ticks de la última fila
                axes[i, j].xaxis.set_visible(False)
            if j > 0:  # No eliminar ticks de la primera columna
                axes[i, j].yaxis.set_visible(False)

    # Iterar sobre datasets y modelos
    for col_idx, dataset_type in enumerate(dataset_types):
        # Generar datos
        X, y = generate_data(dataset_type=dataset_type, n_samples=num_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy().flatten(), test_size=0.3, random_state=42, stratify=y.numpy().flatten()
        )

        # Normalizar características para modelos clásicos
        if len(X_train.shape) > 1 and X_train.shape[1] > 1:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Entrenar y almacenar modelos ResNet
        for row_idx, (model_name, resnet_model) in enumerate(
            [("ResNet RK", resnet_rk), ("ResNet Euler", resnet_euler)]
        ):
            input_dim = X.shape[1]
            model = resnet_model(input_dim)

            # Medir tiempo de entrenamiento
            start_time = time.time()
            model_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model_criterion = torch.nn.BCELoss()
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                ),
                batch_size=64,
                shuffle=True,
            )
            train_model(model, train_loader, model_optimizer, model_criterion, num_epochs)
            end_time = time.time()

            # Guardar tiempo y modelo entrenado
            execution_times[model_name].append(end_time - start_time)
            trained_models[model_name].append(model)

        # Entrenar y almacenar modelos clásicos
        for row_idx, (model_name, model) in enumerate(classical_models.items(), start=2):
            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()

            # Guardar tiempo y modelo entrenado
            execution_times[model_name].append(end_time - start_time)
            trained_models[model_name].append(model)

    # Mostrar tiempos de ejecución por dataset
    print("\nTiempos de Ejecución por Dataset:")
    for col_idx, dataset_type in enumerate(dataset_types):
        print(f"\nDataset: {dataset_type}")
        for model_name in model_names:
            avg_time = execution_times[model_name][col_idx]
            print(f"{model_name}: {avg_time:.4f} segundos ")
    # Graficar límites de decisión
    for col_idx, dataset_type in enumerate(dataset_types):
        for row_idx, model_name in enumerate(model_names):
            model = trained_models[model_name][col_idx]
            if isinstance(model, ResNet):  # ResNet
                plot_decision_boundary_subplot(axes[row_idx, col_idx], model, X_train, y_train)
            else:  # Modelos clásicos
                plot_decision_boundary_subplot(axes[row_idx, col_idx], model, X_train, y_train)

    # Mover nombres de datasets a la parte inferior
    for col_idx, dataset_type in enumerate(dataset_types):
        axes[-1, col_idx].set_xlabel(dataset_type, fontsize=14, labelpad=10)

    # Título general
    fig.suptitle("Frontera de decisión para varios modelos y conjuntos de datos", fontsize=16, y=0.95)

    # Ajustar diseño para evitar superposición
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.92])
    plt.show()



if __name__ == "__main__":
    # Datasets disponibles
    dataset_types = ["donut_1d", "donut_2d", "spiral_2d", "squares_2d"]

    # Comparar modelos y graficar
    compare_models_and_plot(dataset_types, num_epochs=50)