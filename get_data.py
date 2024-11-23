import numpy as np

np.random.seed(42)

def get_data_donut_1d(n_samples=1000, nx=100):
    """
    Generates a 1D donut dataset with two classes.
    
    Args:
        n_samples (int): Number of samples to generate.
        nx (int): Grid resolution for generating densities.

    Returns:
        features (numpy.ndarray): Features of shape (2, n_samples).
        labels (numpy.ndarray): Labels of shape (n_samples,).
    """
    # Generate grid and radial function
    z1 = np.linspace(-1, 1, nx)
    z2 = 0
    r1, r2 = np.meshgrid(z1, np.array([z2]))
    rad = lambda r1, r2: np.sqrt(r1**2 + r2**2)
    
    # Define densities for two classes
    mean_rad1, sigma1 = 0, 0.1
    density1 = np.exp(-((rad(r1, r2) - mean_rad1)**2) / (2 * sigma1**2))
    density1 /= np.sum(density1)

    mean_rad2, sigma2 = 0.5, 0.1
    density2 = np.exp(-((rad(r1, r2) - mean_rad2)**2) / (2 * sigma2**2))
    density2 /= np.sum(density2)

    # Number of samples per class
    m1 = int(np.ceil(n_samples / 3))
    m2 = n_samples - m1

    # Generate samples using densities
    def sample_from_density(z1, z2, density, n_samples):
        samples = []
        for _ in range(n_samples):
            idx = np.random.choice(len(z1), p=density.flatten())
            x, y = z1[idx % nx], z2 + idx // nx
            samples.append([x, y])
        return np.array(samples).T

    features_class1 = sample_from_density(z1, z2, density1, m1)
    features_class2 = sample_from_density(z1, z2, density2, m2)
    features = np.hstack([features_class1, features_class2])

    # Labels: 1 for class 1, 0 for class 2
    labels = np.array([1] * m1 + [0] * m2)

    # Shuffle data
    shuffle_idx = np.random.permutation(n_samples)
    features = features[:, shuffle_idx]
    labels = labels[shuffle_idx]

    # Add noise to features
    features += 0.1 * np.random.randn(*features.shape)

    return features, labels


def get_data_donut_2d(n_samples=100, nx=100):
    """
    Generates a 2D donut dataset with two classes.
    
    Args:
        n_samples (int): Number of samples to generate.
        nx (int): Grid resolution for generating densities.

    Returns:
        features (numpy.ndarray): Features of shape (2, n_samples).
        labels (numpy.ndarray): Labels of shape (n_samples,).
    """
    # Generate grid and radial function
    r = np.linspace(-1, 1, nx)
    r1, r2 = np.meshgrid(r, r)
    rad = lambda r1, r2: np.sqrt(r1**2 + r2**2)
    
    # Define densities for two classes
    mean_rad1, sigma1 = 0, 0.1
    density1 = np.exp(-((rad(r1, r2) - mean_rad1)**2) / (2 * sigma1**2))
    density1 /= np.sum(density1)

    mean_rad2, sigma2 = 0.5, 0.1
    density2 = np.exp(-((rad(r1, r2) - mean_rad2)**2) / (2 * sigma2**2))
    density2 /= np.sum(density2)

    # Number of samples per class
    m1 = int(np.ceil(n_samples / 2))
    m2 = n_samples - m1

    # Generate samples using densities
    def sample_from_density(r, density, n_samples):
        samples = []
        for _ in range(n_samples):
            idx = np.random.choice(len(r)**2, p=density.flatten())
            x_idx, y_idx = np.unravel_index(idx, (len(r), len(r)))
            samples.append([r[x_idx], r[y_idx]])
        return np.array(samples).T

    features_class1 = sample_from_density(r, density1, m1)
    features_class2 = sample_from_density(r, density2, m2)
    features = np.hstack([features_class1, features_class2])

    # Labels: 1 for class 1, 0 for class 2
    labels = np.array([1] * m1 + [0] * m2)

    # Shuffle data
    shuffle_idx = np.random.permutation(n_samples)
    features = features[:, shuffle_idx]
    labels = labels[shuffle_idx]

    # Add noise to features
    features += 0.1 * np.random.randn(*features.shape)

    return features, labels


def get_data_spiral_2d(n_samples=1000):
    """
    Generates a 2D spiral dataset with two classes.
    
    Args:
        n_samples (int): Number of samples to generate.

    Returns:
        features (numpy.ndarray): Features of shape (2, n_samples).
        labels (numpy.ndarray): Labels of shape (n_samples,).
    """
    # Number of samples per class
    m1 = int(np.ceil(n_samples / 2))
    m2 = n_samples - m1

    # Spiral parameters
    n_turns = 1
    phi1, phi2 = np.pi, np.pi + np.pi
    r1 = np.linspace(0.1, 1, m1)
    r2 = np.linspace(0.1, 1, m2)
    a1 = np.linspace(0.1, 2 * np.pi * n_turns, m1)
    a2 = np.linspace(0.1, 2 * np.pi * n_turns, m2)

    # Generate spirals
    d1 = np.array([r1 * np.cos(a1 + phi1), r1 * np.sin(a1 + phi1)])
    d2 = np.array([r2 * np.cos(a2 + phi2), r2 * np.sin(a2 + phi2)])
    features = np.hstack([d1, d2])

    # Labels: 1 for spiral 1, 0 for spiral 2
    labels = np.array([1] * m1 + [0] * m2)

    # Shuffle data
    shuffle_idx = np.random.permutation(n_samples)
    features = features[:, shuffle_idx]
    labels = labels[shuffle_idx]

    # Add noise to features
    features += 0.05 * np.random.randn(*features.shape)

    return features, labels


def get_data_squares_2d(n_samples=1000):
    """
    Generates a 2D checkerboard dataset with two classes.
    
    Args:
        n_samples (int): Number of samples to generate.

    Returns:
        features (numpy.ndarray): Features of shape (2, n_samples).
        labels (numpy.ndarray): Labels of shape (n_samples,).
    """
    # Generate random points in a square domain
    a = 0.7
    features = 2 * a * (np.random.rand(2, n_samples) - 0.5)

    # Define labels based on quadrant
    labels = (features[0, :] * features[1, :] > 0).astype(int)

    # Add noise to features
    features += 0.1 * np.random.randn(*features.shape)

    return features, labels

# Testing all functions
#donut_1d_features, donut_1d_labels = get_data_donut_1d(500, 100)
#donut_2d_features, donut_2d_labels = get_data_donut_2d(500)
#spiral_2d_features, spiral_2d_labels = get_data_spiral_2d(500)
#squares_2d_features, squares_2d_labels = get_data_squares_2d(500)

import matplotlib.pyplot as plt

def plot_dataset(features, labels, title="Dataset Visualization"):
    """
    Plots a 2D dataset with two classes.
    
    Args:
        features (numpy.ndarray): Features of shape (2, n_samples).
        labels (numpy.ndarray): Labels of shape (n_samples,).
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(
        features[0, labels == 1], features[1, labels == 1],
        color="red", label="Class 1", alpha=0.6
    )
    plt.scatter(
        features[0, labels == 0], features[1, labels == 0],
        color="blue", label="Class 0", alpha=0.6
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# Main
if __name__ == "__main__":
    print("Select a dataset to visualize:")
    print("1: Donut 1D")
    print("2: Donut 2D")
    print("3: Spiral 2D")
    print("4: Squares 2D")
    choice = int(input("Enter your choice (1-4): ").strip())

    n_samples = 500
    if choice == 1:
        features, labels = get_data_donut_1d(n_samples)
        plot_dataset(features, labels, title="Donut 1D Dataset")
    elif choice == 2:
        features, labels = get_data_donut_2d(n_samples)
        plot_dataset(features, labels, title="Donut 2D Dataset")
    elif choice == 3:
        features, labels = get_data_spiral_2d(n_samples)
        plot_dataset(features, labels, title="Spiral 2D Dataset")
    elif choice == 4:
        features, labels = get_data_squares_2d(n_samples)
        plot_dataset(features, labels, title="Squares 2D Dataset")
    else:
        print("Invalid choice. Please select a number between 1 and 4.")
