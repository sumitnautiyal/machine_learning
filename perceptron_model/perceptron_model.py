import numpy as np
import matplotlib.pyplot as plt

def generate_linearly_separable_data(num_samples, num_features=2, random_seed=None):
    """
    Generates a random linearly separable dataset.

    Parameters:
    num_samples (int): Number of samples to generate.
    num_features (int): Number of features (dimensions) excluding the bias term.
    random_seed (int): Seed for random number generator.

    Returns:
    tuple: Tuple containing the feature matrix X, labels y, and target weights w_target.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    target_weights = np.random.uniform(-1, 1, num_features + 1)  # num_features dimensions + bias term
    feature_matrix = np.random.uniform(-1, 1, (num_samples, num_features))
    feature_matrix = np.c_[np.ones(num_samples), feature_matrix]  # Add a bias term (1) to each point
    labels = np.sign(feature_matrix @ target_weights)  # Label based on the target hyperplane
    return feature_matrix, labels, target_weights

class Perceptron:
    def __init__(self, max_iterations=1000):
        """
        Initializes the Perceptron with a maximum number of iterations.

        Parameters:
        max_iterations (int): Maximum number of iterations for the learning algorithm.
        """
        self.max_iterations = max_iterations
        self.weights = None
        self.iterations = 0

    def fit(self, feature_matrix, labels):
        """
        Fits the Perceptron model to the data.

        Parameters:
        feature_matrix (ndarray): Feature matrix.
        labels (ndarray): Labels.

        Returns:
        int: Number of iterations until convergence.
        """
        self.weights = np.zeros(feature_matrix.shape[1])  # Initialize weights
        self.iterations = 0
        while self.iterations < self.max_iterations:
            misclassified_indices = [i for i in range(len(feature_matrix)) if np.sign(feature_matrix[i] @ self.weights) != labels[i]]
            if not misclassified_indices:
                break  # Stop if no misclassified points
            random_index = np.random.choice(misclassified_indices)  # Random misclassified point
            self.weights += labels[random_index] * feature_matrix[random_index]
            self.iterations += 1
        return self.iterations

    def predict(self, feature_matrix):
        """
        Predicts the labels for the given feature matrix.

        Parameters:
        feature_matrix (ndarray): Feature matrix.

        Returns:
        ndarray: Predicted labels.
        """
        return np.sign(feature_matrix @ self.weights)

def plot_data_and_hypotheses(feature_matrix, labels, target_weights, learned_weights=None, title=""):
    """
    Plots the data points and the target and learned hypotheses.

    Parameters:
    feature_matrix (ndarray): Feature matrix.
    labels (ndarray): Labels.
    target_weights (ndarray): Target weights.
    learned_weights (ndarray): Learned weights (optional).
    title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data points
    for label, marker, color in zip([-1, 1], ['o', 's'], ['red', 'blue']):
        ax.scatter(feature_matrix[labels == label][:, 1], feature_matrix[labels == label][:, 2], marker=marker, color=color, label=f"Class {label}")

    # Plot target function
    x_vals = np.linspace(-1, 1, 100)
    y_vals_target = -(target_weights[0] + target_weights[1] * x_vals) / target_weights[2]
    ax.plot(x_vals, y_vals_target, 'k--', label="Target Function f")

    # Plot learned hypothesis if available
    if learned_weights is not None:
        y_vals_learned = -(learned_weights[0] + learned_weights[1] * x_vals) / learned_weights[2]
        ax.plot(x_vals, y_vals_learned, 'g-', label="Learned Hypothesis g")

    # Configure plot
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()
    plt.grid()
    plt.show()

def run_experiment(num_samples, num_features=2, random_seed=None):
    """
    Runs the Perceptron Learning Algorithm experiment.

    Parameters:
    num_samples (int): Number of samples.
    num_features (int): Number of features (dimensions) excluding the bias term.
    random_seed (int): Seed for random number generator.

    Returns:
    int: Number of iterations until convergence.
    """
    feature_matrix, labels, target_weights = generate_linearly_separable_data(num_samples, num_features, random_seed)
    perceptron = Perceptron()
    iterations = perceptron.fit(feature_matrix, labels)
    title = f"PLA: Target Function and Learned Hypothesis (N={num_samples}, d={num_features})\nConverged in {iterations} iterations"
    if num_features == 2:
        plot_data_and_hypotheses(feature_matrix, labels, target_weights, perceptron.weights, title)
    return iterations

def randomized_updates_experiment(num_samples, num_features=2, num_trials=100, random_seed=None):
    """
    Runs the randomized updates experiment for the Perceptron Learning Algorithm.

    Parameters:
    num_samples (int): Number of samples.
    num_features (int): Number of features (dimensions) excluding the bias term.
    num_trials (int): Number of trials.
    random_seed (int): Seed for random number generator.

    Returns:
    list: List of iterations until convergence for each trial.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    iterations_list = []
    for _ in range(num_trials):
        feature_matrix, labels, _ = generate_linearly_separable_data(num_samples, num_features)
        perceptron = Perceptron()
        iterations = perceptron.fit(feature_matrix, labels)
        iterations_list.append(iterations)
    plt.hist(iterations_list, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f"Histogram of PLA Convergence Iterations (N={num_samples}, d={num_features}, Trials={num_trials})")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
    return iterations_list

# Running experiments for different dataset sizes
dataset_sizes = [20, 100, 1000]
experiment_results = {}
for size in dataset_sizes:
    iterations = run_experiment(size, num_features=2, random_seed=42)
    experiment_results[size] = iterations

# Experiment for 10 dimensions
num_samples_10d = 1000
iterations_10d = run_experiment(num_samples_10d, num_features=10, random_seed=42)
print(f"PLA converged in {iterations_10d} iterations for N={num_samples_10d}, d=10.")

# Run randomized updates experiment
randomized_updates_experiment(1000, num_features=2, num_trials=100, random_seed=42)