import numpy as np
from three_layer_neural_network.NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    iris = load_iris()
    X = iris.data  # 4 features: sepal length, sepal width, petal length, petal width
    y = iris.target.reshape(-1, 1)  # Labels: 0, 1, 2 (for 3 different species of iris flowers)
    return X, y

class Layer:
    def __init__(self, input_size, output_size, activation_function, activation_function_prime, actFun_type):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation_function = activation_function
        self.activation_function_prime = activation_function_prime
        self.actFun_type = actFun_type

    def feedforward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function(self.z, self.actFun_type)  # Pass the activation function type here
        return self.a

    def backprop(self, output_gradient, learning_rate, l2_lambda):
        activation_gradient = self.activation_function_prime(self.z,
                                                             self.actFun_type)  # Pass the activation function type here
        delta = output_gradient * activation_gradient

        weight_gradient = np.dot(self.inputs.T, delta) + l2_lambda * self.weights
        bias_gradient = np.sum(delta, axis=0, keepdims=True)
        input_gradient = np.dot(delta, self.weights.T)

        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient

        return input_gradient

class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, actFun_type='tanh', l2_lambda=0.01):
        super().__init__(layer_sizes[0], layer_sizes[1], layer_sizes[-1], actFun_type, reg_lambda=l2_lambda)
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], self.actFun, self.diff_actFun, actFun_type))  # pass actFun_type here
        self.l2_lambda = l2_lambda

    def feedforward(self, X):
        a = X
        for layer in self.layers:
            a = layer.feedforward(a)
        return a

    def backprop(self, X, y, learning_rate):
        m = y.shape[0]
        output = self.feedforward(X)
        loss_gradient = (output - y) / m

        for layer in reversed(self.layers):
            loss_gradient = layer.backprop(loss_gradient, learning_rate, self.l2_lambda)

    def calculate_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = np.mean(np.square(y_pred - y_true)) / 2
        l2_loss = (self.l2_lambda / 2) * sum([np.sum(layer.weights ** 2) for layer in self.layers])
        return loss + l2_loss

    def fit(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.feedforward(X)
            loss = self.calculate_loss(y_pred, y)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss}")
            self.backprop(X, y, learning_rate)

    def predict(self, X):
        '''
        Predicts the output of the network for a given input X
        :param X: input data
        :return: output predictions
        '''
        return self.feedforward(X)

    def visualize_decision_boundary(self, X, y):
        '''
        Visualize decision boundary created by the trained network
        :param X: input data
        :param y: true labels
        :return: None
        '''
        self.plot_decision_boundary(lambda x: self.predict(x), X, y)

    def plot_decision_boundary(self, pred_func, X, y):
        '''
        Helper function to plot decision boundary
        :param pred_func: a function that takes input data and outputs predictions
        :param X: input data
        :param y: true labels
        :return: None
        '''
        # Set min and max values for the grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # Generate a grid of points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict output for each point in the grid
        Z = pred_func(grid)
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        # Plot contour and decision boundary
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

        # Plot the original data points
        plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Decision Boundary")
        plt.show()

def main():
    X, y = generate_data()
    y = y.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    layer_sizes = [2, 5, 5, 3]

    network = DeepNeuralNetwork(layer_sizes, actFun_type='sigmoid')
    network.fit(X_train_pca, y_train, epochs=500, learning_rate=0.01)
    network.visualize_decision_boundary(X_train_pca, y_train)


if __name__ == "__main__":
    main()
