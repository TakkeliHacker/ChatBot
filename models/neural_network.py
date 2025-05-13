import numpy as np
import pickle
import time
import os
from datetime import datetime

# Aktivasyon fonksiyonları (pickle uyumlu dış fonksiyonlar)
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivative(self.input))

class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_derivative)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_derivative)

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_derivative)

class Softmax(Layer):
    def forward(self, input_data):
        self.input = input_data
        exp_values = np.exp(input_data - np.max(input_data, axis=0, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, output_gradient, learning_rate):
        n_samples = self.output.shape[1]
        return output_gradient / n_samples

class CrossEntropy:
    def forward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if len(y_true.shape) == 1 or y_true.shape[0] == 1:
            return -np.mean(y_true * np.log(y_pred))
        else:
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

    def backward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred

class SimpleNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def predict(self, input_data):
        samples = input_data.shape[0]
        result = []
        for i in range(samples):
            output = input_data[i].reshape(-1, 1)
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return np.array(result).reshape(samples, -1)

    def fit(self, x_train, y_train, epochs, learning_rate, batch_size=32, verbose=True):
        samples = x_train.shape[0]
        training_history = {'loss': []}

        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            permutation = np.random.permutation(samples)
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for i in range(0, samples, batch_size):
                batch_end = min(i + batch_size, samples)
                x_batch = x_train_shuffled[i:batch_end]
                y_batch = y_train_shuffled[i:batch_end]
                batch_size_actual = batch_end - i
                batch_loss = 0

                for j in range(batch_size_actual):
                    output = x_batch[j].reshape(-1, 1)
                    for layer in self.layers:
                        output = layer.forward(output)
                    y_true = y_batch[j].reshape(-1, 1)
                    batch_loss += self.loss.forward(output, y_true)
                    grad = self.loss.backward(output, y_true)
                    for layer in reversed(self.layers):
                        grad = layer.backward(grad, learning_rate)

                batch_loss /= batch_size_actual
                epoch_loss += batch_loss * batch_size_actual

            epoch_loss /= samples
            training_history['loss'].append(epoch_loss)

            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch + 1}/{epochs}, loss: {epoch_loss:.4f}, time: {epoch_time:.2f}s")

        return training_history

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model

class SequenceToSequenceModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.encoder = SimpleNeuralNetwork()
        self.encoder.add(Dense(input_dim, hidden_dim))
        self.encoder.add(Tanh())

        self.decoder = SimpleNeuralNetwork()
        self.decoder.add(Dense(hidden_dim, output_dim))
        self.decoder.add(Softmax())

        self.loss = CrossEntropy()
        self.encoder.set_loss(self.loss)
        self.decoder.set_loss(self.loss)
        self.hidden_dim = hidden_dim

    def predict(self, input_sequence):
        hidden_state = self.encoder.predict(input_sequence)
        output_sequence = self.decoder.predict(hidden_state)
        return output_sequence

    def fit(self, input_sequences, output_sequences, epochs, learning_rate, batch_size=32, verbose=True):
        hidden_targets = np.zeros((input_sequences.shape[0], self.hidden_dim))
        encoder_history = self.encoder.fit(
            input_sequences,
            hidden_targets,
            epochs,
            learning_rate,
            batch_size,
            verbose=verbose
        )

        hidden_states = self.encoder.predict(input_sequences)

        decoder_history = self.decoder.fit(
            hidden_states,
            output_sequences,
            epochs,
            learning_rate,
            batch_size,
            verbose=verbose
        )

        return {
            'encoder_loss': encoder_history['loss'],
            'decoder_loss': decoder_history['loss']
        }

    def save(self, filepath_prefix):
        encoder_path = f"{filepath_prefix}_encoder.pkl"
        decoder_path = f"{filepath_prefix}_decoder.pkl"
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)
        model_info = {
            'hidden_dim': self.hidden_dim,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(f"{filepath_prefix}_info.pkl", 'wb') as file:
            pickle.dump(model_info, file)

    @classmethod
    def load(cls, filepath_prefix):
        encoder_path = f"{filepath_prefix}_encoder.pkl"
        decoder_path = f"{filepath_prefix}_decoder.pkl"
        info_path = f"{filepath_prefix}_info.pkl"
        with open(info_path, 'rb') as file:
            model_info = pickle.load(file)
        encoder = SimpleNeuralNetwork.load(encoder_path)
        decoder = SimpleNeuralNetwork.load(decoder_path)
        input_dim = encoder.layers[0].weights.shape[1]
        hidden_dim = model_info['hidden_dim']
        output_dim = decoder.layers[0].weights.shape[0]
        model = cls(input_dim, hidden_dim, output_dim)
        model.encoder = encoder
        model.decoder = decoder
        return model
