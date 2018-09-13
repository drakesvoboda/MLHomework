import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(r, y):
    return ((r - y)**2)

def mse_prime(r, y):
    return 2 * (r - y)

class Regressor:
    def __init__(self, input_dim):
        self.weights = np.random.rand(input_dim + 1)
        self.losses = []

    def inference(self, x):
        return sigmoid(np.dot(x, self.weights))

    def compute_gradient(self, batch):
        x, r = batch
        logits = np.dot(x, self.weights)
        y = sigmoid(logits)
        return np.dot(np.transpose(x), -1 * mse_prime(r, y) * sigmoid_prime(logits))

    def sgd(self, inputs, labels, batch_size, lr):
        correct = 0
        n = 0
        inputs = np.array([np.concatenate(([-1], x)) for x in inputs])
        batches = [(inputs[k:k+batch_size], labels[k:k+batch_size]) for k in range(0, len(inputs), batch_size)]
        for batch in batches:
            x, r = batch
            y = self.inference(x)
            self.losses.append(np.mean(mse(r, y)))
            n += len(x)
            y = np.where(y > .5, 1, 0)          
            correct += np.sum(r == y)
            cost_grad = self.compute_gradient(batch)
            self.weights -= lr * cost_grad


        print("Accuracy: ", correct / n)

    def plot_loss(self):
        fig, ax = plt.subplots()

        ax.set_ylabel("Loss")
        ax.set_xlabel("Iterations")

        ax.plot(range(len(self.losses)), self.losses)
        

    