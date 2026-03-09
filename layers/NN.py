import numpy as np
from tqdm import tqdm
from activation.Activation import relu, relu_derivative, softmax, cross_entropy_loss


class NeuralNetwork:

    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1

        np.random.seed(seed)
        self.weights = []
        self.biases = []


        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)


    def forward(self, X):

        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        self._activations = [X]
        self._z_values = []

        current = X
        for i in range(self.num_layers):
            z = current @ self.weights[i] + self.biases[i]
            self._z_values.append(z)

            if i < self.num_layers - 1:
                current = relu(z)
            else:
                current = softmax(z)

            self._activations.append(current)

        if single:
            return self._activations[-1].flatten()
        return self._activations[-1]

    def backward(self, y_true):

        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)

        m = y_true.shape[0]

        delta = self._activations[-1] - y_true

        self._dW = []
        self._db = []

        for i in range(self.num_layers - 1, -1, -1):
            dW = (self._activations[i].T @ delta) / m
            db = np.mean(delta, axis=0, keepdims=True)

            self._dW.insert(0, dW)
            self._db.insert(0, db)

            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(self._z_values[i - 1])

    def _update_weights(self):
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * self._dW[i]
            self.biases[i] -= self.learning_rate * self._db[i]

    def train_step(self, X_batch, y_batch):

        y_pred = self.forward(X_batch)
        loss = cross_entropy_loss(y_pred, y_batch)
        self.backward(y_batch)
        self._update_weights()

        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        acc = np.mean(pred_labels == true_labels)
        return loss, acc

    def predict(self, X):

        probs = self.forward(X)
        if probs.ndim == 1:
            return np.argmax(probs)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y, y_labels=None):

        y_pred = self.forward(X)
        loss = cross_entropy_loss(y_pred, y)
        pred_labels = np.argmax(y_pred, axis=1)

        if y_labels is not None:
            true_labels = y_labels
        else:
            true_labels = np.argmax(y, axis=1)

        acc = np.mean(pred_labels == true_labels)
        return loss, acc

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, logger=None):

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
        for epoch in pbar:

            idx = np.random.permutation(X_train.shape[0])
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]


            epoch_loss, epoch_acc = self.train_step(X_shuf, y_shuf)
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

     
            val_loss, val_acc = None, None
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

   
            postfix = {"loss": f"{epoch_loss:.4f}", "acc": f"{epoch_acc:.4f}"}
            if val_loss is not None:
                postfix["val_loss"] = f"{val_loss:.4f}"
                postfix["val_acc"] = f"{val_acc:.4f}"
            pbar.set_postfix(postfix)

            if logger:
                logger.log_epoch(epoch, epochs, epoch_loss, epoch_acc, val_loss, val_acc)

        return history

    def info(self, logger=None):
        lines = [
            "Neural Network Architecture:",
            f"  Layers        : {self.layer_sizes}",
            f"  Learning rate : {self.learning_rate}",
        ]
        total = 0
        for i in range(self.num_layers):
            p = self.weights[i].size + self.biases[i].size
            total += p
            act = "ReLU" if i < self.num_layers - 1 else "Softmax"
            lines.append(f"  Layer {i+1}: {self.layer_sizes[i]:>5} -> {self.layer_sizes[i+1]:>4}  ({act})  [{p:,} params]")
        lines.append(f"  Total parameters: {total:,}")
        for line in lines:
            if logger:
                logger.info(line)
            else:
                print(line)
