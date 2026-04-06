import numpy as np
import pickle
import json
import os
from datetime import datetime


class ModelSaver:
   
    def __init__(self, model_dir='models'):

        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def save_neural_network(self, nn, history, filename='neural_network_model'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"{filename}_{timestamp}.pkl")

        model_data = {
            'layer_sizes': nn.layer_sizes,
            'learning_rate': nn.learning_rate,
            'weights': nn.weights,
            'biases': nn.biases,
            'history': history,
            'timestamp': timestamp,
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        return model_path

    def save_numpy_weights(self, nn, filename='weights'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_path = os.path.join(self.model_dir, f"{filename}_{timestamp}.npz")

        # Konversi ke numpy arrays untuk penyimpanan yang lebih efisien
        weights_dict = {}
        for i, w in enumerate(nn.weights):
            weights_dict[f'weight_{i}'] = w
        for i, b in enumerate(nn.biases):
            weights_dict[f'bias_{i}'] = b

        np.savez_compressed(weights_path, **weights_dict)

        return weights_path

    def save_training_results(self, history, metrics, hyperparameters, filename='training_results'):
       
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.model_dir, f"{filename}_{timestamp}.json")

        results = {
            'timestamp': timestamp,
            'hyperparameters': hyperparameters,
            'training_history': {
                k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
                for k, vals in history.items()
            },
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in metrics.items()},
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results_path

    def save_conv_layer(self, conv, filename='conv_layer'):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_path = os.path.join(self.model_dir, f"{filename}_{timestamp}.pkl")

        conv_data = {
            'pool_size': conv.pool_size,
            'timestamp': timestamp,
        }

        with open(conv_path, 'wb') as f:
            pickle.dump(conv_data, f)

        return conv_path

    def save_full_model(self, nn, conv, history, metrics, hyperparameters, filename='full_model'):

        saved_files = {
            'neural_network': self.save_neural_network(nn, history, f'{filename}_nn'),
            'weights_npz': self.save_numpy_weights(nn, f'{filename}_weights'),
            'conv_layer': self.save_conv_layer(conv, f'{filename}_conv'),
            'results': self.save_training_results(history, metrics, hyperparameters, 
                                                  f'{filename}_results'),
        }

        return saved_files

    @staticmethod
    def load_neural_network(model_path):
        """
        Muat neural network dari file pkl

        Parameters:
        -----------
        model_path : str
            Path ke file model

        Returns:
        --------
        dict : Data model (weights, biases, history, etc)
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data

    @staticmethod
    def load_weights_npz(weights_path):
     
        loaded = np.load(weights_path, allow_pickle=True)
        return dict(loaded)

    @staticmethod
    def load_training_results(results_path):

        with open(results_path, 'r') as f:
            results = json.load(f)
        return results


def save_complete_training(nn, conv, history, metrics, hyperparameters, 
                          logger=None, model_dir='models'):
 
    saver = ModelSaver(model_dir=model_dir)
    saved_files = saver.save_full_model(
        nn, conv, history, metrics, hyperparameters
    )

    if logger:
        logger.info("=" * 60)
        logger.info("MODEL TRAINING RESULTS SAVED")
        logger.info("=" * 60)
        for key, path in saved_files.items():
            logger.info(f"✓ {key.replace('_', ' ').title()}: {path}")
        logger.info("=" * 60)

    return saved_files
