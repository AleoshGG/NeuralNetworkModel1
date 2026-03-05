import numpy as np
import numpy.linalg as la

from entities.data_results import DataResults
from entities.activation_functions import ActivationFunctions

class NeuronalNetwork:
    def __init__(self, path_dataset: str, skiprows = 1, activation='step'):
        self.dataset = np.loadtxt(path_dataset, delimiter=",", skiprows=skiprows)
        self.Y_values = self.__slicing_Y_values()
        self.X_values = self.__slicing_X_values()
        self.W_values = self.__generate_W_values()
        self.__set_activation_function(activation)

    def __slicing_Y_values(self):
        Y_values = self.dataset[:, -1]
        Y_values = Y_values.reshape(-1, 1)
        return Y_values
    
    def __slicing_X_values(self):
        X_values = self.dataset[:, :-1]
        m = X_values.shape[0]
        colum_ones = np.ones((m,1))
        X_values = np.hstack((colum_ones, X_values))
        return X_values
    
    def __generate_W_values(self):
        m = self.X_values.shape[1]
        W_values = np.random.randn(m, 1)
        return W_values
    
    def show_data_init(self):
        print(f"Dataset Inicial: \n{self.dataset.shape}")
        print(f"\nValores Y: \n{self.Y_values.shape}")
        print(f"\nValores X: \n{self.X_values.shape}")
        print(f"\nValores W: \n{self.W_values.shape}")

    def __set_activation_function(self, activation_name):
        """Mapea el string ingresado a la función real de la clase ActivationFunctions."""
        activations = {
            'linear': ActivationFunctions.linear,
            'step': ActivationFunctions.step,
            'bipolar_step': ActivationFunctions.bipolar_step,
            'relu': ActivationFunctions.relu,
            'sigmoid': ActivationFunctions.sigmoid
        }
        
        if activation_name not in activations:
            raise ValueError(f"Función '{activation_name}' no soportada. Usa: {list(activations.keys())}")
            
        self.activation_func = activations[activation_name]
    
    def train(self, epochs: int, lambda_value: float) -> DataResults:

        history_evolution = DataResults(errors=[], weights=[])

        for i in range(epochs):
            y_c = self.activation_func(self.X_values @ self.W_values)

            error = y_c - self.Y_values
            delta_w = - (lambda_value * (self.X_values.T @ error))
            self.W_values += delta_w

            
            error_magnitude = la.norm(error)

            history_evolution.errors.append(error_magnitude)
            history_evolution.weights.append(self.W_values.copy())
        
        return history_evolution
            
    def __funtion_activation(self, u):
        # np.where funciona así: np.where(condición, valor_si_verdad, valor_si_falso)
        return np.where(u >= 0, 1, 0)
 