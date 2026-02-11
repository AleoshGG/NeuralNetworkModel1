import numpy as np

from entities.data_results import DataResults

class NeuronalNetwork:
    def __init__(self, path_dataset: str, skiprows = 1):
        self.dataset = np.loadtxt(path_dataset, delimiter=",", skiprows=skiprows)
        self.Y_values = self.__slicing_Y_values()
        self.X_values = self.__slicing_X_values()
        self.W_values = self.__generate_W_values()

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
    
    def train(self, epochs: int, lambda_value: float) -> DataResults:

        history_evolution = DataResults(errors=[],weights=self.W_values)
        m = self.X_values.shape[0]

        for i in range(epochs):
            y_c = self.__funtion_activation(self.X_values @ self.W_values)

            error = y_c - self.Y_values
            delta_w = - (lambda_value * (self.X_values.T @ error))
            self.W_values += delta_w

            history_evolution.errors.append(error)
            history_evolution.weights(self.W_values)
        
        return history_evolution
            
    def __funtion_activation(self, u):
        if u < 0: return 0
        elif u == 0: return 1
 