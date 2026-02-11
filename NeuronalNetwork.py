import numpy as np

class NeuronalNetwork:
    def __init__(self, path_dataset: str):
        self.dataset = np.loadtxt(path_dataset, delimiter=",", skiprows=1)
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