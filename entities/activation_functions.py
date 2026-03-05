import numpy as np

class ActivationFunctions:
    @staticmethod
    def linear(u):
        """Función Lineal: f(x) = x. Usada para Regresión."""
        return u

    @staticmethod
    def step(u, threshold=0):
        """Escalón Binario: f(x) = 1 si x >= umbral, sino 0. Usada en Perceptrón Simple."""
        return np.where(u >= threshold, 1, 0)

    @staticmethod
    def bipolar_step(u, threshold=0):
        """Escalón Bipolar: f(x) = 1 si x >= umbral, sino -1."""
        return np.where(u >= threshold, 1, -1)

    @staticmethod
    def relu(u):
        """ReLU: f(x) = max(0, x). Estándar en redes modernas."""
        return np.maximum(0, u)

    @staticmethod
    def sigmoid(u):
        """Sigmoide: f(x) = 1 / (1 + e^-x). Usada para probabilidades."""
        # np.clip evita desbordamientos matemáticos (overflow) si 'u' es muy grande o pequeño
        u_clipped = np.clip(u, -500, 500) 
        return 1 / (1 + np.exp(-u_clipped))