import matplotlib.pyplot as plt
import numpy as np

class DataResults:
    def __init__(self, errors: list, weights: list):
        self.errors = errors
        self.weights = weights

    def show_final_metrics(self):
        """Muestra en consola los valores de la última época."""
        if not self.errors:
            print("No hay datos de entrenamiento.")
            return

        print("\n=== Resultados Finales ===")
        print(f"Error Final (Norma): {self.errors[-1]:.6f}")
        print("Pesos Finales:")
        print(self.weights[-1])
        print("========================\n")

    def plot_learning_curve(self):
        """Genera una gráfica de la evolución del error."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.errors, label='Error Total', color='blue')
        
        plt.title('Curva de Aprendizaje')
        plt.xlabel('Épocas')
        plt.ylabel('Magnitud del Error (Norma)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        print("Mostrando gráfica...")
        plt.show()