import matplotlib.pyplot as plt
import numpy as np
import csv

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
    
    def save_report_to_csv(self, filename="reporte_red_neuronal.csv"):
        """Guarda un reporte CSV con: Pesos Iniciales, Pesos Finales y Error Final."""
        if not self.weights or not self.errors:
            print("No hay datos para generar el reporte.")
            return

        try:
            initial_w = np.array(self.weights[0]).flatten()
            final_w = np.array(self.weights[-1]).flatten()
            final_err = self.errors[-1]

            headers = []
            row_data = []

            for i in range(len(initial_w)):
                headers.append(f"Peso_Inicial_W{i}")
                row_data.append(initial_w[i])

            for i in range(len(final_w)):
                headers.append(f"Peso_Final_W{i}")
                row_data.append(final_w[i])

            headers.append("Error_Final_Magnitud")
            row_data.append(final_err)

            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)   
                writer.writerow(row_data)  

            print(f"Reporte guardado exitosamente en: {filename}")

        except Exception as e:
            print(f"Error al guardar el reporte: {e}")

    def plot_learning_curve(self, filename):
        """Genera una gráfica de la evolución del error."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.errors, label='Error Total', color='blue')
        
        plt.title('Curva de Aprendizaje')
        plt.xlabel('Épocas')
        plt.ylabel('Magnitud del Error (Norma)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.savefig(filename)
    
    def plot_weights_curve(self, filename):
        """Genera una gráfica de la evolución de cada peso individualmente."""
        if not self.weights:
            print("No hay datos de pesos.")
            return

        weights_history = np.array(self.weights).squeeze()
        
        if weights_history.ndim == 1:
            weights_history = weights_history.reshape(-1, 1)

        plt.figure(figsize=(10, 5))
        
        num_weights = weights_history.shape[1]
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i in range(num_weights):
            color = colors[i % len(colors)]
            plt.plot(weights_history[:, i], label=f'Peso $W_{i}$', color=color)

        plt.title('Evolución de los Pesos')
        plt.xlabel('Épocas')
        plt.ylabel('Valor del Peso')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(filename)

    