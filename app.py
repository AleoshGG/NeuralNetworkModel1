import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NeuronalNetwork import NeuronalNetwork

lambdas = [0.000001, 1, 10, 0.89, 3.1416]
epocas = [1000, 150, 50, 800, 360]
dataset_path = 'datasets/A233333.csv'

resultados_resumen = []
pesos_historial_31416 = None

plt.figure(figsize=(12, 6))

print("Iniciando experimentos...\n")
for lam, ep in zip(lambdas, epocas):
    # inicien con exactamente los mismos pesos aleatorios y la comparación sea justa.
    np.random.seed(42) 
    
    nn = NeuronalNetwork(dataset_path, skiprows=0)
    
    # Guardamos los pesos iniciales (aplanados para la tabla)
    w_inicial = nn.W_values.copy().flatten()
    
    # Entrenamos
    print(f"Entrenando -> Lambda: {lam:^8} | Épocas: {ep}")
    resultados = nn.train(epochs=ep, lambda_value=lam)
    
    # Extraemos datos finales
    w_final = resultados.weights[-1].flatten()
    error_final = resultados.errors[-1]
    
    # --- Guardar datos para la tabla ---
    resultados_resumen.append({
        'Lambda': lam,
        'Epocas': ep,
        'W_inicial': np.round(w_inicial, 4).tolist(),
        'W_final': np.round(w_final, 4).tolist(),
        '|Error_Final|': round(error_final, 6)
    })
    
    # --- Graficar el error en la figura principal ---
    plt.plot(resultados.errors, label=f'$\lambda$={lam} (Ep:{ep})')
    
    # --- Guardar el historial de pesos solo para 3.1416 ---
    if lam == 3.1416:
        pesos_historial_31416 = np.array(resultados.weights).squeeze()

# --- Tabla Resumen ---
df_resumen = pd.DataFrame(resultados_resumen)
print("\n" + "="*80)
print("TABLA RESUMEN DE ENTRENAMIENTO")
print("="*80)
print(df_resumen.to_string(index=False))
# Guardar la tabla en un CSV real
df_resumen.to_csv('results/reporte_red_neuronal.csv', index=False)
print("\n[!] La tabla también ha sido guardada como '.csv'")

# --- Gráfica de evolución del error ---
plt.title('Evolución del Error para 5 Tasas de Aprendizaje')
plt.xlabel('Épocas')
plt.ylabel('Magnitud del Error (Norma)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("results/evolucion_error_lambdas.png")
plt.show()

# --- Gráfica de evolución de los pesos (Lambda 3.1416) ---
if pesos_historial_31416 is not None:
    # Aseguramos que tenga la dimensión correcta
    if pesos_historial_31416.ndim == 1:
        pesos_historial_31416 = pesos_historial_31416.reshape(-1, 1)
        
    plt.figure(figsize=(12, 6))
    num_pesos = pesos_historial_31416.shape[1]
    
    for i in range(num_pesos):
        plt.plot(pesos_historial_31416[:, i], label=f'Peso $W_{i}$', linewidth=2)

    plt.title('Evolución de los Pesos ($\lambda$ = 3.1416, Épocas = 360)')
    plt.xlabel('Épocas')
    plt.ylabel('Valor del Peso')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/evolucion_pesos.png")
    plt.show()
