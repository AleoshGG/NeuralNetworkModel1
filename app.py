from NeuronalNetwork import NeuronalNetwork

neuronalNet = NeuronalNetwork('datasets/or.csv')
neuronalNet.show_data_init()

print("\n--- Iniciando Entrenamiento ---")

# 2. Entrenar
# Nota: Para la compuerta OR, a veces se requieren más épocas o un lambda ajustado
results = neuronalNet.train(epochs=100, lambda_value=0.1)

# 3. Presentar Resultados con los nuevos métodos
results.show_final_metrics()    # Texto en consola
results.plot_learning_curve()   # Gráfica visual