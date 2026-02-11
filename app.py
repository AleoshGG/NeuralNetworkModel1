from NeuronalNetwork import NeuronalNetwork

neuronalNet = NeuronalNetwork('datasets/or.csv')
neuronalNet.show_data_init()

print("\n--- Iniciando Entrenamiento ---")
results = neuronalNet.train(epochs=10, lambda_value=1)

print(f"Errores: \n{results.errors}")
print(f"\nPesos: \n{results.weights}")