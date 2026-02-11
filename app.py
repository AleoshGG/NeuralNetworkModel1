import numpy as np
import matplotlib.pyplot as plt
from NeuronalNetwork import NeuronalNetwork

lambdas = [1, 1e-6, 10, 0.1, 0.01]

results = []

for lambda_value in lambdas:
    neuronalNet = NeuronalNetwork('datasets/or.csv')
    print(f"\n--- Iniciando Entrenamiento Lambda = {lambda_value} ---")
    results.append(neuronalNet.train(epochs=500, lambda_value=lambda_value))


plt.figure(figsize=(12, 8))
plt.plot(results[0].errors, label='Lambda = 1', color='blue')
plt.plot(results[1].errors, label='Lambda = 1e-6', color='green')
plt.plot(results[2].errors, label='Lambda = 10', color='red')
plt.plot(results[3].errors, label='Lambda = 0.1', color='orange')
plt.plot(results[4].errors, label='Lambda = 0.01', color='purple')
plt.xlabel('Épocas')
plt.ylabel('Error')
plt.title('Evolución del Error por Lambda')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show() 