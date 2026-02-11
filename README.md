# Implementación de Red Neuronal Simple para la Compuerta OR

Este proyecto implementa una red neuronal perceptrón simple desde cero utilizando `numpy` para resolver el problema de la compuerta lógica OR. La red entrena sobre un conjunto de datos predefinido y visualiza su curva de aprendizaje utilizando `matplotlib`.

## Características

*   **Red Neuronal Perceptrón Simple**: Implementación básica de una red neuronal para clasificación binaria.
*   **Entrenamiento Supervisado**: La red aprende de un conjunto de datos etiquetado (`or.csv`).
*   **Visualización de Resultados**: Gráfica de la curva de aprendizaje para monitorear el progreso del entrenamiento.
*   **Métricas de Entrenamiento**: Muestra el error final y los pesos aprendidos.

## Estructura del Proyecto

```
.
├── app.py                      # Script principal para ejecutar el entrenamiento.
├── NeuronalNetwork.py          # Clase que implementa la red neuronal.
├── requirements.txt            # Dependencias del proyecto.
├── datasets/
│   └── or.csv                  # Dataset para la compuerta OR.
└── entities/
    └── data_results.py         # Clase para manejar y visualizar los resultados del entrenamiento.
```

## Instalación

Sigue estos pasos para configurar el entorno y ejecutar el proyecto:

1.  **Clonar el repositorio** (si aplica, o asegúrate de tener los archivos en tu máquina local).

2.  **Crear un entorno virtual** (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar las dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para entrenar la red neuronal y ver los resultados, ejecuta el script principal `app.py`:

```bash
python app.py
```

El script imprimirá las dimensiones iniciales del dataset y los pesos, los resultados finales del entrenamiento (error y pesos), y luego mostrará una gráfica de la curva de aprendizaje que representa la magnitud del error a lo largo de las épocas.

## Explicación del Código

### `app.py`

Este es el punto de entrada de la aplicación.
1.  Inicializa una instancia de `NeuronalNetwork` cargando el dataset `or.csv`.
2.  Muestra la información inicial del dataset y los pesos.
3.  Inicia el proceso de entrenamiento llamando al método `train()` de la red neuronal, especificando el número de épocas y el valor de lambda (tasa de aprendizaje).
4.  Una vez finalizado el entrenamiento, utiliza la instancia `DataResults` devuelta para mostrar las métricas finales en la consola y generar la gráfica de la curva de aprendizaje.

### `NeuronalNetwork.py`

Define la clase `NeuronalNetwork`, que encapsula la lógica de la red neuronal.

*   **`__init__(self, path_dataset: str, skiprows = 1)`**:
    *   Carga el dataset desde la ruta especificada.
    *   Prepara los valores de entrada `X` (añadiendo un sesgo/bias) y los valores objetivo `Y`.
    *   Inicializa los pesos `W` de forma aleatoria.
*   **`__slicing_Y_values(self)` / `__slicing_X_values(self)`**: Métodos internos para preprocesar el dataset, separando las características de las etiquetas y añadiendo la columna de bias a `X`.
*   **`__generate_W_values(self)`**: Genera los pesos iniciales de forma aleatoria.
*   **`show_data_init(self)`**: Imprime las formas (shapes) de los tensores iniciales del dataset, `X`, `Y` y `W`.
*   **`train(self, epochs: int, lambda_value: float) -> DataResults`**:
    *   Implementa el bucle de entrenamiento.
    *   En cada época, realiza una propagación hacia adelante (`__funtion_activation`), calcula el error, actualiza los pesos `W` utilizando la regla de aprendizaje del perceptrón (ajustada por `lambda_value`).
    *   Registra la magnitud del error y los pesos en cada época en un objeto `DataResults`.
    *   Devuelve un objeto `DataResults` con el historial de entrenamiento.
*   **`__funtion_activation(self, u)`**: Implementa la función de activación escalón (heaviside), que devuelve 1 si la entrada es mayor o igual a 0, y 0 en caso contrario.

### `entities/data_results.py`

Define la clase `DataResults`, utilizada para almacenar y presentar los resultados del entrenamiento.

*   **`__init__(self, errors: list, weights: list)`**: Almacena las listas de errores (magnitud del error en cada época) y los pesos correspondientes a cada época.
*   **`show_final_metrics(self)`**: Imprime en la consola el error final y los pesos finales de la red.
*   **`plot_learning_curve(self)`**: Genera y muestra una gráfica de la evolución del error a lo largo de las épocas utilizando `matplotlib.pyplot`.

---

¡Disfruta experimentando con esta implementación simple de red neuronal!
