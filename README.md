# AIsolation
Proyecto de detección de la soledad prematuramente

 ## Autores:
- Santiago Bellés Castellet
- Cristina Pardo Arrufat
- Joan Padilla Orenga
- Daniel Gómez Garbí


## Objetivo
Procesar un texto como respuesta a unas preguntas para determinar si el inviduo se encuentra solo o no.


## Estructura del proyecto
### Input
Una pregunta en la que el individuo puede expresar lo que siente.


Tres preguntas de ámbito personal: edad, género y si vive solo o no.


### Interfaz
Cuestionario realizado en una aplicación con la herramienta stremlit.io


### Modelo de análisis de sentimientos
Con este modelo ya entrenado se analiza el input.
Modelo obtenido de Hugging Face, concretamente es el pysentimiento.


### Modelo obtención de soledad
El archivo aisolation.py es el modelo final donde se utiliza el algoritmo supervisado regresivo RandomForest.


### Dataset
Modelo final entrenado con datos_serena_mod.csv.


## Notebook Google Colab
Notebook: https://colab.research.google.com/drive/1s44drJEzbw6BTeabrKGIGlCEnVCZ5j1B


## Links externos
Articulo en Medium: ----url

https://huggingface.co/

https://streamlit.io/
