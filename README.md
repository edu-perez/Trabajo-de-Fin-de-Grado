# Trabajo-de-Fin-de-Grado
En este repositorio se recoge el código utilizado en el Trabajo de Fin de Grado (TFG) de Eduardo Pérez Álvarez, estudiante del Grado en Física en la Universidad de Granada.

El archivo multicapa.cpp contiene el perceptrón multicapa clásico con el que se han hecho las simulaciones. Es un programa C++ y ninguna librería específica de machine learning es usada. Los archivos aleatorios.h y dranxor2new.f son necesarios para el generador de números aleatorios utilizado, aunque si se tiene algún problema con ellos se puede modificar el código e incluir cualquier otro generador. Habría que sustituir las funciones dranini_(...) y dranu_().

El archivo StrawberryF-TFG-GH.ipynb contiene la continuous-variable quantum neural network de Strawberry Fields. Es un Jupyter Notebook con el siguiente soporte: Python 3.6.10, TensorFlow 1.3.0 , StrawberryFields 0.10.0 , Numpy 1.18.1 , and Matplotlib
3.1.3. NOTA: El soporte de TensorFlow con Strawberry Fields no está diseñado para versiones superiores a las mencionadas.

El archivo NNetwork-TFG.ipynb contiene el programa utilizado para detectar entrelazamiento cuántico a través del criterio de Peres-Horodecki. Es un Jupyter Notebook con el siguiente soporte: Python 3.7.6, Keras 2.3.1, TensorFlow 2.1.0, Numpy 1.18.1, Pandas
1.0.1 and Matplotlib 3.1.3.
