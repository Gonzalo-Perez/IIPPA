import numpy as np
import cv2
from iter_methods import *
from measures import *

N = 20

"""
To do's:
(M): Martín es encargado
(A): Alonso es encargado
(I): Ignacio es encargado

### URGENTES ###
- Gradiente estocástico más "serio". ie. leer algo de literatura e implementar algo con soporte teórico.

- Método acelerado.
    - no implementado aun. En principio no requiere nada especial.

- Greedy coordinate descent.
    - En principio no requiere nada en especial, quizás un linesearch más sofisticado


### No urgentes ###
- Construir gráficos bonitos para los experimentos.

- Estudiar las condiciones de término de los algoritmos.

- Cambio de coordenadas, utilizar coordinadas polares.
    -Ya hay implementado un método de transformación en la dirección Polares -> cartesianas.
    Con esto es fácil probar sin modificar mucho el código existente. 

- Terminar de hacer andar el "warm start" para calcular derivadas direccionales.
    - Para usar MSE hay que pasar de tomar como argumento dos imágenes, a dos imágenes más las 
    dimensiones generales. Así se evita que al hacer ventanas se normalize por dimensiones erroneas.

- Método de newton?
    - Hay formulas de diferencias finitas para calcular derivadas de orden superior. Con eso se podría calcular
    una matriz gessiana numérica. Hay que darle una vuelta ya que esta wea es tremendamente no suave.

- Implementar con círculos

- Implementar algoritmo genético simple para comparar

### Ideas ###
- Agregar una variable offset para los triangulos, ver si ella mejora el método.

- Sampling para reducir costo de computar diferencias. Está la opción de usar fotos chicas, explorar otras.

- El gradiente simple estocástico se queda pegado al final en puntos durante varias iteraciones.
    ¿Cómo evitar/mejorar este comportamiento?
    
- Difuminar con un filtro gaussiano o un kernel de calor (el primero es difuminación lineal, el segundo no lineal)
las imágenes para darles uan "pendiente" de asenso al algoritmo.
    - En términos prácticos es fácil de implementar, especialmente los filtros gaussianos.

- Hacer comparación de parámetros para el gradiente estocástico. hacer variar ratio_vars_to_compute

### Otras optimizaciones y pendientes###
- El método para calcular centroides mediante variables necesita debugging

- Para el "warm start" se debe utilizar un método para encontrar la ventana relevante. El método implementado
tiene bugs, se utilizó una comparación "a la bruta" en el intertanto.

"""

if __name__ == "__main__":
    img_objective = cv2.imread("Test_cases/blue_circle.png")
    img_objective = (img_objective / 255)
    np.random.seed(1235)
    cv2.namedWindow("Objetivo Simple")
    cv2.moveWindow("Objetivo Simple", 500, 250)
    cv2.imshow("Objetivo Simple", img_objective)
    cv2.waitKey(0)

    # x = simple_gradient_method_parallel(img_objective, N, 0, lambda n: .3 / np.log(np.e + n), 300, 1e-4, _delta=.05,
    #                                     diff_scheme_to_use=2, show_progress=True)

    y = sim_grad_stochastic_parallel(img_objective, N, norm_mode=0, initial_x='',
                                     stoc_ratio=.5, v_max=10, linesearch_num_steps=40, linesearch_step_size=.125,
                                     max_iter=300, tol=1e-4, null_triag_correction=True, steps_tolerance=4,
                                     triag_area_tol=0.003, _delta=.05, diff_scheme_to_use=2, show_progress=True,
                                     choose_triags=True)
    # np.save('solution_stochastic_grad', y)
    quit()
