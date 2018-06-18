import matplotlib.pyplot as plt
import numpy as np
import pickle

try:
    d = pickle.load(open("Data", "rb"))
except:
    print("No se pudo leer correctamente el archivo")

x = np.linspace(1,10,2000)
y1 = x**3
y2 = 2*x**3 - 50*np.floor(x)
y3 = x**3 + 30*np.floor(x)

plt.figure("Figura 1")#Generamos una figura ("ventana de ploteo")
plt.plot(x,y1,label="y1",color = "red")#Graficamos (dibujamos) y1
plt.plot(x,y2,label="y2",color = "green")#Graficamos (dibujamos) y1
plt.plot(x,y3,label="y3",color = "blue")#Graficamos (dibujamos) y1
plt.title("Evolución de la F.O.")#Le ponemos título al gráfico
plt.xlabel("Tiempo (s)",fontsize=16)#Le ponemos título al eje de las abscisas
plt.ylabel("MSE",fontsize=16)#Le ponemos título al eje de las ordenadas
plt.tight_layout()#Este comando permite, en muchos casos, ajustar la visualización del gráfico para que quepa dentro de la figura
plt.legend()#Anotamos etiquetas para las curvas en el gráfico
plt.show()#Mostramos (si no no se abre ni se ve nada) la figura con el gráfico