import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import copy,deepcopy
from random import randint, choice, uniform
from skimage.measure import structural_similarity as ssim
from threading import Thread
import time
import pickle

start_time = time.time()

class Circulo:

    def __init__(self,x,y,r = 6, color = (50,50,50)):
        self.pos_x = x
        self.pos_y = y
        self.radio = r
        self.color = color
        self.id = None
    def get_positions(self):
        return (self.pos_x,self.pos_y)
    def get_color(self):
        return self.color
    def get_radio(self):
        return self.radio
    def __str__(self):
        return str(self.get_positions())+", " + str(self.get_radio()) +", " + str(self.get_color())
    # def __repr__(self):
    #     return str(self.get_positions())+", " + str(self.get_radio()) +", " + str(self.get_color())
    def custom_str(self):
        return str(self.get_positions())+", " + str(self.get_radio()) +", " + str(self.get_color())

    def draw(self,image):
        cv2.circle(image, self.get_positions(), self.get_radio(), self.get_color(), -1)


class Triangulo:
    def __init__(self,pos_v1=None,pos_v2=None,pos_v3=None, color=(50, 50, 50), center_pos=None,angle = 0, envergadura = 30):
        if type(pos_v1)!=type(None): #Debe haber una forma más elegante de hacerlo
            self.pos_v1 = pos_v1
            self.pos_v2 = pos_v2
            self.pos_v3 = pos_v3
        else:
            relative_not_rotated_pos_v1 = np.array([randint(-envergadura, envergadura), randint(-envergadura, envergadura)])
            relative_not_rotated_pos_v2 = np.array([randint(-envergadura, envergadura), randint(-envergadura, envergadura)])
            relative_not_rotated_pos_v3 = np.array([randint(-envergadura, envergadura), randint(-envergadura, envergadura)])
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated_rel_pos_v1 = R.dot(relative_not_rotated_pos_v1)
            rotated_rel_pos_v2 = R.dot(relative_not_rotated_pos_v2)
            rotated_rel_pos_v3 = R.dot(relative_not_rotated_pos_v3)
            self.pos_v1 = rotated_rel_pos_v1 + center_pos
            self.pos_v2 = rotated_rel_pos_v2 + center_pos
            self.pos_v3 = rotated_rel_pos_v3 + center_pos

        self.color = color
        if type(center_pos) != type(None):
            self.center_pos = center_pos
        else:
            self.center_pos = np.sum(np.vstack((self.pos_v1, self.pos_v2, self.pos_v3)), axis=0) / 3
        self.angle = angle
        self.id = None

    def get_center(self):
        if self.center_pos!= None:
            return self.center_pos
        else:
            return np.sum(np.vstack((self.pos_v1,self.pos_v2,self.pos_v3)),axis=0)/3
    def get_color(self):
        return self.color

    def get_angle(self):
        return self.angle

    def move(self):
        old_center = self.center_pos
        self.center_pos = np.array([randint(0,700),randint(0,700)])
        move_vector = self.center_pos-old_center
        self.pos_v1 = self.pos_v1 + move_vector
        self.pos_v2 = self.pos_v2 + move_vector
        self.pos_v3 = self.pos_v3 + move_vector

    def random_rotate(self):
        angle = uniform(0,2*np.pi)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        v1_rel_vect = self.pos_v1 - self.center_pos
        v2_rel_vect = self.pos_v2 - self.center_pos
        v3_rel_vect = self.pos_v3 - self.center_pos
        v1_rel_vect = R.dot(v1_rel_vect)
        v2_rel_vect = R.dot(v2_rel_vect)
        v3_rel_vect = R.dot(v3_rel_vect)
        self.pos_v1 = self.center_pos + v1_rel_vect
        self.pos_v2 = self.center_pos + v2_rel_vect
        self.pos_v3 = self.center_pos + v3_rel_vect

    def __str__(self):
        return str(self.get_center())

    def draw(self,image):
        polygon_points = np.array([self.pos_v1, self.pos_v2, self.pos_v3], np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        cv2.fillPoly(image, np.array([polygon_points], np.int32), self.color)



class Poblacion:
    def __init__(self):
        self.circulos = []
        self.triangulos = []

    def agregar_criatura(self,shape):
        if type(shape) == Circulo:
            shape.id = len(self.circulos)+len(self.triangulos)
            self.circulos.append(shape)
        elif type(shape) == Triangulo:
            shape.id = len(self.circulos) + len(self.triangulos)
            self.triangulos.append(shape)

    def __str__(self):
        return str([circulo.custom_str() for circulo in self.circulos])

    def draw_acumulative(self,imagen):
        for circulo in self.circulos:
            circulo.draw(imagen)
        for triangulo in self.triangulos:
            triangulo.draw(imagen)

    def draw(self,imagen): #Borra la imagen antes de dibujar de nuevo
        # imagen = np.full((imagen.shape[1],imagen.shape[0],3), 0, np.uint8)
        # cv2.circle(imagen, (200,200), 2000, (0,0,220), -1)
        # imagen = np.full((700, 700, 3), 50, np.uint8)
        for circulo in self.circulos:
            circulo.draw(imagen)
        for triangulo in self.triangulos:
            triangulo.draw(imagen)

    def erase(self,imagen):
        imgzero = np.full((700, 700, 3), 255, np.uint8)
        # print(imagen)
        imagen = deepcopy(imgzero)
        # print(imagen)

    def mutate_circles_positions(self):
        for circulo in self.circulos:
            circulo.pos_x = circulo.pos_x + randint(-30,30)
            circulo.pos_y = circulo.pos_y + randint(-30,30)
            # circulo.pos_x = randint(circulo.radio, len(img[0]) - circulo.radio)
            # circulo.pos_y = randint(circulo.radio, len(img) - circulo.radio)

    def mutate_circles_positions_one(self):
        if len(self.circulos)>=1:
            circulo = choice(self.circulos)
            # circulo.pos_x = circulo.pos_x + randint(-30,30)
            # circulo.pos_y = circulo.pos_y + randint(-30,30)
            circulo.pos_x = randint(circulo.radio, len(img[0]) - circulo.radio)
            circulo.pos_y = randint(circulo.radio, len(img) - circulo.radio)
        else:
            print("No hay círculos para elegir")

    def mutate_circles_colors(self):
        for circulo in self.circulos:
            # circulo.color = (circulo.color[2],circulo.color[0],circulo.color[1]) #Permutar los canales (revisar "canales" es el término?)
            circulo.color = (randint(0,255),randint(0,255),randint(0,255))

    # def mutate_circles_colors_one(self):
    #     if len(self.circulos)>=1:
    #         circulo = choice(self.circulos)
    #         # circulo.color = (circulo.color[2], circulo.color[0], circulo.color[1]) #Permutar los canales (revisar "canales" es el término?)
    #         circulo.color = (randint(0, 255), randint(0, 255), randint(0, 255))
    #     else:
    #         print("No hay círculos para elegir")

    def mutate_circles_colors_one(self):
        if len(self.circulos)>=1:
            circulo = choice(self.circulos)
            # circulo.color = (circulo.color[2], circulo.color[0], circulo.color[1]) #Permutar los canales (revisar "canales" es el término?)
            circulo.color = choice([(randint(0, 255), circulo.color[1], circulo.color[2]),(circulo.color[0], randint(0, 255), circulo.color[2]),(circulo.color[0], circulo.color[1], randint(0, 255))])
        else:
            print("No hay círculos para elegir")

    def mutate_triangles_colors(self):
        for triangulo in self.triangulos:
            triangulo.color = (randint(0,255),randint(0,255),randint(0,255))

    def mutate_triangles_colors_one(self):
        if len(self.triangulos)>=1:
            triangulo = choice(self.triangulos)
            triangulo.color = choice([(randint(0, 255), triangulo.color[1], triangulo.color[2]),(triangulo.color[0], randint(0, 255), triangulo.color[2]),(triangulo.color[0], triangulo.color[1], randint(0, 255))])
        else:
            print("No hay triangulos para elegir")

    def mutate_triangles_angle_one(self):
        if len(self.triangulos)>=1:
            triangulo = choice(self.triangulos)
            triangulo.random_rotate()
        else:
            print("No hay triangulos para elegir")

    def mutate_triangles_angles(self):
        for triangulo in self.triangulos:
            triangulo.random_rotate()

    def mutate_triangles_positions_one(self):
        if len(self.triangulos)>=1:
            triangulo = choice(self.triangulos)
            triangulo.move()
        else:
            print("No hay triangulos para elegir")

    def mutate_triangles_positions(self):
        for triangulo in self.triangulos:
            triangulo.move()

    def remove_one_circle(self):
        if len(self.circulos)>=1:
            i = randint(0,len(self.circulos)-1)
            del self.circulos[i]
        else:
            print("No hay círculos para elegir")

    def add_one_circle(self):
        radio  = 6
        circulo = Circulo(x = randint(radio,len(img[0])-radio), y = randint(radio,len(img)-radio), r = radio, color = (randint(0,255),randint(0,255),randint(0,255)))
        self.agregar_criatura(circulo)

    def change_radius(self):
        if len(self.circulos)>=1:
            i = randint(0,len(self.circulos)-1)
            self.circulos[i].radio = randint(4,100)
        else:
            print("No hay círculos para elegir")




# c1 = Circulo(200,100,40,(150,0,0))
# c2 = Circulo(100,400,100,(0,0,200))
# c3 = Circulo(100,200,60,(0,150,0))
# p1 = Poblacion()
# p1.agregar_criatura(c1)
# p1.agregar_criatura(c2)
# p1.agregar_criatura(c3)
# img11 = np.full((700,700,3), 255, np.uint8)
# p1.draw(img11)
# cv2.imshow("Imagen GA",img11)
# cv2.moveWindow("Imagen GA", 50, 50)
# cv2.waitKey(0)



def mabse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2) #Al cuadrado
    # err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))  # Con abs
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


#Creamos la imagen
# img = np.full((700,700,3), 255, np.uint8)
# cv2.circle(img, (250, 200), 170, (0, 255, 0), -1)
# cv2.circle(img, (250, 200), 60, (0, 80, 0), -1)
#
#
#
# n = 20
# radio = 25
# pos_x_parches = [randint(50,len(img[0])-radio) for i in range(n)]
# pos_y_parches = [randint(50,len(img)-radio) for i in range(n)]

# circuloss = [Circulo(x = randint(50,len(img[0])-radio), y = randint(50,len(img)-radio),r = radio,color = (randint(20,80),randint(20,80),randint(20,80))) for i in range(n)]
#
# pob = Poblacion()
# for c in circuloss:
#     pob.agregar_criatura(c)
# # print(pob)
#
#
#
# img12 = np.full((700,700,3), 255, np.uint8)
# pob.draw(img12)
# cv2.imshow("Imagen GA222",img12)
# cv2.moveWindow("Imagen GA222", 50, 50)
# cv2.waitKey(0)




#
# cv2.namedWindow("Evolving while")
# img = img12
# imgzero = np.full((700, 700, 3), 0, np.uint8)
# img = deepcopy(imgzero)
# while True:
#     cv2.imshow("Evolving while", img)
#
#     k = cv2.waitKey()
#     if k == ord("a"):                      # toggle current image
#         img = deepcopy(imgzero)#Averiguar como poner esto dentro de un método
#         # pob.erase(img) #Dentro de este o de otro
#         pob.mutate_positions()
#
#         pob.draw(img)
#     elif k == ord("w"):
#         img = deepcopy(imgzero)
#         # pob.erase(img)
#         pob.mutate_colors()
#         pob.draw(img)
#         # img = np.full((700, 700, 3), 50, np.uint8)
#     elif k == ord("e"):
#         break
#
#
# cv2.destroyAllWindows()


######################
img = np.full((370,270,3), 255, np.uint8)
n_c = 0
n_t = 1000
radio = 10
envergadura = 20
# random_circulos = [Circulo(x = randint(radio,len(img[0])-radio), y = randint(radio,len(img)-radio),r = radio,color = (randint(0,255),randint(0,255),randint(0,255))) for i in range(n_c)]
# random_triangulos = [Triangulo(np.array([randint(375,425),randint(375,425)]),np.array([randint(375,425),randint(375,425)]),np.array([randint(375,425),randint(375,425)]),color=(randint(0,255),randint(0,255),randint(0,255))) for i in range(n_t)]
random_triangulos = [Triangulo(center_pos=(np.array([randint(0,len(img[0])),randint(0,len(img))])),envergadura=envergadura,color=(randint(0,255),randint(0,255),randint(0,255))) for i in range(n_t)]

#Para fijarlos todos en negro al inicio
# random_circulos = [Circulo(x = randint(50,len(img[0])-radio), y = randint(50,len(img)-radio),r = radio,color = (0,0,0)) for i in range(n)]
pob = Poblacion()
# for c in random_circulos:
#     pob.agregar_criatura(c)
for t in random_triangulos:
    pob.agregar_criatura(t)
# print(pob)
#################
# img_objective = np.full((700,700,3), 255, np.uint8)
# cv2.circle(img_objective, (400, 100), 80, (0, 0, 0), -1)
# cv2.circle(img_objective, (100, 400), 140, (0, 0, 0), -1)
# cv2.circle(img_objective, (400, 400), 100, (0, 0, 0), -1)




#Creamos la imagen objetivo
img_objective = np.full((370,270,3), 255, np.uint8)
cv2.circle(img_objective, (150, 550), 100, (0, 0, 255), -1)
cv2.circle(img_objective, (550, 150), 100, (0, 255, 0), -1)
cv2.circle(img_objective, (150, 150), 100, (255, 0, 0), -1)
cv2.circle(img_objective, (550, 550), 100, (0, 0, 0), -1)

# img_objective = cv2.imread("Dificil evolucion.png")#La de la evolución (dificil)
img_objective = cv2.imread("Arkadi.png")#La de la evolución (dificil)
# img_objective = cv2.imread('Imagen objetivo.png')#La de los cuatro círculos



###############
# cv2.namedWindow("Objective")
# cv2.imshow("Objective", img_objective)
# cv2.waitKey(0)
########
# Cambio de BGR a RBG
img_aux = np.flip(img_objective,2)
#########
plt.imsave('Imagen objetivo v.4.png',img_aux)


ciclos_for_plot = []
fitness_over_iterations = []
# plt.ion()
# fig = plt.figure()#"Fitness evolution"
# # plt.title("Evolución de la F.F.")
# ax = fig.add_subplot(111)
# fitness_curve, = ax.plot(ciclos_for_plot, fitness_over_iterations)
# # p = Thread(target=plt.show(block=False))
# # p.start()
# # p.join()

plt.figure("Evolución de la F.F.")
#Para localizar de forma más ordenada la figura en la pantalla
mngr = plt.get_current_fig_manager()
#Para ponerla en la parte superior izquierda de la pantalla y cambiar sus dimensiones
mngr.window.setGeometry(1440,60,460, 460)
plt.ion()
plt.plot(ciclos_for_plot,fitness_over_iterations)
plt.title("Evolución de la fitness function")
plt.xlabel("Ciclos")
plt.ylabel("Valor de la F.F.")
plt.show()
plt.tight_layout()




ciclos = [i+1 for i in range(100000)]
imgzero = np.full((370, 270, 3), 255, np.uint8)
img = deepcopy(imgzero)
cv2.namedWindow("Evolution v.4")
cv2.moveWindow("Evolution v.4", 1020, 300)
cv2.namedWindow("Best v.4")
cv2.moveWindow("Best v.4", 660, 300)
cv2.namedWindow("Objetivo")
cv2.moveWindow("Objetivo", 310, 300)

pob.mutate_circles_positions()
pob.draw(img)
# Cambio de BGR a RBG
img_aux = np.flip(img,2)
#########
plt.imsave('Imagen original v.4.png',img_aux)
cv2.imshow("Evolution v.4", img)
cv2.imshow("Best v.4", img)
cv2.imshow("Objetivo", img_objective)
print("Press any key to start the GA")
cv2.waitKey(0)
mutation_options = [0,1,2,3,4,5,6,7,8,9,10,11] #Podría tener otras razones de mutación e incluso razones variables!
#Puedo calcular (quizás con una media móvil) el porcentaje de mejora (o grado/derivada de mejora) para
#la mutación de posición y para la mutación de color y en base a eso fijar los rates (las razones) de mutación.
#Hasta ahora solo hay mutación. No hay reproducción por el momento
best_pob = deepcopy(pob)
best_fitness = None #En realidad es mejor inicializarlo con una cota superior para evitar la comparación con el None en el If
#ATENCIÓN: Al dejarlo así, si se pone n = 0, de todas formas marca un improvement al principio y acepta agregar un círculo.
# best_fitness = mabse(img,img_objective)*100000 #Con una cota superior (no demostrada matemáticamente)
p_pos_mut_c = 0.5
p_color_mut_c = 0.0
p_both_mut_c = 0.0
p_remove_mut_c = 0.0
p_add_mut_c = 0.0
p_change_radius_mut = 0.0
p_pos_mut_t = 0.5
p_color_mut_t = 0.0
p_angle_mut_t = 0.0
p_remove_mut_t = 0.0
p_add_mut_t = 0.0
p_dummy_do_nothing = 0.0
for c in ciclos:
    pob = deepcopy(best_pob)
    img = deepcopy(imgzero)#Esto 'borra' la imagen anterior*REVISAR!
    # option = choice(mutation_options)
    if c % 100 == 0:
        print("Número de círculos: {}".format(len(best_pob.circulos)))
        print("Número de triangulos: {}".format(len(best_pob.triangulos)))
##########PARA VARIAR LAS TASAS DE MUTACIÓN DURANTE LA EJECUCIÓN
    if c >= 5000 and c <10000:
        p_pos_mut_c = 0.5
        p_color_mut_c = 0.0
        p_both_mut_c = 0.0
        p_remove_mut_c = 0.0
        p_add_mut_c = 0.0
        p_change_radius_mut = 0.0
        p_pos_mut_t = 0.5
        p_color_mut_t = 0.0
        p_angle_mut_t = 0.0
        p_remove_mut_t = 0.0
        p_add_mut_t = 0.0
        p_dummy_do_nothing = 0.0

    if c >= 10000 and c < 22000:
        p_pos_mut_c = 0.4
        p_color_mut_c = 0.0
        p_both_mut_c = 0.2
        p_remove_mut_c = 0.0
        p_add_mut_c = 0.0
        p_change_radius_mut = 0.0
        p_pos_mut_t = 0.4
        p_color_mut_t = 0.0
        p_angle_mut_t = 0.0
        p_remove_mut_t = 0.0
        p_add_mut_t = 0.0
        p_dummy_do_nothing = 0.0
    elif c >= 22000 and c < 25000:
        p_pos_mut_c = 0.0
        p_color_mut_c = 0.2
        p_both_mut_c = 0.0
        p_remove_mut_c = 0.2
        p_add_mut_c = 0.1
        p_change_radius_mut = 0.1
        p_pos_mut_t = 0.0
        p_color_mut_t = 0.2
        p_angle_mut_t = 0.2
        p_remove_mut_t = 0.0
        p_add_mut_t = 0.0
        p_dummy_do_nothing = 0.0

    elif c >= 25000 and c < 50000:
        p_pos_mut_c = 0.1
        p_color_mut_c = 0.2
        p_both_mut_c = 0.0
        p_remove_mut_c = 0.0
        p_add_mut_c = 0.0
        p_change_radius_mut = 0.2
        p_pos_mut_t = 0.1
        p_color_mut_t = 0.1
        p_angle_mut_t = 0.3
        p_remove_mut_t = 0.0
        p_add_mut_t = 0.0
        p_dummy_do_nothing = 0.0
    elif c >= 50000 and c < 75000:
        p_pos_mut_c = 0.0
        p_color_mut_c = 0.2
        p_both_mut_c = 0.0
        p_remove_mut_c = 0.2
        p_add_mut_c = 0.0
        p_change_radius_mut = 0.2
        p_pos_mut_t = 0.0
        p_color_mut_t = 0.2
        p_angle_mut_t = 0.2
        p_remove_mut_t = 0.0
        p_add_mut_t = 0.0
        p_dummy_do_nothing = 0.0
    elif c >= 75000 and c < 90000:
        p_pos_mut_c = 0.0
        p_color_mut_c = 0.4
        p_both_mut_c = 0.0
        p_remove_mut_c = 0.0
        p_add_mut_c = 0.0
        p_change_radius_mut = 0.0
        p_pos_mut_t = 0.0
        p_color_mut_t = 0.4
        p_angle_mut_t = 0.2
        p_remove_mut_t = 0.0
        p_add_mut_t = 0.0
        p_dummy_do_nothing = 0.0
##############
    option = np.random.choice(mutation_options,p=[p_pos_mut_c,p_color_mut_c,p_both_mut_c, p_remove_mut_c, p_add_mut_c,p_change_radius_mut,p_pos_mut_t,p_color_mut_t,p_angle_mut_t,p_add_mut_t,p_remove_mut_t,p_dummy_do_nothing])
    if option == 0:
        pob.mutate_circles_positions_one()
    elif option == 1:
        pob.mutate_circles_colors_one()
    elif option == 2:
        pob.mutate_circles_positions_one()
        pob.mutate_circles_colors_one()
        pob.mutate_triangles_positions_one()
        pob.mutate_triangles_colors_one()
    elif option == 3:
        pob.remove_one_circle()
    elif option == 4:
        pob.add_one_circle()
    elif option == 5:
        pob.change_radius()
    elif option == 6:
        pob.mutate_triangles_positions_one()
    elif option == 7:
        pob.mutate_triangles_colors_one()
    elif option == 8:
        pob.mutate_triangles_angle_one()
    elif option == 9:
        pass
    elif option == 10:
        pass
    elif option == 11:
        print("Doing nothing ...")

    pob.draw(img)
    new_diff = mabse(img,img_objective)
    if(best_fitness == None or new_diff<best_fitness): #No sé si esto se considera como mala práctica ... Revisar!
        best_pob = pob
        best_fitness = new_diff
        print("Improvement! at cicle {0} with fitnes:= {1}".format(c,best_fitness))
        cv2.imshow("Best v.4", img)
        cv2.waitKey(1)
    cv2.imshow("Evolution v.4", img)
    # cv2.waitKey(300)#300 ms
    # cv2.waitKey(0)#Hasta que se aprete una tecla
    cv2.waitKey(1)  # 1 ms


    ciclos_for_plot.append(c)
    fitness_over_iterations.append(best_fitness)

    plt.plot(ciclos_for_plot,fitness_over_iterations,color = "red")
    plt.draw()
    plt.tight_layout()
    plt.pause(0.001)
    if c == len(ciclos)-1:
        plt.savefig("probando_guardada_fitness_progress.png")
    # plt.figure("Fitness evolution")
    # plt.draw()
    # fitness_curve.set_xdata(ciclos_for_plot)
    # fitness_curve.set_ydata(fitness_over_iterations)
    # fig.canvas.draw()
    # fig.show()

img = deepcopy(imgzero)
best_pob.draw(img)
# Cambio de BGR a RBG
img = np.flip(img, 2)
#########
# Guardamos la imagen
plt.imsave('Mejor criatura GA v.4.png', img)
with open("Mejor imagen v.4","wb") as output_file:  # Atención con que pase a importar el otro módulo antes de guardar y cerrar (revisar!)
    pickle.dump(img, output_file)

with open("Mejor población v.4", "wb") as output_file: #Atención con que pase a importar el otro módulo antes de guardar y cerrar (revisar!)
    pickle.dump(best_pob, output_file)

tiempo_total = time.time() - start_time
with open("tiempo_GA v.4", "wb") as output_file: #Atención con que pase a importar el otro módulo antes de guardar y cerrar (revisar!)
    pickle.dump(tiempo_total, output_file)
print("Tiempo de ejecución: {} segundos".format(tiempo_total))

#
# cv2.imshow("Imagen original",img)
# cv2.moveWindow("Imagen original", 50, 50)
# cv2.waitKey(0)
#
#
# ciclos = [i+1 for i in range(10000)]
# img1 = img
# img2 = np.full((len(img1),len(img1[0]),3), 255, np.uint8)
# cv2.circle(img2, (250+60+55, 200), radio, (0, 0, 0), -1)
#
# previous_diff = mabse(img1,img2)*len(pos_x_parches)
# best_pos_x = deepcopy(pos_x_parches)
# best_pos_y = deepcopy(pos_y_parches)
# best_fitness = previous_diff
# for c in ciclos:
#     r = randint(0,len(pos_x_parches)-1)
#     pos_x_parches = deepcopy(best_pos_x)
#     pos_y_parches = deepcopy(best_pos_y)
#     pos_x_parches[r] = randint(50,len(img1[0])-radio)
#     pos_y_parches[r] = randint(50,len(img1)-radio)
#     img2 = np.full((len(img1),len(img1[0]),3), 255, np.uint8) #Agregaro o quitar este argumento quizás , np.uint8
#     for i in range(len(pos_x_parches)):
#         cv2.circle(img2, (pos_x_parches[i], pos_y_parches[i]), radio, (0, 0, 0), -1)
#     new_diff = mabse(img1, img2)
#     if new_diff<best_fitness:
#         best_pos_x = deepcopy(pos_x_parches)
#         best_pos_y = deepcopy(pos_y_parches)
#         best_fitness = new_diff
#         print("Improvement!")
#     # Para visualizar la evolución!
#     img_aux = deepcopy(img1)
#     for i in range(len(pos_x_parches)):
#         cv2.circle(img_aux, (best_pos_x[i], best_pos_y[i]), radio, (0, 0, 255), -1)
#         cv2.circle(img_aux, (pos_x_parches[i], pos_y_parches[i]), radio, (255, 0, 0), -1)
#     cv2.imshow("Criatura", img_aux)
#     cv2.moveWindow("Criatura", 600, 50)#Como esto se fija cada vez, no permite mover la ventana
#     cv2.waitKey(1)
#
#     print("Fitness ciclo {0}: {1}".format(c,best_fitness))
#
# img_aux = np.full((len(img1),len(img1[0]),3), 255, np.uint8)
# for i in range(len(pos_x_parches)):
#     cv2.circle(img_aux, (best_pos_x[i], best_pos_y[i]), radio, (0, 0, 0), -1)
# cv2.imshow("Criatura ganadora", img_aux)
# cv2.moveWindow("Criatura ganadora", 600, 50)
# cv2.waitKey(0)
#
# ########
# # Cambio de BGR a RBG
# img_aux = np.flip(img_aux,2)
# #########
# plt.imsave('Criatura ganadora 2.png',img_aux)
#
# ########
# # Cambio de BGR a RBG
# img1 = np.flip(img1,2)
# #########
# #Guardamos la imagen
# plt.imsave('Objetivo 2.png',img1)