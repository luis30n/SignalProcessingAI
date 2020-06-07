# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:40:18 2019

@author: Luis Álvarez López
@title: Red neuronal Compleja desde cero con CLMS
"""
# GENERACION DE DATASET DE ENTRADA

import numpy as np
import scipy.io as sio
import math   
from scipy.spatial import distance
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")

#Obtencion de dataset y division en train y test

dataset = sio.loadmat('newsignal.mat')
X = dataset["xsw"]
X = X-np.mean (X)
Y = dataset["ymed"]
Y = Y -np.mean (Y)
mem=0
X_train =X [0:1000]
X_test = X 
y_test= Y
for i in range( 1, mem+1):
   X_train = np.append(X_train, X[i:1000+i], 1)
   X_test =  np.append(X_test, X[0+i:30000+i], 1)

y_train = Y [mem:1000+mem]
# y_test = Y [0+mem:30000+mem]
#DEFINICION DE UNA CAPA
p=mem+1
topology = [p , 1 ,  1]


class Layer ():
    def __init__(self, conn, n_neur, act_f, der_f, der_f_vconj):
        self.neur=[]
        for i in range (0, n_neur):
            self.neur.append( Neuron (act_f(2*i+1), der_f(2*i+1), der_f_vconj(2*i+1)))
   #     self.b = (np.random.rand(1,neur) +  np.random.rand(1,neur)*1j) * 2 -1 -1j
        self.W = (np.random.rand(conn , n_neur) + np.random.rand(conn , n_neur)*1j)  * 2 - 1 - 1j
class Neuron ():
    def __init__(self,act_f, der_f, der_f_vconj):
        self.act_f= act_f
        self.der_f=der_f
        self.der_f_vconj=der_f_vconj
        
        
#DEFINICION DE LAS FUNCIONES DE ACTIVACION Y SUS DERIVADAS RESPECTO A W

# clms en funcion del orden de la neurona


def clms_function (n):
    clms= lambda v: v * (np.absolute(v)**(n-1))
    return clms

def der_clms_v (n):
    der_clms_v = lambda v: ((n+1)/2) * (np.absolute(v)**(n-1))
    return der_clms_v

def der_clms_vconj (n):
    der_clms_vconj = lambda v : ((n-1)/2) * ((np.absolute(v)**(n-3))*v**2)
    return der_clms_vconj

def nmse (y, yest):
    calc= 20*math.log10(np.linalg.norm(y-yest)/np.linalg.norm(y))
    return calc
    
    
    
# Creacion de topología

def create_ann (topology, act_f, der_f, der_f_vconj):
    ann = []
    
    for n_layer in range(1, len(topology)):
        ann.append(Layer(topology[n_layer-1],topology[n_layer], act_f, der_f, der_f_vconj))
        
    return ann

#FUNCION DE COSTE por ahora MSE
    
error = lambda y, yest : (y-yest) 
f_cost = lambda y, yest :(1/2)*np.mean((np.absolute(error(y,yest)))**2)
der_fcost = lambda y, yest: np.absolute(y,yest)
neural_net = create_ann(topology, clms_function, der_clms_v, der_clms_vconj)  

#ENTRENAMIENTO 

def train(neural_net, X, Y , der_cost_mse, lr=0.5, train = True):
    out = [(None,X)]
    #forward
    
    for n_layer in range(0, len(neural_net)):
        v =  out[-1][1] @ neural_net[n_layer].W  #+  neural_net[n_layer].b
        y =  np.zeros([ len (X[:]),len(neural_net[n_layer].neur)]) + np.zeros([ len (X[:]),len(neural_net[n_layer].neur)])*1j
        for n_neur,neur in enumerate(neural_net[n_layer].neur):
            y[:,n_neur] = neur.act_f(v[:,n_neur])
        out.append((v, y))
    
    #backward
    if train: 
      
        deltas = []
        
        for n_layer in reversed(range(0, len(neural_net))):
            v= out[n_layer+1][0]
            y = out[n_layer+1][1]
            y[:,n_neur] = neur.act_f(v[:,n_neur])

            der_f_vconj=  np.zeros([ len (X[:]),len(neural_net[n_layer].neur)]) + np.zeros([ len (X[:]),len(neural_net[n_layer].neur)])*1j
            der_f_v =  np.zeros([ len (X[:]),len(neural_net[n_layer].neur)]) + np.zeros([ len (X[:]),len(neural_net[n_layer].neur)])*1j
            for n_neur , neur in enumerate(neural_net[n_layer].neur):
                der_f_vconj[:,n_neur] = neur.der_f(v[:, n_neur])
                der_f_v[:, n_neur] = neur.der_f(np.conj(v[:, n_neur]))
            if n_layer == len (neural_net) -1:
                delta=  -0.5*(  np.conj(out[n_layer][1]).T @ (np.conj(error(Y,y))* der_f_vconj)+  np.conj(out[n_layer][1]).T @ (error(Y,y)*der_f_vconj))                
                deltas.insert (0, delta)
              
            else:
                deltas.insert (0, out[n_layer][1].T @ ( der_f_v @ ((deltas[0] @ _W.T ))))
            _W = neural_net[n_layer].W
                # Gradient descent
                # Ecuacion de actualizacion
     #      neural_net[n_layer].b = neural_net[n_layer].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[n_layer].W = neural_net[n_layer].W -  deltas[0] * lr        # Incremento de W es lr*delta*salida de esa capa 
           # print( neural_net[n_layer].W)
            #print (der_f_vconj[0])
    return out[-1][1]
      


# VISUALIZACIÓN Y TEST y entrenamiento

import time
from IPython.display import clear_output

neural_n = create_ann(topology, clms_function, der_clms_v, der_clms_vconj)

loss = []

for i in range(0, 6000):
    
  # Entrenemos a la red!
  pY = train(neural_n, X_train, y_train, der_fcost , lr=0.001)
  
  if i % 100 == 0:
   # print(pY)
    print("Error conseguido durante el entrenamiento = ")
    last_loss=f_cost(y_train, pY)
    loss.append(last_loss)
    print (last_loss)
    clear_output(wait=True)
    plt.clf()
    plt.plot( loss)
    plt.pause(0.001)
   # time.sleep(0.5)    
    y_pred = train(neural_n, X_test, y_test , der_fcost, lr=0.0001, train = False)
    NMSE =   nmse(y_test, y_pred)
    print("Valor del NMSE conseguido = ")
    print (NMSE)    
    