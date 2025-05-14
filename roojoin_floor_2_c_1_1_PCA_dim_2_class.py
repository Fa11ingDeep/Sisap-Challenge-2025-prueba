import numpy as np
import math
import h5py
import csv
import os
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

def Manhattan(v1,v2):
    """
    Calcula la distancia de Manhattan entre dos vectores v1 y v2.

    Parámetros:
    v1 (array-like): Primer vector.
    v2 (array-like): Segundo vector.

    Retorna:
    float: La distancia de Manhattan entre v1 y v2.
    """
    v3=v1-v2
    return sum(abs(x) for x in v3)

def cos_sim(v1, v2):
    """
    Calcula la similitud del coseno entre dos vectores v1 y v2.

    La similitud del coseno mide la similitud entre dos vectores en un espacio de 
    características, comparando el ángulo entre ellos. El valor devuelto varía entre 
    -1 (totalmente opuestos) y 1 (idénticos), con 0 significando que no tienen similitud.

    Parámetros:
    v1 (array-like): Primer vector.
    v2 (array-like): Segundo vector.

    Retorna:
    float: La similitud del coseno entre v1 y v2.
    """
    return  1-(np.dot(v1,v2))
@dataclass
class Point: # un vector con su id
    id: int
    vector: np.ndarray

@dataclass
class LabeledPoint: # un point con los proximos grupos más cercanos
    point: Point
    nearest_groups: list 

@dataclass
class Group: # un grupo que tiene lista de point, un radio y el punto más lejano
    points: list 
    radius: float =-1
    furthest_point: Point = None

def getCenters(data, c):
    """
    Selecciona aleatoriamente un subconjunto de puntos de datos para usarlos como centros de clústeres.

    La función elige un número de centros basados en un factor de `c` multiplicado por la raíz cuadrada del número total de puntos de datos. 
    Luego, elimina los centros seleccionados de los datos restantes.

    Parámetros:
    data (numpy.ndarray): Una matriz de datos de tamaño (n, d), donde n es el número de puntos de datos y d es la dimensionalidad de cada punto.
    c (float): Un factor que se utiliza para calcular cuántos centros se deben seleccionar. El número de centros será `c * sqrt(n)`.

    Retorna:
    tuple: Una tupla que contiene:
        - un entero, el número total de puntos en `data`,
        - una lista de los puntos restantes después de seleccionar los centros (como objetos `Point`),
        - una lista de los puntos seleccionados como centros (como objetos `Point`).
    """
    n, d = data.shape  # Obtenemos la cantidad de vectores en los datos
    centers = []  # Los centros que se eligen
    newData = []  # Los vectores sin los centros
    idx = np.random.choice(n, size=math.floor(c * math.sqrt(n)), replace=False)  # Elegimos aleatoriamente c*raiz de n indices sin reposición
    # Crear los puntos y agregar a la lista de centros
    for i in idx:
        centers.append(Point(id=int(i), vector=data[i]))  # Crear el Point y agregarlo a centers
    newData = [Point(id=int(i), vector=vector) for i, vector in enumerate(data)]# Enumerar los datos y crear la tupla (índice, valor) para cada elemento de data
    newData = [item for item in newData if item.id not in idx]# Eliminar los puntos elegidos para ser centros
    return n, newData, centers