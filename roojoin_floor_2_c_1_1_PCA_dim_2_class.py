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


def makeGroups(n, data, centers, metric_fn, c, size):
    """
    Agrupa puntos de datos en clusters basados en los centros iniciales utilizando una métrica de distancia.

    Esta función agrupa los puntos de datos (`data`) en clústeres definidos por los centros iniciales proporcionados en `centers`. 
    Los puntos se asignan a los grupos más cercanos basándose en una función de métrica (`metric_fn`). 
    La función ajusta dinámicamente el tamaño de los grupos, asegurándose de que cada grupo no supere un tamaño máximo, y expande los grupos si es necesario.

    Parámetros:
    n (int): El número total de puntos de datos.
    data (list of LabeledPoint): Una lista de puntos de datos etiquetados (`LabeledPoint`) que serán asignados a grupos.
    centers (list of Point): Una lista de objetos `Point` que representan los centros iniciales de los grupos.
    metric_fn (function): Una función que calcula la distancia entre dos vectores (como la distancia Euclidiana, Manhattan, etc.).
    c (float): Un factor que afecta el tamaño máximo de los grupos, multiplicado por el tamaño del grupo.
    size (int): El tamaño total de los datos (número de puntos de datos en `data`).

    Retorna:
    dict: Un diccionario donde las claves son los índices de los centros, y los valores son los grupos (instancias de `Group`).
        Cada grupo contiene:
            - Una lista de puntos (`LabeledPoint`) asignados a ese grupo.
            - El radio máximo de ese grupo.
            - El punto más lejano de ese grupo.
    """
    maxSize = math.floor(c * size) #Tamaño maximo del grupo
    groups = {}

    # Calcular el próximo grupo más cercano para los centros
    for i in range(len(centers)):
        center_dist = []
        for j in range(len(centers)):
            if i != j:
                center_dist.append((j, metric_fn(centers[i].vector, centers[j].vector)))  # Inserción en la lista (candidato a próximo centro, dist)
        center_dist.sort(key=lambda x: x[1])  # Ordenar según distancia

        next_centers = []
        for k in range(len(center_dist)):
            next_center, dist = center_dist[k]
            next_centers.append(next_center)
        groups[i] = Group([LabeledPoint(point=centers[i], nearest_groups=next_centers[:2])], radius=-1,furthest_point=None)  # La segunda variable (-1) es el radio
    # Agregar puntos a los grupos
    for h in range((maxSize - 1) * len(centers)):
        id_point = data[h]
        dists = []
        for num, values in groups.items():  # En points, el primer punto siempre es el centro
            datas = values.points
            idx, point = id_point.id, id_point.vector
            id_points= datas[0].point
            points = id_points.vector

            dists.append((metric_fn(point, points), num, idx))  # Agregamos las distancias con el num del grupo

        dists.sort(key=lambda x: x[0])  # Reordenar los puntos
        k = 0
        while True:
            dist1, num1, id = dists[k]
            if len(groups[num1].points) < maxSize:  # Si el grupo era menor que el tamaño máximo
                dists.remove(dists[k])  # Eliminamos el punto
                next_group_list = []  # Lista de centros
                for l in range(2):  # Extrae los centros de las tuplas y los coloca en la lista
                    dist_temp, group_temp, id = dists[l]
                    next_group_list.append(group_temp)
                groups[num1].points.append(LabeledPoint(Point(id=id, vector=point), nearest_groups=next_group_list))  # Agrega el nuevo punto con los nuevos datos
                if dist1 > groups[num1].radius:  # Si la nueva distancia del punto hacia el centro es más grande que el radio que había
                    groups[num1].radius=dist1  # Actualiza el radio del grupo y el punto más lejano
                    groups[num1].furthest_point=Point(id=id, vector=point)
                break  # Salir del loop
            else:  # Estaba lleno el grupo
                k += 1  # Ir al próximo grupo
        

    # Expansión de grupos (si es necesario)
    x = math.ceil((n - maxSize * len(centers)) / len(centers))
    
    if x != 0:
        extendedsizegroup = maxSize + x
        extendedpointsingroups = len(centers) * extendedsizegroup
        assert extendedpointsingroups >= n
        o = ((maxSize - 1) * len(centers))  # Posición del primer punto no insertado en la parte anterior
        nuevosPendientes = []
        while o < len(data):
            id_point_extended = data[o]
            dists_extended = []
            for num, values in groups.items():  # En points, el primer punto siempre es el centro
                datas = values.points
                idx, point = id_point_extended.id, id_point_extended.vector
                id_points= datas[0].point
                points = id_points.vector
                dists_extended.append((metric_fn(point, points), num, idx))  # Agregamos las distancias con el num del grupo

            dists_extended.sort(key=lambda x: x[0])
            p = 0
            while True:
                dist1, num1, id = dists_extended[p]
                
                if len(groups[num1].points) < extendedsizegroup:  # Si el grupo era menor que el tamaño máximo nuevo
                    dists_extended.remove(dists_extended[p])  # Eliminamos el punto
                    next_group_list = []  # Lista de centros
                    for u in range(2):  # Extrae los centros de las tuplas y los coloca en la lista
                        dist_temp, group_temp, id = dists_extended[u]
                        next_group_list.append(group_temp)
                    groups[num1].points.append(LabeledPoint(Point(id=id, vector=point), nearest_groups=next_group_list))
                    if dist1 > groups[num1].radius:  # Si la nueva distancia del punto hacia el centro es más grande que el radio que había
                        groups[num1].radius=dist1  # Actualiza el radio del grupo y el punto más lejano
                        groups[num1].furthest_point=Point(id=id, vector=point)
                    o += 1
                    break  # Salir del loop
                else:  # Estaba lleno el grupo
                    if(dist1<groups[num1].radius): # Si la distancia que tengo es menor que el radio y el grupo estaba lleno, lo insertamos de toda forma
                        if(groups[num1].furthest_point!=None):
                            nuevosPendientes.append(groups[num1].furthest_point)# agregamos el punto mas lejano como nuevos pendientes
                            groups[num1].points = [item for item in groups[num1].points if item.point.id != groups[num1].furthest_point.id]
                        dists_extended.remove(dists_extended[p])
                        next_group_list=[] # Lista de centros
                        for p in range(2): # Extrae los centros de las tuplas y lo coloca en la lista
                            dist_temp,group_temp,id=dists_extended[p]
                            next_group_list.append(group_temp)
                        groups[num1].points.append(LabeledPoint(Point(id=id, vector=point), nearest_groups=next_group_list))  # agrega el nuevo punto con los nuevos datos
                        
                        groups[num1].radius=-10000 # Actualiza el radio del grupo para indicar que ya se hizo el cambio
                        groups[num1].furthest_point=None # Actualiza el punto mas lejano del grupo para indicar que ya se hizo el cambio
                        o+=1
                        break # Salir del loop
                    else:# Si ya esta lleno y mi radio era mayor, ir al siguiente grupo
                        p+=1 # Ir al prox grupo

        # Procesar puntos pendientes
        for id_point in nuevosPendientes:
            dists_pend = []
            for num, values in groups.items():  # En points, el primer punto siempre es el centro
                datas = values.points
                idx, point = id_point.id, id_point.vector
                id_points = datas[0].point
                points = id_points.vector
                dists_pend.append((metric_fn(point, points), num, idx))  # Agregamos las distancias con el num del centro

            dists_pend.sort(key=lambda x: x[0])
            l = 0
            while True:
                dist1, num1, id = dists_pend[l]
                
                if len(groups[num1].points) < extendedsizegroup:  # Si el grupo era menor que el tamaño máximo nuevo
                    dists_pend.remove(dists_pend[l])  # Eliminamos el punto
                    next_group_list = []  # Lista de centros
                    for u in range(2):  # Extrae los centros de las tuplas y los coloca en la lista
                        dist_temp, group_temp, id = dists_pend[u]
                        next_group_list.append(group_temp)
                    groups[num1].points.append(LabeledPoint(Point(id=id, vector=point), nearest_groups=next_group_list))  # Agrega el nuevo punto con los nuevos datos
                    if (dist1 > groups[num1].radius and (groups[num1].radius >-10000)):  # Si la nueva distancia del punto hacia el centro es más grande que el radio que había
                        groups[num1].radius=dist1  # Actualiza el radio del grupo y el punto más lejano
                        groups[num1].furthest_point=Point(id=id, vector=point)
                    break  # Salir del loop
                else:  # Estaba lleno el grupo
                    l += 1  # Ir al siguiente grupo
    return groups

def get_knn(k,e,target,metric_fn):
    """
    Encuentra los k vecinos más cercanos de un punto dado usando una función de métrica.

    Esta función toma un punto de entrada `e` y calcula las distancias a todos los puntos en el conjunto `target` 
    utilizando una función de métrica (`metric_fn`). Luego, devuelve los índices de los `k` puntos más cercanos 
    a `e`, ordenados por distancia creciente.

    Parámetros:
    k (int): El número de vecinos más cercanos que se desean obtener.
    e (Point): El punto de referencia al que se le calcularán las distancias.
    target (list of LabeledPoint): Una lista de objetos `LabeledPoint` que contienen los puntos a comparar.
    metric_fn (function): Una función que calcula la distancia entre dos vectores. Esta función debe tomar dos vectores 
                           como entrada y devolver un valor de distancia.

    Retorna:
    list: Una lista de los `k` índices de los puntos más cercanos a `e` en el conjunto `target`.
    """
    temp=[] # Arreglo temporal para guardar las distancias
    target=[item for item in target if not np.array_equal(item.point.vector, e)] # Quitamos el elemento para que no aparezca de nuevo
    for element in target:
        id_element=element.point.id
        point_element=element.point.vector
        dist=metric_fn(e,point_element) # Para cada elemento se calcula la distancia
        temp.append((dist,id_element)) # Se agrega la tupla distancia, punto
    temp.sort(key=lambda x: x[0]) # Se ordena segun la distancia
    return  [int(x[1]) for x in temp][:k] # Retorna los k elementos