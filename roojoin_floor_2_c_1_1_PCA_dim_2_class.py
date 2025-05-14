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
    v3=v1-v2
    return sum(abs(x) for x in v3)

def cos_sim(v1, v2):
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