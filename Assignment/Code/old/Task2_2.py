# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:45:12 2019

@author: Daniel Guy (16607811)
"""
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt

def compute_euclidean_distance(vec_1, vec_2):
    distance = 0
    #Code
    return distance

def initialise_centroids(data, k):
    centroids = np.random.uniform(10.0, size=(k,2))
    return centroids

def kmeans(data,k):
    centroids = 0
    cluster_assigned = 0
    #Code
    return centroids, cluster_assigned

def get_data(col):
    dataInput = pd.read_csv('k_means.csv').values    
    #np.random.shuffle(dataInput)
    
    data = np.zeros((0,2))

    for i in dataInput:
        a = np.array([i[0], i[col]])
        data = np.vstack([data, a])

    return data


#####    MAIN LOOP    #######
    
