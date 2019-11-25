import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import random as r

#Calculate the (euclidean) distancce of two vectors
def compute_euclidean_distance(vec_1, vec_2):
    #Code
    distance = np.linalg.norm(vec_1-vec_2)
    
    return distance

#Randomly initializes the centroids
def initialise_centroids(data, k):
    #Create k-number of Coordinates 
    centroids = np.random.uniform(10.0, size=(k,2))
#    centroids[0][0] = 3
#    centroids[0][1] = 5
#    centroids[1][0] = 9
#    centroids[1][1] = 2
    return centroids


def sanitize_cluster(cluster_assigned, k):
    g1 = cluster_assigned['centroid_0']
    g1 = [g for g in g1 if g is not None]
    
    g2 = cluster_assigned['centroid_1']
    g2 = [g for g in g2 if g is not None]
    
    if k == 3:
        g3 = cluster_assigned['centroid_2']
        g3 = [g for g in g3 if g is not None]   
    
    if k == 3:
        return g1, g2, g3
    return g1, g2

def update_centroids(cluster_assigned, centroids, k):
    #Code    
    change = 0
    #Store Old Centroids
    old_centroids = {}
    for i in range(k):
        old_centroids['centroid_{0}'.format(i)] = abs(centroids[i])
    #print(old_centroids)
    
    #Get Mean Location of Centroids
    if k == 3:
        g1, g2, g3 = sanitize_cluster(cluster_assigned, k)
    else: g1, g2 = sanitize_cluster(cluster_assigned, k)
    #print(centroids)
    
    if k == 3:
        if len(g1) > 0:
            centroids[0] = sum(g1)/len(g1)
        if len(g2) > 0:
            centroids[1] = sum(g2)/len(g2)
        if len(g3) > 0:
            centroids[2] = sum(g3)/len(g3)
    else:
        if len(g1) > 0:
            centroids[0] = sum(g1)/len(g1)
        if len(g2) > 0:
            centroids[1] = sum(g2)/len(g2)
    #print(centroids)
    
    #Get coords of clusters    
    #Update Centroids
    
    #Get rate of change
    change = {}
    for i in range(k):
        change['centroid_{0}'.format(i)] = abs(old_centroids['centroid_{0}'.format(i)] - centroids[i])
    #print(change)
    
    #Plot new centroid locations (X)
    plt.scatter(centroids[0][0], centroids[0][1], c="m", marker="x")
    plt.scatter(centroids[1][0], centroids[1][1], c="y", marker="x")
    if k == 3:
        plt.scatter(centroids[2][0], centroids[2][1], c='c', marker='c')
    return change, centroids

#clusters the data into k groups
def kmeans(data, k):
    #Code
    centroids = initialise_centroids(data, k)
    #plot_data(data)
    #plot_data(centroids)
    plt.scatter(centroids[0][0], centroids[0][1], c="m", marker='+')
    plt.scatter(centroids[1][0], centroids[1][1], c="y", marker='+')
    if k == 3:
        plt.scatter(centroids[2][0], centroids[2][1], c='c', marker='+')
    

    l = len(data)
    cluster_assigned = {}
    
    for i in range(k):
        cluster_assigned['centroid_{0}'.format(i)] = [None]*l

    distance = [0]*k
    for i in range(l):
       for j in range(k):
           distance[j] = compute_euclidean_distance(centroids[j], data[i])
           nearest = distance.index(min(distance))
           cluster_assigned['centroid_{0}'.format(nearest)][i] = data[i]

    
    if k == 3:
        g1, g2, g3 = sanitize_cluster(cluster_assigned, k)
    else: g1, g2 = sanitize_cluster(cluster_assigned, k)
    
    x1 = []
    y1 = []
    
    x2 = []
    y2 = []
    
    for point in g1:
        x1.append(point[0])
        y1.append(point[1])
    for point in g2:
        x2.append(point[0])
        y2.append(point[1])
        
    if k == 3:
        x3 = []
        y3 = []
        for point in g3:
            x3.append(point[0])
            y3.append(point[1])
        plt.scatter(x3,y3, c='b')
        
    plt.scatter(x1,y1, c='r')
    plt.scatter(x2,y2, c='g')
    
    #Update Centroids
    change, centroids = update_centroids(cluster_assigned, centroids, k)
    
    #Is the change == 0?
    for i in range(k):
        if abs(change['centroid_{0}'.format(i)][0])+change['centroid_{0}'.format(i)][1] > 0:
            Repeat = True
    if Repeat:
        print("Repeating")
        plt.clf()
        kmeans(data, k)
    
    #OLD CODE
#    distance = [None] * k
#    for point in data:
#        for i in range(k):
#            distance[i] = compute_euclidean_distance(centroids[i], point) 
#        #returns the position (and thus the centroid) each point is closest to
#        nearest = distance.index(min(distance))
#        cluster_assigned = np.append(cluster_assigned, [nearest, point])
#    c1 = centroids[0]
#    c2 = centroids[1]
    #OLD CODE
    
    #print(cluster_assigned)
    
#    if k == 3:
#        c3 = centroids[2]
#    for i in range(k):
#        sum = np.array((0,0))
#        for n in cluster_assigned:
#            if n == i:
#                sum = np.add(sum)
#        mean = np.divide(sum, cluster_assigned.count(i))
#        print(mean)
    
    return centroids, cluster_assigned





#Gets data from .csv file
def get_data(col):
    dataInput = pd.read_csv('k_means.csv').values    
    #np.random.shuffle(dataInput)
    
    data = np.zeros((0,2))

    for i in dataInput:
        a = np.array([i[0], i[col]])
        data = np.vstack([data, a])

    return data

def plot_data(data):
    x = data[:,0]
    y = data[:,1]
    plt.scatter(x,y)

data1 = get_data(1)
data2 = get_data(2)

centroids, cluster_assigned = kmeans(data1, 2)


#plt.scatter()
#plt.scatter(x,y)



#plt.scatter(centroids[0][0],centroids[0][1])
#plt.scatter(centroids[1][0],centroids[1][1])

#point = np.array(x[0],y1[0])
#centroid = np.array(centroids[0])


#print(compute_euclidean_distance(centroid,point))


#print(data[])
#print(compute_euclidean_distance(x[0],y1[0]))