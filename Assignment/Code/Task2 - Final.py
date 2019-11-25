import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#Calculate the (euclidean) distancce of two vectors
def compute_euclidean_distance(vec_1, vec_2):
    #Code
    distance = np.linalg.norm(vec_1-vec_2)
    
    return distance

#Randomly initializes the centroids
def initialise_centroids(data, k):
    #Create k-number of Coordinates 
#    centroids = np.random.uniform(10.0, size=(k,2)) #Completely Random
    
    #Create k-number of Coordinates. Randomly picked from sample size
    centroids = [None]*k
    for i in range(k):
        centroids[i] = data[random.randrange(0,len(data))]
#    print(centroids)
    
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
#    plt.scatter(centroids[0][0], centroids[0][1], c="m", marker="x")
#    plt.scatter(centroids[1][0], centroids[1][1], c="y", marker="x")
#    if k == 3:
#        plt.scatter(centroids[2][0], centroids[2][1], c='c', marker='c')
    return change, centroids

def get_coords(cluster_assigned, k):
    if k == 3:
            g1, g2, g3 = sanitize_cluster(cluster_assigned, k)
    else: g1, g2 = sanitize_cluster(cluster_assigned, k)
#        print(len(g1))
#        print(g1)
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
        
    if k == 3:
        return x1,x2,x3,y1,y2,y3
    else:
        return x1, x2, y1, y2


#clusters the data into k groups
def kmeans(data, k):
    #Code   
    change_list = []
    cluster_assigned = None
    Repeat = True
    centroids = initialise_centroids(data, k)
    while Repeat:
        Repeat = False
        #plot_data(data)
        #plot_data(centroids)
        
        l = len(data)
        cluster_assigned = {}
        
        for i in range(k):
            cluster_assigned['centroid_{0}'.format(i)] = [None]*l
    
        distance = [0]*k
        for i in range(l):
           for j in range(k):
               nearest = None
               distance[j] = compute_euclidean_distance(centroids[j], data[i])
#               print("{0}:{1}".format(i,j))
#               print(distance[j])
               nearest = distance.index(min(distance))
#               print(nearest)
           cluster_assigned['centroid_{0}'.format(nearest)][i] = data[i]

        #Update Centroids
        change, centroids = update_centroids(cluster_assigned, centroids, k)
#        print(change['centroid_0'][0]+change['centroid_0'][1])
        
        #Is the change == 0?
        change_sum = 0
        for i in range(k):
            change_sum += change['centroid_{0}'.format(i)][0]+change['centroid_{0}'.format(i)][1]
#            print(change_sum)
            if abs(change['centroid_{0}'.format(i)][0])+change['centroid_{0}'.format(i)][1] > 0:
                Repeat = True
                plt.clf()
        change_list = np.append(change_list, change_sum)
#    print(change_list)
    #print(changeList)
    #print(change_list)
    return centroids, cluster_assigned, change_list





#Gets data from .csv file
def get_data(col):
    dataInput = pd.read_csv('k_means.csv').values    
    #np.random.shuffle(dataInput)
    
    data = np.zeros((0,2))

    for i in dataInput:
        a = np.array([i[0], i[col]])
        data = np.vstack([data, a])

    return data

def plot_data(ax, qX, qY, x, y, colour, shape):
    ax[qX,qY].scatter(x,y, c=colour, marker=shape)

#############################################
####    MAIN    ###############


data1 = get_data(1)
data2 = get_data(2)

k = 3

centroids1, cluster_assigned1, changes1 = kmeans(data1, k)
centroids2, cluster_assigned2, changes2 = kmeans(data2, k)

if k == 3:
    x1, x2, x3, y1, y2, y3 = get_coords(cluster_assigned1, k)
else:
    x1, x2, y1, y2 = get_coords(cluster_assigned1, k)

fig, ax = plt.subplots(2, 2)

##Plot k = 2 graph 0,0
plot_data(ax, 0, 0, x1, y1, 'r', 'o')
plot_data(ax, 0, 0, x2, y2, 'g', 'o')
plot_data(ax, 0, 0, centroids1[0][0], centroids1[0][1], 'm', 'x')
plot_data(ax, 0, 0, centroids1[1][0], centroids1[1][1], 'y', 'x')
ax[0,0].set(xlabel='Height', ylabel='Tail Length')

##Plot k=3 graph 0,0
if k == 3:
    x12, x22, x32, y12, y22, y32 = get_coords(cluster_assigned2, k)
else:
    x12, x22, y12, y22 = get_coords(cluster_assigned2, k)
    
##Plot changes made in 1,0
c1 = []
c1_size = []
for i in range(len(changes1)):
    c1.append(changes1[i])
    c1_size.append(i)

ax[1,0].plot(c1_size,c1)
ax[1,0].set(xlabel='Iteration Step', ylabel='Objective Function')

#ax[1,0].plot([len(changes1)],[changes1])

    
plot_data(ax, 0, 1, x12, y12, 'r', 'o')
plot_data(ax, 0, 1, x22, y22, 'g', 'o')
plot_data(ax, 0, 1, centroids2[0][0], centroids2[0][1], 'm', 'x')
plot_data(ax, 0, 1, centroids2[1][0], centroids2[1][1], 'y', 'x')
ax[0,1].set(xlabel='Height', ylabel='Leg Length')

if k == 3:
    plot_data(ax, 0, 0, x3, y3, 'b', 'o')
    plot_data(ax, 0, 0, centroids1[2][0], centroids1[2][1], 'c', 'x')
    plot_data(ax, 0, 1, x32, y32, 'b', 'o')
    plot_data(ax, 0, 1, centroids2[2][0], centroids2[2][1], 'c', 'x')

c2 = []
c2_size = []
for i in range(len(changes2)):
    c2.append(changes2[i])
    c2_size.append(i)

ax[1,1].plot(c2_size,c2)
ax[1,1].set(xlabel='Iteration Step', ylabel='Objective Function')

plt.show()
