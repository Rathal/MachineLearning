import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#Calculate the (euclidean) distancce of two vectors
def compute_euclidean_distance(vec_1, vec_2):
    distance = np.linalg.norm(vec_1-vec_2)
    return distance

#Randomly initializes the centroids
def initialise_centroids(data, k):
    #Create k-number of Coordinates. Randomly picked from sample size
    centroids = [None]*k
    for i in range(k):
        centroids[i] = data[random.randrange(0,len(data))]
    return centroids

#Removes 'None' values from each cluster g
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

##Get centroids new locations
def update_centroids(cluster_assigned, centroids, k):
    change = 0
    #Store Old Centroids
    old_centroids = {}
    for i in range(k):
        old_centroids['centroid_{0}'.format(i)] = abs(centroids[i])
    
    #Remove None values from each cluster g
    if k == 3:
        g1, g2, g3 = sanitize_cluster(cluster_assigned, k)
    else: g1, g2 = sanitize_cluster(cluster_assigned, k)
    
    #Get Mean Location of Centroids
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
    
    #Get rate of change
    change = {}
    for i in range(k):
        change['centroid_{0}'.format(i)] = abs(old_centroids['centroid_{0}'.format(i)] - centroids[i])
    return change, centroids

##Get Coordinates of points in each cluster g
def get_coords(cluster_assigned, k):
    if k == 3:
            g1, g2, g3 = sanitize_cluster(cluster_assigned, k)
    else: g1, g2 = sanitize_cluster(cluster_assigned, k)
    
    #Initialisations
    x1 = []
    y1 = []
    
    x2 = []
    y2 = []
    
    #For each point in each cluster g, get coordinates
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


##clusters the data into k groups
def kmeans(data, k):
    #Initialisations
    change_list = []
    cluster_assigned = None
    Repeat = True
    centroids = initialise_centroids(data, k)
    
    #Repeat until local minima is found
    while Repeat:
        Repeat = False
        
        #Initialisations
        l = len(data)
        cluster_assigned = {}
        for i in range(k):
            cluster_assigned['centroid_{0}'.format(i)] = [None]*l
        distance = [0]*k
        
        
        for i in range(l):
            #Determines closest centroid
           for j in range(k):
               nearest = None
               distance[j] = compute_euclidean_distance(centroids[j], data[i])
               nearest = distance.index(min(distance))
               
            #Assigns each datapoint to the cluster of its nearest centroid
           cluster_assigned['centroid_{0}'.format(nearest)][i] = data[i]
    
        #Update Centroids
        change, centroids = update_centroids(cluster_assigned, centroids, k)
        
        #Is the change == 0?
        change_sum = 0
        for i in range(k):
            #Gets rate of change from old and new centroid locations
            change_sum += change['centroid_{0}'.format(i)][0]+change['centroid_{0}'.format(i)][1]
            
            #If there is a change, repeat, else stop
            if abs(change['centroid_{0}'.format(i)][0])+change['centroid_{0}'.format(i)][1] > 0:
                Repeat = True
                plt.clf()
        #Keep track of changes for objective function
        change_list = np.append(change_list, change_sum)
        
    return centroids, cluster_assigned, change_list





##Gets data from .csv file
def get_data(col):
    #Read data from .csv file
    dataInput = pd.read_csv('k_means.csv').values
#    np.random.shuffle(dataInput) #Seemingly no difference
    
    #Initialise new data array
    data = np.zeros((0,2))

    #Concatenate coordinates into data array
    for i in dataInput:
        a = np.array([i[0], i[col]])
        data = np.vstack([data, a])
    return data



def plot_data(ax, qX, qY, x, y, colour, shape):
    ax[qX,qY].scatter(x,y, c=colour, marker=shape)



##########################
####    MAIN LOOP    #####

#Initialise datasets 1 and 2
data1 = get_data(1)
data2 = get_data(2)

# for k = 2 & k = 3
for k in range(2,4):

    #Running kmeans algorithm for each dataset
    centroids1, cluster_assigned1, changes1 = kmeans(data1, k)
    centroids2, cluster_assigned2, changes2 = kmeans(data2, k)
    
    #Get coordinates of each point in a cluster for dataset 1
    if k == 3:
        x1, x2, x3, y1, y2, y3 = get_coords(cluster_assigned1, k)
    else:
        x1, x2, y1, y2 = get_coords(cluster_assigned1, k)
    
    #Defining subplots
    fig, ax = plt.subplots(2, 2)
    
    ##Plot clusters for dataset 1
    plot_data(ax, 0, 0, x1, y1, 'r', 'o')
    plot_data(ax, 0, 0, x2, y2, 'g', 'o')
    
    ##Plot centroids for dataset 1
    plot_data(ax, 0, 0, centroids1[0][0], centroids1[0][1], 'm', 'x')
    plot_data(ax, 0, 0, centroids1[1][0], centroids1[1][1], 'y', 'x')
    ax[0,0].set(xlabel='Height', ylabel='Tail Length')
    
    ##Get coords of each point in a cluster for dataset 2
    if k == 3:
        x12, x22, x32, y12, y22, y32 = get_coords(cluster_assigned2, k)
    else:
        x12, x22, y12, y22 = get_coords(cluster_assigned2, k)
        
    #Calculate changes made per iteration in dataset 2 for plotting
    c1 = []
    c1_size = []
    for i in range(len(changes1)):
        c1.append(changes1[i])
        c1_size.append(i)
    
    #Plot changes against iteration step for dataset 1
    ax[1,0].plot(c1_size,c1)
    ax[1,0].set(xlabel='Iteration Step', ylabel='Objective Function')
    
    #Plot clusters for dataset 2
    plot_data(ax, 0, 1, x12, y12, 'r', 'o')
    plot_data(ax, 0, 1, x22, y22, 'g', 'o')
    
    #plot centroids for dataset 2
    plot_data(ax, 0, 1, centroids2[0][0], centroids2[0][1], 'm', 'x')
    plot_data(ax, 0, 1, centroids2[1][0], centroids2[1][1], 'y', 'x')
    
    #label graph
    ax[0,1].set(xlabel='Height', ylabel='Leg Length')
    
    #plot third centroids and cluster fro dataset 2
    if k == 3:
        plot_data(ax, 0, 0, x3, y3, 'b', 'o')
        plot_data(ax, 0, 0, centroids1[2][0], centroids1[2][1], 'c', 'x')
        plot_data(ax, 0, 1, x32, y32, 'b', 'o')
        plot_data(ax, 0, 1, centroids2[2][0], centroids2[2][1], 'c', 'x')
    
    #Calculate changes made per iteration in dataset 2 for plotting
    c2 = []
    c2_size = []
    for i in range(len(changes2)):
        c2.append(changes2[i])
        c2_size.append(i)
        
    #Plot changes against iteration step for dataset 2
    ax[1,1].plot(c2_size,c2)
    ax[1,1].set(xlabel='Iteration Step', ylabel='Objective Function')
    
    #Show each set of graphs for each k value
    plt.show()