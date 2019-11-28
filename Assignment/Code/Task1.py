import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Get Polynomial Data Matrix given coordinates and degree
def get_poly_data_matrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1, degree+1):
        X = np.column_stack((X, x**i))
    return X


##Get Weights/Parameters given coorinates and polynomial degree
def get_weights(x,y,degree):
    if degree > 0:
        X = get_poly_data_matrix(x,degree)
        XX  = X.transpose().dot(X)
        weight = np.linalg.solve(XX, X.transpose().dot(y))
    else:
        weight = sum(y)/len(x)
    return weight



##Calculating the Root Mean Square Error, given
def eval_pol_regression(parameters, x, y, degree):
    rmse = 0

    ##Squaring the residuals
    ##Finding the average
    ##Taking the square root of the result

    
    xMatrix = get_poly_data_matrix(x,degree)
    #Calculate Root Mean Square Error
    rmse = np.sqrt(((xMatrix.dot(parameters)-y)**2).mean())
    return rmse



##Plot the Data
def plot_data(axs, sX, para):
    #Get matrix of x coordinates based on x = -5 -> 5 (step 0.1).
    if isinstance(para, np.float64): ##If degree = 0, then para has no length
        axs[sX].plot([-5,5],[para,para])
    else:
        xCoords = get_poly_data_matrix(np.arange(-5,5,0.1),len(para)-1)
        yCoords = xCoords.dot(para)
        axs[sX].plot(np.arange(-5,5,0.1),yCoords)
    return


##Polynomial Regression
def pol_regression(x_train, y_train, degree):
    parameters = get_weights(x_train, y_train, degree)
    plot_data(axs,0,parameters)

    return parameters



#Get Data
def get_data():
    ##Import data from csv file
    data = pd.read_csv('data_pol_regression.csv')
    
    #Randomise the data
    data = data.sample(frac=1)
    return data



##Sort Data
def sort_data(x_in, y_in):
    #Creates a dataframe of x and y coordinates
    df = pd.DataFrame([x_in, y_in])
    
    #Use the data frames prebuilt sorting function to sort the data in order of x coordinates
    df.sort_values(0, axis=1, ascending = True, inplace = True)
    
    #Initialise coordinate arrays
    x = []
    y = []
    
    #For every point in the dataframe, append each coord list with its respective coordinate
    for point in df:
        x = np.append(x, df[point][0])
        y = np.append(y, df[point][1])
    return x, y



##Get Training Data
def get_training_data(data, split):
    #Initialse X and Y coords from the data
    x_input = data['x'].as_matrix()
    y_input = data['y'].as_matrix() 
    
    #Determines the size of traning data set based on the float parsed
    training_size = round((len(x_input)*split))
    
    #Initialises x and y training coordinate arrays
    x_train = []
    y_train = []
    
    #Appends coordinate points to the array
    for i in range(training_size):
        x_train = np.append(x_train, x_input[i])
        y_train = np.append(y_train, y_input[i])
    
    #Sorts data into order 
    x_train, y_train = sort_data(x_train, y_train)
        
    #Initialises x and y testing coordinate arrays
    x_test = []
    y_test = []
    
    #If the split isn't 100% training, put remaining coordinates into the test arrays
    if split != 1:
        for i in range(training_size,len(data)):
            x_test = np.append(x_test, x_input[i])
            y_test = np.append(y_test, y_input[i])
        #Sort the data
        x_test, y_test = sort_data(x_test, y_test)
    return x_train, y_train, x_test, y_test



####    Main Loop    #####
data = get_data()
x_train, y_train, x_test, y_test = get_training_data(data, 1)
degrees = [0,1,2,3,5,10]
parameters = [None]*6

####    Initialising Graphs Data    ####
fig, axs = plt.subplots(2)
axs[0].set_xlim(-5,5)
axs[0].set_ylim(-200,50)
axs[1].set_yscale('log')
#axs[1].set_ylim(0,10)


##Plotting *all* given data
axs[0].plot(x_train, y_train, 'bo')

##Plotting polynomials of degree 1,2,3,5,10
for i in range(len(degrees)):
    parameters[i] = pol_regression(x_train, y_train, degrees[i])
    #plot_data(axs,0,parameters[i])
axs[0].legend(['Ground Truth','$x^{0}$', '$x^{1}$','$x^{2}$','$x^{3}$','$x^{5}$','$x^{10}$'],bbox_to_anchor=(1.05,1))


#Training and Testing data
x_train, y_train, x_test, y_test = get_training_data(data, 0.7)
axs[1].set(xlabel='Degree', ylabel='RSME')

####    Show data with only 70% training data    ####
#for i in range(1,len(degrees)):
#    parameters[i] = pol_regression(x_train, y_train, degrees[i])
#    plot_data(axs,0,parameters[i])

##Initialise the test and training rsme lists
rsme_Train = [None]*6
rsme_Test = [None]*6

##For each degree, determine parameters for those lists, then calculates the RMSE for the training and test data.
for i in range(len(degrees)):
    parameters[i] = get_weights(x_train,y_train,degrees[i])
    rsme_Train[i] = eval_pol_regression(parameters[i], x_train, y_train, degrees[i])
    rsme_Test[i] = eval_pol_regression(parameters[i], x_test, y_test, degrees[i]) #(len(parameters[i])-1)

##Plot Data
axs[1].plot(degrees,rsme_Train)
axs[1].plot(degrees,rsme_Test)
axs[1].legend(["Training Data","Test Data"],bbox_to_anchor=(1.35,0.75))

