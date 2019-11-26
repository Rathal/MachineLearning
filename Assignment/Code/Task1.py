import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import numpy.linalg as linalg

def get_poly_data_matrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1, degree+1):
        X = np.column_stack((X, x**i))
#    print(X)
    return X



def get_weights(x,y,degree):
    X = get_poly_data_matrix(x,degree)
    XX  = X.transpose().dot(X)
    weight = np.linalg.solve(XX, X.transpose().dot(y))
#    weight = np.polyfit(x,y,degree)
#    print(weight)
    return weight





def eval_pol_regression(parameters, x, y, degree):
    rmse = 0
    
    ##Squaring the residuals
    ##Finding the average
    ##Taking the square root of the result
    
    ## (y - wx^2 + wx + w)^2
    ## sum of all the points
    ## sqrt the result
#    print("Parameters:")
#    print(parameters)
    sum_residuals = 0
#    print("Maths:")
    for i in range(0, len(parameters)):
        sum_residuals += (parameters[i])*(x**i)
#        print(sum_residuals)
#    print("SE:")
    se = y - sum_residuals
    for i in se:
        rmse += i
    for i in x:
        rmse += parameters[len(parameters)-1]*i
#    print(rmse)
    rmse = rmse**2
#    print(len(x))
    rmse = rmse / len(x)
#    print(rmse)
    rmse = np.sqrt(rmse)
#    print(rmse)
    
    return rmse





def plot_data(axs, sX, para):
    y_coords = []
    for x in np.arange(-5,5,0.1):
        sumY = 0
        for b in range(len(para)):
        #        exp = len(para)-(b+1)
            exp = b
            coeff = para[b]
        #        print("{0}X^{1}".format(coeff,exp))
            sumY += coeff * (x**exp)
#            print(sumY)
        y_coords.append(sumY)
    axs[sX].plot(np.arange(-5,5,0.1),y_coords)
    return



def pol_regression(x_train, y_train, degree):
    #Code
    parameters = 0
#    print(coeffs)
#    print(coeffs[0])
    if degree > 0:
        w = get_weights(x_train, y_train, degree)
        Xtest = get_poly_data_matrix(x_train, degree)
        ytest = Xtest.dot(w)
        x_test, y_test = sort_data(x_train, ytest)
#        plt.plot(x_test, y_test)
        parameters = w
#    else:        
#        plt.plot([-5,5],[0,0])
    
    
    return parameters




#Get Data
def get_data():
    data = pd.read_csv('data_pol_regression.csv')
#    print(data)
    data = data.sample(frac=1)
#    print(data)
    return data





def sort_data(x_in, y_in):
##Sort Data
        df = pd.DataFrame([x_in, y_in])
        df.sort_values(0, axis=1, ascending = True, inplace = True)
        x = []
        y = []
        for point in df:
            x = np.append(x, df[point][0])
            y = np.append(y, df[point][1])
        return x, y






def get_training_data(data, split):
    x_input = data['x'].as_matrix()
    y_input = data['y'].as_matrix() 
#    print(len(x_input)*split)
    test_size = round((len(x_input)*split))
#    print("Training Data:")
    x_train = []
    y_train = []
    for i in range(test_size):
#        print("X:{0} : Y:{1}".format(x_input[i],y_input[i]))
        x_train = np.append(x_train, x_input[i])
        y_train = np.append(y_train, y_input[i])
#    print("Test Data:")
    x_train, y_train = sort_data(x_train, y_train)
        
    x_test = []
    y_test = []
    if split != 1:    
        
        for i in range(test_size,len(data)):
    #        print("X:{0} : Y:{1}".format(x_input[i],y_input[i]))
            x_test = np.append(x_test, x_input[i])
            y_test = np.append(y_test, y_input[i])
    x_test, y_test = sort_data(x_test, y_test)
    
    
    return x_train, y_train, x_test, y_test

####    Main Loop    #####
data = get_data()
x_train, y_train, x_test, y_test = get_training_data(data, 1)
####    Plotting Data    ####

fig, axs = plt.subplots(2)

#plt.figure()
axs[0].set_xlim(-5,5)
axs[0].set_ylim(-200,50)
axs[0].plot(x_train, y_train, 'bo')

para0 = pol_regression(x_train, y_train, 0)
para1 = pol_regression(x_train, y_train, 1)
para2 = pol_regression(x_train, y_train, 2)
para3 = pol_regression(x_train, y_train, 3)
para5 = pol_regression(x_train, y_train, 5)
para10 = pol_regression(x_train, y_train, 10)

plot_data(axs,0,para1)
plot_data(axs,0,para2)
plot_data(axs,0,para3)
plot_data(axs,0,para5)
plot_data(axs,0,para10)
axs[0].legend(["Data",0,1,2,3,5,10],bbox_to_anchor=(1.05,1))

#y_coords = []
#for x in np.arange(-5,5,0.1):
#    sumY = 0
#    for b in range(len(para10)):
##        exp = len(para2)-(b+1)
#        exp = b
#        coeff = para10[b]
##        print("{0}X^{1}".format(coeff,exp))
#        sumY += coeff * (x**exp)
#        print(sumY)
#    y_coords.append(sumY)

#plt.plot(np.arange(-5,5,0.1),y_coords)






x_train, y_train, x_test, y_test = get_training_data(data, 0.7)

axs[1].set(xlabel='Degree', ylabel='RSME')

rsme1 = eval_pol_regression(para1, x_train, y_train, (len(para1)-1))
rsme2 = eval_pol_regression(para2, x_train, y_train, (len(para2)-1))
rsme3 = eval_pol_regression(para3, x_train, y_train, (len(para3)-1))
rsme5 = eval_pol_regression(para5, x_train, y_train, (len(para5)-1))
rsme10 = eval_pol_regression(para10, x_train, y_train, (len(para10)-1))
axs[1].plot([1,2,3,4,10],[rsme1,rsme2,rsme3,rsme5,rsme10])

rsme1 = eval_pol_regression(para1, x_test, y_test, (len(para1)-1))
rsme2 = eval_pol_regression(para2, x_test, y_test, (len(para2)-1))
rsme3 = eval_pol_regression(para3, x_test, y_test, (len(para3)-1))
rsme5 = eval_pol_regression(para5, x_test, y_test, (len(para5)-1))
rsme10 = eval_pol_regression(para10, x_test, y_test, (len(para10)-1))
axs[1].plot([1,2,3,5,10],[rsme1,rsme2,rsme3,rsme5,rsme10])
axs[1].legend(["Training Data","Test Data"],bbox_to_anchor=(1.35,0.75))
