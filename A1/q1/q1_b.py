# ANSWER 1.a - after 2*10^6 iterations
# 1.1947898109839311e-06: Cost
# ('Final theta', array([9.90349706e-01, 7.77771542e-04]))
# After fixing this value of intercept, I was able to get the same slope in half iterations than before (i.e. 10^6)
# 1.1947898109838752e-06
# ('Final theta', array([9.90349706e-01, 7.77771503e-04]))

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
threshold = 1.6855447430592224e-10

f1 = open('./q1/linearX.csv')
f2 = open('./q1/linearY.csv')

Data = np.zeros(shape=(100,2))

cnt = 0
for x in f1:
    Data[cnt][0] = float(x.split('\n')[0])
    cnt += 1

cnt = 0
for y in f2:
    Data[cnt][1] = float(y.split('\n')[0])
    cnt += 1

# Normalizing
temp = Data[:,0].copy()
temp -= np.mean(Data[:, 0])
temp /= np.std(Data[:, 0])
Data[:, 0] = temp

# temp = Data[:,1].copy()
# temp -= np.mean(Data[:, 1])
# temp /= np.std(Data[:, 1])
# Data[:, 1] = temp


theta = np.ones(2)
learning_rate = 0.025

def hypothesis(x, theta):
    return theta[0] + x*theta[1]

def cost(Data, theta):
    sum = 0
    for i in range(100):
            sum = sum + (0.01*(Data[i,1]-hypothesis(Data[i,0], theta))**2)/2
    return sum

def error(Data, theta, num):
    sum = 0
    if num == 0:
        for i in range(100):
            sum = sum + (0.01*(Data[i,1]-hypothesis(Data[i,0], theta)))/2	
    elif num == 1:
        for i in range(100):
            sum = sum + (0.01*num*Data[i,0]*(Data[i,1]-hypothesis(Data[i,0], theta)))/2
    return sum

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    plt.title("Data Set Q1") 
    plt.xlabel("Acidity of wine") 
    plt.ylabel("Density of wine") 
    plt.plot(Data[:,0], Data[:,1], "ob")
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
    plt.show()

# theta[0] = 0.99662096
diff = 1
i=0
while(diff>threshold):
    temp = cost(Data, theta)
#     print('X')
    theta[0] = theta[0] + learning_rate*error(Data, theta, 0)
    theta[1] = theta[1] + learning_rate*error(Data, theta, 1)
    diff = temp - cost(Data, theta)
    print(diff)
    if(i == 999999):
        print(temp - cost(Data, theta), i)
    i += 1
print('Number of iterations:%s' % str(i))
# SGD
# for i in range(100000):
# 	for j in range(100/2):
# 		theta[0] = theta[0] + learning_rate*(Data[j,1]-hypothesis(Data[j,0],theta))/(2*(100/2))
# 		theta[1] = theta[1] + learning_rate*(Data[j,0])*(Data[j,1]-hypothesis(Data[j,0],theta))/(2*(100/2))
# 	print(cost(Data, theta))

print(cost(Data, theta))
print('Final theta',theta)
abline(theta[1],theta[0])



# 2.773200821746989e-05 // SGD cost after 10^5 iterations with full dataset
# 4.620409793755026e-05 // BGD cost after 10^5 iterations but half sample
