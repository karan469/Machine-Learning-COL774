# ANSWER 1.a - after 2*10^6 iterations
# 1.1947898109839311e-06: Cost
# ('Final theta', array([9.90349706e-01, 7.77771542e-04]))
# After fixing this value of intercept, I was able to get the same slope in half iterations than before (i.e. 10^6)
# 1.1947898109838752e-06
# ('Final theta', array([9.90349706e-01, 7.77771503e-04]))

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
f1 = open('./data/q1/linearX.csv')
f2 = open('./data/q1/linearY.csv')

Data = np.zeros(shape=(100,2))

cnt = 0
for x in f1:
	Data[cnt][0] = float(x.split('\n')[0])
	cnt += 1

cnt = 0
for y in f2:
	Data[cnt][1] = float(y.split('\n')[0])
	cnt += 1


plt.title("Data Set Q1") 
plt.xlabel("Acidity of wine") 
plt.ylabel("Density of wine") 
plt.plot(Data[:,0], Data[:,1], "ob")
# plt.show()

theta = np.ones(2)
learning_rate = 0.0004

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
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

# theta[0] = 9.90349706e-01
for i in range(1000000):
	# temp = cost(Data, theta)
	theta[0] = theta[0] + learning_rate*error(Data, theta, 0)
	theta[1] = theta[1] + learning_rate*error(Data, theta, 1)
	# print(temp - cost(Data, theta))

# SGD
# for i in range(100000):
# 	for j in range(100/2):
# 		theta[0] = theta[0] + learning_rate*(Data[j,1]-hypothesis(Data[j,0],theta))/(2*(100/2))
# 		theta[1] = theta[1] + learning_rate*(Data[j,0])*(Data[j,1]-hypothesis(Data[j,0],theta))/(2*(100/2))
# 	print(cost(Data, theta))

print(cost(Data, theta))
print('Final theta',theta)
abline(theta[1],theta[0])
plt.show()


# 2.773200821746989e-05 // SGD cost after 10^5 iterations with full dataset
# 4.620409793755026e-05 // BGD cost after 10^5 iterations but half sample
