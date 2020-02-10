#!/usr/bin/env python
# coding: utf-8

# In[1]:

# comment printing before running
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import system
import os
from IPython.display import clear_output

# Params
lr = 0.001
batch_size = 10000
SAMPLING_NUM = 1000000
COST_THRESHOLD = 2.1e-05

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
def update_line(hl, new_data):
    xdata, ydata, zdata = hl._verts3d
    hl.set_xdata(list(np.append(xdata, new_data[0])))
    hl.set_ydata(list(np.append(ydata, new_data[1])))
    hl.set_3d_properties(list(np.append(zdata, new_data[2])))
    plt.draw()

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)


X, Y = np.meshgrid(x, y)

def cost(x_data, y_data, params, ll, ul):
    sum = 0
    for i in range(ll, ul):
        sum = sum + ((1/(ul-ll+1))*(y_data[i]-np.dot(params, x_data[i]))**2)/2
    return sum

def sgd_main(x_data, y_data, do3DPlot):
    m = int(y_data.shape[0])
    r = batch_size
    tru_theta = np.ones(shape = (1,3))
    tru_theta[0][0] = 3
    tru_theta[0][2] = 2
    theta = np.zeros(shape = (1, 3))
    epoch = 0
    ll = 0
    ul = r
    prevcost=0
    if(do3DPlot):
        map = plt.figure()
        map_ax = Axes3D(map)
        map_ax.autoscale(enable=True, axis='both', tight=True)
        # # Setting the axes properties
        map_ax.set_xlim3d([0.0, 5.0])
        map_ax.set_ylim3d([0.0, 5.0])
        map_ax.set_zlim3d([0.0, 5.0])
        hl, = map_ax.plot3D([theta[0][0]], [theta[0][1]], [theta[0][2]])

    while(abs(cost(x_data, y_data, theta, ll, ul))>COST_THRESHOLD):
        p = 0
        ll=0
        ul=r
        
        while(p <= m):    
            # p-th batch
            if(do3DPlot):
                update_line(hl, (theta[0][0], theta[0][1], theta[0][2]))
                plt.show(block=False)
                plt.pause(0.2)
            
#             os.system('clear')
#             print('=================EPOCH %s=================' % epoch)
            for i in range(p, r+p):
                gradient = np.zeros(3)
                for j in range(3):
                    gradient[j] += ((y_data[i] - np.dot(theta, x_data[i])) * x_data[i][j])/r
                    theta[0][j] += lr * gradient[j]
#             print('Batch %s out of %s batches\n' % (p/r, m/r))
#             print(theta[0])
            if(p==0):
                print('epoch')
                print(theta[0])
            p += r
            if(p == m):
                break
            now_cost = cost(x_data, y_data, theta, ll, ul)
            print('COST NOW: %s \n' % now_cost)
            if(abs(now_cost-cost(x_data, y_data, tru_theta, ll, ul))<COST_THRESHOLD):
                return theta
            ll += r
            ul += r
        epoch+=1
    return theta


# In[24]:


# learned_params = sgd_main(x_data, y_data)

# # Report
# 1. batch_size = 10 | threshold = 
# 2. batch_size = 100 | threshold = 
# 3. batch_size = 10000 | threshold = 
# 4. batch_size
# because of low lr, not able to see much randomness in the path of sgd even with batch_size = 1

def test(filename, learned_params):
    TESTING_SIZE = 10000
    testing_x = np.ones(shape = (TESTING_SIZE, 3))
    testing_y = np.ones(shape = (TESTING_SIZE, 1))
    print(testing_x[:,0])
    print(testing_y[:,0])

    f = open(filename,'r')
    cnt = 0
    for x in f:
        lt = x.split(',')
        testing_x[cnt][0] = 1
        testing_x[cnt][1] = lt[0]
        testing_x[cnt][2] = lt[1]
        testing_y[cnt][0] = lt[2]
        cnt += 1

    print(testing_y.shape)
    print(testing_x.shape)

    print(cost(testing_x, testing_y, learned_params,0, TESTING_SIZE))

def plot_data(x_data, y_data):
    plt.title("Data Set Q2 - sampled") 
    plt.xlabel("x1") 
    plt.ylabel("x2") 
    plt.scatter(x_data[:,1], x_data[:,2], s=0.1)
    plt.show()
    plt.close()

def main(x_data, y_data):
#     plot_data(x_data, y_data)
#     prama = np.ones(shape = (1,3))
#     prama[0][0] = 2.96718453
#     prama[0][1] = 1.08841158
#     prama[0][2] = 2.13035516
#     print(cost(x_data, y_data, prama, 0, 100))
    if(batch_size==100):
        num_batch_thresh = 2200
        COST_THRESHOLD = 1e-04
    elif(batch_size==10000):
        COST_THRESHOLD=2.1e-05
    learned_params = sgd_main(x_data, y_data, do3DPlot = True)
#     test('./q2/q2test.csv', learned_params)
    
if __name__ == '__main__':
    # INIT
    params = np.zeros(shape = (3,1))
    params[0][0] = 3
    params[1][0] = 1
    params[2][0] = 2
    x_data = np.ones(shape=(SAMPLING_NUM,3)) # 1, x1, x2
    y_data = np.zeros(shape = (SAMPLING_NUM, 1)) # y
    noise = np.zeros(shape = (SAMPLING_NUM, 1))
    noise[:,0] = np.random.normal(0, 1.414, SAMPLING_NUM)
    x_data[:,1] = np.random.normal(3, 2, SAMPLING_NUM)
    x_data[:,2] = np.random.normal(-1, 2, SAMPLING_NUM)

    y_data = np.dot(x_data, params)
    y_data = np.add(y_data, noise)
    main(x_data, y_data)