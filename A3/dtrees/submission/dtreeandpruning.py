"""Implementation of the CART algorithm to train decision tree classifiers."""
import numpy as np
# import pandas as pd
from xclib.data import data_utils
import matplotlib.pylab as plt
from random import seed
from random import randrange
import math
import time
import numexpr as ne
from multiprocessing import Pool
from joblib import Parallel, delayed
from collections import deque

# paths
data_folder_path = './data/'
train_x_path = data_folder_path + 'train_x.txt'
train_y_path = data_folder_path + 'train_y.txt'
train_size = (64713, 482)
test_x_path = data_folder_path + 'test_x.txt'
test_y_path = data_folder_path + 'test_y.txt'
test_size = (21571, 482)
val_x_path = data_folder_path + 'valid_x.txt'
val_y_path = data_folder_path + 'valid_y.txt'
val_size = (21572, 482)

# data_folder_path = './data/'
# train_x_path = data_folder_path + 'small_x.txt'
# train_y_path = data_folder_path + 'small_y.txt'
# # train_size = (64713, 482)
# train_size = (14,4)
# test_x_path = data_folder_path + 'small_x.txt'
# test_y_path = data_folder_path + 'small_y.txt'
# # test_size = (21571, 482)
# test_size = (14,4)
# val_x_path = data_folder_path + 'small_x.txt'
# val_y_path = data_folder_path + 'small_y.txt'
# val_size = (14,4)


def load_y(path, size):
    y = np.ones(size)
    f = open(path)
    cnt = 0
    for x in f:
        y[cnt] = int(x)
        cnt += 1
    return y

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.index = 1
        self.depth = 0


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, 1)


    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _entropy(self, X, y):
        m = y.size
        n = X.shape[0]
        a2 = np.count_nonzero(y==1)
        a1 = m - a2
        
        # print('ggg', a1, a2)
        if(a1==0):
            return 0
        elif(a2==0):
            return 1
        return (-1 * (a1/y.shape[0]) * math.log2((a1/y.shape[0])) + (-1 * (a2/y.shape[0]) * math.log2((a2/y.shape[0]))))

            # return 0.5
    def _tovectoriz(self, X, y, idx, entropy_class):
        # print(idx)
        median = np.median(X[:, idx])
        
        data_left = np.array([np.array([i, clas]) for i,clas in zip(X[:, idx],y) if i<=median])
        if(data_left.size==0):
            entropy_left = 0
            size0 = 0
        else:
            entropy_left = self._entropy(data_left[:,0], data_left[:,1])
            size0 = data_left[:,1].shape[0]

        
        data_right = np.array([np.array([i, clas]) for i,clas in zip(X[:, idx],y) if i>median])
        if(data_right.size==0):
            entropy_right = 0
            size1 = 0
        else:
            entropy_right = self._entropy(data_right[:,0], data_right[:,1])
            size1 = data_right[:,1].shape[0]
        
        temp = size0/(size0+size1)
        varr = (temp * entropy_left + (1-temp) * entropy_right)
        # print('CHUUT',varr, entropy_class)
        return varr

    # def _best_split(self, X, y):
    #     m = y.size
    #     n = X.shape[1]
    #     if m<=1:
    #         return None, None
        
    #     best_idx, best_thr = None, None
    #     entropy_class = self._entropy(X, y)
        
    #     entropy_attr = [(entropy_class - self._tovectoriz(X, y, i, entropy_class)) for i in range(n)]
        
    #     # entropy_attr = [0] * n
    #     # for i in range(n):
    #     #     remp = self._tovectoriz(X, y, i, entropy_class)
    #     #     # if(remp>entropy_class):
    #     #     #     print(remp, entropy_class)
    #     #     #     return None, None
    #     #     entropy_attr[i] = entropy_class - self._tovectoriz(X, y, i, entropy_class)

    #     # entropy_attr = [0] * (n)
    #     # entropy_attr = Parallel(n_jobs=-1, verbose=0, backend="threading")(map(delayed(self._tovectoriz), X, y, i, entropy_class))

    #     # print(entropy_attr)
    #     # if(np.count_nonzero(np.array(entropy_attr)==0)==len(entropy_attr)):
    #     #     return None, None
    #     best_idx = np.argmax(entropy_attr)
    #     best_thr = np.median(X[:,best_idx])
    #     if(entropy_attr[best_idx]>=entropy_class):
    #         return None, None
        
    #     return best_idx, best_thr

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = int(classes[i - 1])
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
                # print(best_idx, best_thr)
        # print(best_idx, best_thr)
        return best_idx, best_thr

    def _grow_tree(self, X, y, index, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(2)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        node.index = index
        print(depth)
        idx = -1
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            # if(depth>0):
                # print('Le del=khle: ', idx, thr)
            if idx is not None:
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X[X[:,idx]<=thr],  y[X[:, idx]<=thr], node.index*2, depth + 1)
                node.left.depth = depth + 1
                node.right = self._grow_tree(X[X[:, idx]>thr], y[X[:, idx]>thr], node.index*2+1, depth + 1)
                node.right.depth = depth + 1

        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

def get_accuracy(clf, a, b):
    i = 0
    cnt = 0
    for x in clf.predict(a):
        if(x==b[i]):
            cnt += 1
        i+=1
    return (cnt/i)

if __name__ == "__main__":
    import sys
    from sklearn.datasets import load_iris
    #___________________________________________________________________________________
    train_x = data_utils.read_sparse_file(train_x_path)
    train_y = load_y(train_y_path, train_x.shape[0])

    test_x = data_utils.read_sparse_file(test_x_path)
    test_y = load_y(test_y_path, test_x.shape[0])

    val_x = data_utils.read_sparse_file(val_x_path)
    val_y = load_y(val_y_path, val_x.shape[0])

    i = 0
    train_data = np.zeros(shape=train_size)
    for x in train_x.toarray():
        train_data[i] = [int(i) for i in x]
        i += 1
    train_data = train_data[:100, :]
    train_y = train_y[:100]
    
    i = 0
    test_data = np.zeros(shape=test_size)
    for x in test_x.toarray():
        test_data[i] = [int(i) for i in x]
        i += 1
    test_data = test_data[:100, :]
    test_y = test_y[:100]
    
    i = 0
    val_data = np.zeros(shape=val_size)
    for x in val_x.toarray():
        val_data[i] = [int(i) for i in x]
        i += 1
    val_data = val_data[:100, :]
    val_y = val_y[:100]

    print('Data loaded...')
    #___________________________________________________________________________________
   
    # dataset = load_iris()

    X, y = train_data, train_y  # pylint: disable=no-member
    
    DO_ONE_TIME = True

    if(DO_ONE_TIME):
        clf = DecisionTreeClassifier(max_depth=40)
        start = time.time()
        clf.fit(X, y)
        print('Training Time: ', (time.time()-start)/60)

        # Debugging the root node
        # print('---------------------classifier info---------------------')
        # # print('Node index: ', clf.tree_.index)
        # print('Feature Attribute: ', clf.tree_.feature_index)
        # print('Threshold: ', clf.tree_.threshold)
        # print('Left: ', clf.tree_.left)
        # print('Right: ', clf.tree_.right)
        
        # print('---------------------------------------------------------')

        # appending every index into ll and finding max index among them
        q = deque()
        q.append(clf.tree_)
        
        ll = []
        cnt = 0
        while(q):
            temp = q.popleft()
            if(temp is None):
                continue
            if(temp and temp.left is not None):
                q.append(temp.left)
            if(temp and temp.right is not None):
                q.append(temp.right)
            ll.append(temp.index)

            # print('index and depth: ', temp.index, temp.depth)
            cnt += 1

        # print(ll)
        index_max = ll[len(ll)-1]
        # print(index_max)


        # Creating bfs-list
        # bfs_list = [-1]*index_max
        # for i in range(index_max):
        #     if(i+1 in ll):
        #         bfs_list[i] = i+1
        
        # print('BFS LIST -> ')
        # print(bfs_list)
        # print('Max index', index_max)
        # print('Num nodes: ', cnt)

        # PRUNING
        print('pruning starts...')
        best_score = get_accuracy(clf, val_data, val_y)
        lx = [cnt]
        l_val = [best_score]
        l_train = [get_accuracy(clf, train_data, train_y)]
        l_test = [get_accuracy(clf, test_data, test_y)]
        for x in ll:
            q = deque()
            q.append(clf.tree_)
            left = None
            right = None
            pruned_depth = None
            while(q):
                temp = q.popleft()
                if(temp is None):
                    continue
                if(temp.left==None and temp.right==None):
                    continue
                if(temp and temp.left is not None):
                    q.append(temp.left)
                if(temp and temp.right is not None):
                    q.append(temp.right)
                
                # ll.append(temp.index)
                if(temp.index==x):
                    # print('pruned!!')
                    left = temp.left
                    right = temp.right
                    pruned_depth = temp.depth
                    print(temp.depth)
                    temp.left = None
                    temp.right = None
                    break

            now_score = get_accuracy(clf, val_data, val_y)
            # print('Now score: ', now_score)
            if(now_score>best_score):
                print(best_score, now_score)
                best_score = now_score
                cnt += 2**(pruned_depth-4)
                lx.append(lx[0]-cnt)
                l_val.append(best_score)
                l_train.append(get_accuracy(clf, train_data, train_y))
                l_test.append(get_accuracy(clf, test_data, test_y))
            else:
                temp.left = left
                temp.right = right

            # print('nm nodes after pruning/not pruning %d: %d' % (x, cnt))
        print('Val accuracy', best_score)

        print(lx)
        print(l_val)
        print(l_test)
        print(l_train)
        fig, ax = plt.subplots()
        ax.set_xlim(lx[0]+1, lx[len(lx)-1]-1)
        plt.plot(lx, l_train, label='Training acc')
        plt.plot(lx, l_test, label='Testing acc')
        plt.plot(lx, l_val, label = 'Validation acc')
        plt.legend(loc = 'upper right')
        plt.savefig('pruning_py.png')


        # print('Train accuracy', get_accuracy(clf, train_data, train_y))

        # print('Train accuracy', get_accuracy(clf, test_data, test_y))
        
        # print('Val accuracy', get_accuracy(clf, val_data, val_y))
    else:

        lx = [1,2,3,4,5]
        la = [0] * len(lx)
        ly = [0] * len(lx)
        lz = [0] * len(lx)
        cntg = 0
        plt.xlabel('depth ->')
        plt.ylabel('accuracy ->')
        for i in lx:
            clf = DecisionTreeClassifier(max_depth=i)
            start = time.time()
            clf.fit(X, y)
            print('(depth = ' + str(i) + ')Training Time:  ', (time.time()-start)/60)

            i = 0
            cnt = 0
            for x in clf.predict(train_data):
                if(x==train_y[i]):
                    cnt += 1
                i+=1
            la[cntg] = (cnt/i)

            i = 0
            cnt = 0
            for x in clf.predict(test_data):
                if(x==test_y[i]):
                    cnt += 1
                i+=1
            ly[cntg] = (cnt/i)

            i = 0
            cnt = 0
            for x in clf.predict(val_data):
                if(x==val_y[i]):
                    cnt += 1
                i+=1
            lz[cntg] = (cnt/i)

            cntg += 1
        print(ly)
        print(lz)
        print(la)
        plt.plot(la, ly, label='Training acc')
        plt.plot(lx, ly, label='Testing acc')
        plt.plot(lx, lz, label = 'Validation acc')
        plt.legend(loc = 'upper right')
        plt.savefig('output.png')




# PART - A
# Training Time:  26.030078411102295
# 0.7889295813824115

# max_depth = 14
# Training Time:  14.470615061124166
# Train accuracy 0.8073184676958262
# Test accuracy 0.7821148764544991
# Val accuracy 0.7762377155572038


# (depth = 24)Training Time:   40.05412704149882
# (depth = 25)Training Time:   42.37587464650472
# (depth = 26)Training Time:   45.35138691663742
# [0.6564832413889018, 0.6624171341152473, 0.6671457048815539, 0.667794724398498, 0.6841592879328728, 0.7057623661397247, 0.7162393954846785, 0.7348755273283575, 0.7418756664039683, 0.7513791664735061, 0.7521672616012238, 0.7728431690695842, 0.7768763617820221, 0.7821148764544991, 0.7824393862129712, 0.7835056325622364, 0.7860089935561634, 0.7861480691669371, 0.7889295813824115, 0.7863335033146354, 0.7850818228176719, 0.7853599740392193, 0.7839228593945575, 0.7838765008576329, 0.7819294423068008]
# [0.6553866122751715, 0.6615983682551455, 0.6634526237715557, 0.66604858149453, 0.6785184498423883, 0.6988225477470795, 0.712961246059707, 0.7287224179491933, 0.7367420730576674, 0.7446690153903208, 0.7468014092341925, 0.7648340441312813, 0.7703040979046912, 0.7762377155572038, 0.776840348600037, 0.780826997960319, 0.781754125718524, 0.7818004821064343, 0.7840255887261265, 0.7840719451140367, 0.7824958279250881, 0.7831911737437419, 0.7816150565547932, 0.7812905618394215, 0.7811514926756907]
# [0.6569777324494306, 0.6624634926521719, 0.6654149861697032, 0.6673620447205353, 0.6846846846846847, 0.7076321604623491, 0.7215551743853631, 0.7400367777726268, 0.7500193160570519, 0.7596155332004388, 0.76621389828937, 0.7876779008854481, 0.7976604391698732, 0.8073184676958262, 0.8150139848253056, 0.8237139369214841, 0.8299877922519432, 0.8366016101865158, 0.8494738306059061, 0.8559022143927805, 0.8616506729714277, 0.8684190193624156, 0.8729776088266654, 0.8788342373248034, 0.8831919397957134]





# PRUNING 
# lx = [7279, 7278, 7277, 7276, 7275, 7274, 7273, 7272, 7271, 7270, 7269, 7268, 7267, 7266, 7265, 7264, 7263, 7262, 7261, 7260, 7259, 7258, 7257, 7256, 7255, 7254, 7253, 7252, 7251, 7250, 7249, 7248, 7247, 7246, 7245, 7244, 7243, 7242, 7241, 7240, 7239, 7238, 7237, 7236, 7235, 7234, 7233, 7232, 7231, 7230, 7229, 7228, 7227, 7226, 7225, 7224, 7223, 7222, 7221, 7220, 7219, 7218, 7217, 7216, 7215, 7214, 7213, 7212, 7211, 7210, 7209, 7208, 7207, 7206, 7205, 7204, 7203, 7202, 7201, 7200, 7199, 7198, 7197, 7196, 7195, 7194, 7193, 7192, 7191, 7190, 7189, 7188, 7187, 7186, 7185, 7184, 7183, 7182, 7181]
# Validation accuracy:
# l_val = [0.7850454292601521, 0.7865288336732802, 0.7868069720007417, 0.787038753940293, 0.7875023178193955, 0.788151307250139, 0.7881976636380493, 0.7883367328017801, 0.7884294455776006, 0.7884758019655108, 0.7885221583534211, 0.7886612275171518, 0.7887539402929724, 0.7888466530687929, 0.7888930094567032, 0.7898201372149082, 0.7898664936028185, 0.7900055627665492, 0.7903764138698313, 0.7906081958093826, 0.790700908585203, 0.7912108288522158, 0.7913498980159466, 0.7913962544038569, 0.7915353235675876, 0.7915816799554979, 0.7916280363434082, 0.792137956610421, 0.7921843129983311, 0.7922306693862414, 0.7922770257741517, 0.7923697385499722, 0.7924160949378825, 0.7924624513257927, 0.7925088077137029, 0.7925551641016132, 0.7927405896532542, 0.7927869460411645, 0.7932041535323567, 0.793250509920267, 0.7933432226960875, 0.7933895790839978, 0.7935286482477285, 0.7937140737993695, 0.7937604301872798, 0.7940849249026516, 0.794177637678472, 0.7942239940663823, 0.7944094196180234, 0.7945484887817541, 0.7945948451696644, 0.7946412015575747, 0.7947339143333951, 0.7947802707213054, 0.7948266271092157, 0.7948729834971259, 0.7949656962729464, 0.7950584090487669, 0.7951047654366772, 0.7951511218245875, 0.7951974782124976, 0.7952438346004079, 0.7955219729278694, 0.7955683293157797, 0.7957073984795104, 0.7957537548674207, 0.795800111255331, 0.7958464676432412, 0.7958928240311515, 0.7959391804190618, 0.7961709623586131, 0.7962173187465232, 0.7963100315223438, 0.7964027442981643, 0.7964491006860746, 0.7964954570739848, 0.796541813461895, 0.7965881698498053, 0.7966345262377156, 0.796727239013536, 0.7967735954014463, 0.7968663081772668, 0.7969126645651771, 0.7969590209530873, 0.7970517337289078, 0.7971444465047284, 0.7971908028926386, 0.7972835156684591, 0.7973298720563694, 0.7973762284442796, 0.7974225848321899, 0.7975152976080104, 0.7975616539959206, 0.7977007231596513, 0.7977470795475616, 0.7977934359354719, 0.7978397923233822, 0.7979325050992027, 0.7980252178750232]
# Training + Pruning + Testing Time:
# 71.4684647123019