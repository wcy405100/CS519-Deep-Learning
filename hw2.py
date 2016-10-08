# name : Chenyu Wang
# ONID: 932-079-604

# this program need to input some parameters
# The preferred parameters are:
# Input the hidden unit size: 100
# Please input the iteration number: 25
# Please input the mini-batch size: 100
# Please input the learn rate: 0.01

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import rl 
import linear_tr as lt
import sigmod as smod
import output_layer as ot
import cross_entropy as ce

# subgradient of W2;
# z2 is final layer output; y is ground truth;
# z1 is the hiddenlayer output
def subg_w2(z2,y,z1):
    return (z2-y)*np.array(z1)

# subgradient of b2
def subg_b2(z2,y):
    return z2-y

# subgradient of w1; z2 is final layer output;
# y is ground truth;
# ReLu_g is the gradient of ReLu,
# x is the input image
def subg_w1(z2,y,w2,ReLu_g,x):
    w2 = w2.reshape(len(w2),1)
    change = z2-y
    temp = np.dot(w2,change)
    temp1 = temp * ReLu_g
    temp2 = np.outer(x,np.transpose(temp1))
    return temp2

# subgradient of b1
def subg_b1(z2,y,w2,ReLu_g):
    temp = np.multiply(ReLu_g,w2)
    return (z2-y)*np.array(temp)

# forward network, input all necessary parameters
# return:
# layer_temp = w1*x + b1
# layer1, reLu_gradient = ReLu result,ReLu gradient of layer_temp
# layer2_temp = w2*layer1 +b2
# layer2 = sigmod result of layer2_temp
def forward(x,w1,b1,w2,b2):
    layer1_temp = lt.calcu(x,w1,b1)
    layer1, reLu_gradient = rl.ReLu(layer1_temp)
    layer2_temp = lt.calcu(layer1,w2,b2)
    layer2 = smod.sigmod(layer2_temp)
    return layer1_temp,layer1,reLu_gradient,layer2_temp,layer2

# stochastic mini-batch gradient descent
# i is the mini batch size
# alp is the learning rate
# mom_r is the momentum rate
# in the function we update w1,w2,b1,b2 after every mini-batch size with momentum
def grad_des(i,alp,mom_r):
    global tr_img , tr_lab ,w1,b1,w2,b2,hiddensize,momen1,momen2
    train_size = len(dict["train_data"])
    mbatches_img = [tr_img[j:j+i] for j in range(0, train_size,i)]
    mbatches_lab = [tr_lab[j:j+i] for j in range(0, train_size,i)]
    for ind in range(len(mbatches_img)):
        lay1_temp,lay1,ReLu_g,lay2_temp,lay2 = forward(mbatches_img[ind],w1,b1,w2,b2)
        temp_w2 = [0]*hiddensize
        temp_b2 = 0
        temp_w1 = np.zeros((3072,hiddensize))
        temp_b1 = [0]*hiddensize
        for g in range(len(mbatches_img[ind])):
            temp_w2 = temp_w2 + subg_w2(lay2[g],mbatches_lab[ind][g],lay1[g])
            temp_w1 = temp_w1 + subg_w1(lay2[g],mbatches_lab[ind][g],w2,ReLu_g[g],mbatches_img[ind][g])
            temp_b2 = temp_b2 + subg_b2(lay2[g],mbatches_lab[ind][g])
            temp_b1 = temp_b1 + subg_b1(lay2[g],mbatches_lab[ind][g],w2,ReLu_g[g])
        momen2 = mom_r*momen2 - (alp/i)*temp_w2
        w2 += momen2
        momen1 = mom_r*momen1 - (alp/i)*temp_w1
        w1 += momen1
        b2 -= (alp/i)*temp_b2
        b1 -= (alp/i)*temp_b1
    print "Stochastic gradient descent with mini-batch size " + str(i) + " and momentum 0.8 is done!"

# the accuracy function gives the overall accuracy of training or testing images
def accuracy(a2,ground_t):
    ground_t = np.squeeze(ground_t)
    temp = abs(a2 - ground_t)
    acc = float(np.sum(temp))/float(len(ground_t))
    if acc > 0.5:
        return acc
    else:
        return (1 - acc)

if __name__ == "__main__":
    dict = cPickle.load(open("cifar_2class_py2.p", "rb"))

    # gain data from dataset
    tr_img = dict["train_data"]
    tr_lab = dict["train_labels"]
    ts_img = dict["test_data"]
    ts_lab = dict["test_labels"]

    # set the hidden unit size, train iterations, mini batch size and learning rate
    hiddensize = int(raw_input("Input the hidden unit size: "))
    n = int(raw_input("Please input the iteration number: "))
    m = int(raw_input("Please input the mini-batch size: "))
    p = float(raw_input("Please input the learn rate: "))

    # initialize the w1, b1, w2, b2 and momentum for w1 and w2
    w1 = np.random.randn(3072,hiddensize)/np.sqrt(3072)
    w2 = np.random.randn(hiddensize)/np.sqrt(hiddensize)
    b1 = np.random.randn(hiddensize)
    b2 = 1
    momen1 = np.zeros((3072,hiddensize))
    momen2 = np.zeros(hiddensize)

    # normalize the testing and training data for better results
    tr_img = [(x-np.mean(x))/np.std(x) for x in tr_img]
    ts_img = [(x-np.mean(x))/np.std(x) for x in ts_img]
    res = [[],[]]

    #starting the for loop of training
    for i in range(n):
        grad_des(m,p,0.8)
        lay1_temp,lay1,ReLu_g,lay2_temp,lay2 = forward(tr_img,w1,b1,w2,b2)
        temp,temp,temp,temp,lay2_ts = forward(ts_img,w1,b1,w2,b2)  # use trained weight to predict the testing image
        pred = ot.output(lay2)  # output {0,1} value for each training image
        pred_ts = ot.output(lay2_ts) # output {0,1} value for each testing image
        accu = accuracy(pred,tr_lab)
        res[0].append(accu)
        accu_ts = accuracy(pred_ts,ts_lab)
        res[1].append(accu_ts)
        entropy = ce.cross_en(lay2,tr_lab)
        entropy_ts = ce.cross_en(lay2_ts,ts_lab)
        print "This is the "+str(i+1)+"th iteration."
        print "The cross entropy of training set is:" + str(entropy)
        print "The cross entropy of testing set is:" + str(entropy_ts)
        print "The training accuracy is:" + str(accu)
        print "The testing accuracy is:" + str(accu_ts)

    # plot the accuracy from training and testing images
    axes = plt.gca()
    plt.plot(np.linspace(1,n,n),np.squeeze(res[0]),color="blue", linewidth=2.5, linestyle="-",label ="training")
    plt.plot(np.linspace(1,n,n),np.squeeze(res[1]),color="red", linewidth=2.5, linestyle="-",label ="testing")
    plt.legend(loc='upper left')
    axes.set_ylim([0.5,1])
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.show()
    print "done!"