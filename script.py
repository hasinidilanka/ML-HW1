import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import math


K_list = [3]
N_list = [5,10,15,20,25,30]
lamda = [0.01, 0.1, 1, 10]
normal_loss = [0]*len(N_list)
ridge_loss = [[0]*len(N_list)]*len(lamda)



for K in K_list:

    # Plot between -10 and 10 with .001 steps.
    x_axis = np.arange(-20, 20, 0.01)

     # Plotting Guassian distribution for mean = 0 and variance = 1
    mean = 0
    std = 1
    # plt.plot(x_axis, norm.pdf(x_axis, mean, std))
    # plt.savefig("distribution1.pdf", bbox_inches='tight')
    params = random.normal(loc=mean, scale=std, size=(1, K))
    print(params)

    i=0
    for N in N_list:
        print(K,N)

        # Plotting Guassian distribution for mean = 0 and variance = 4
        mean = 0
        std = 2
        # plt.plot(x_axis, norm.pdf(x_axis, mean, std))
        # plt.savefig("distribution2.pdf", bbox_inches='tight')
        x = random.normal(loc=mean, scale=std, size=(N, K))
        # print(x)

        # Plotting Guassian distribution for mean = 0 and variance = 0.1
        mean = 0
        std = math.sqrt(0.1)
        # plt.plot(x_axis, norm.pdf(x_axis, mean, std))
        # plt.savefig("distribution3.pdf", bbox_inches='tight')
        epsilon = random.normal(loc=mean, scale=std, size=(N, 1))
        # print(epsilon)
        # plt.show()

        # Generating Data Set
        # y=[epsilon[0][0]]*N
        y=epsilon
        # print(y)
        for n in range(N):
            # print(y[n])
            for k in range(K):
                y[n] = y[n] + x[n][k]*params[0][k]

        # print(y)

        # Calculate theta using OLS
        theta_best_values = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        normal_results = theta_best_values.reshape(-1)
        print(normal_results)


        # Calculate Ridge Regression
        I = np.identity(K)
        # print(I)
        # A = x.T.dot(x)+(lamda**2).dot(I)
        # for i in range(len(lamda)):
        #     theta_best_values = np.linalg.inv(x.T.dot(x)+(lamda[i]**2)*(I)).dot(x.T).dot(y)
        #     ridge_results = theta_best_values.reshape(-1)
        #     print(ridge_results)

        # Calculate L2 distance (Loss Function)
        l1 = 0
        l2 = 0
        for k in range(K):
            l1 += (normal_results[k]-params[0][k])**2
            l2 += (ridge_results[k]-params[0][k])**2
        loss1 = math.sqrt(l1)
        loss2 = math.sqrt(l2)

        print("Ordinary Regression Loss : ", loss1)
        print("Ridge Regression Loss : ", loss2)
        normal_loss[i] = loss1
        ridge_loss[i] = loss2
        i+=1

print(normal_loss)
print(ridge_loss)
# plt.plot(N_list,normal_loss)
# plt.xlabel("Smaple Size (N)")
# plt.ylabel("L2 Distance")
# plt.title("Ordinary Regression: L2 Norm vs N (K=3)")
# plt.savefig("K3OR.pdf", bbox_inches='tight')
# plt.show()