import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import math


K_list = [3]
N_list = [5,10,15,20,25,30]
# N_list = [10,20,30,40,50,60,70,80,90,100]
# N_list = [20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,200]
normal_loss = [0]*len(N_list)

for K in K_list:

    # Plot between -10 and 10 with .001 steps.
    x_axis = np.arange(-20, 20, 0.01)

     # Plotting Guassian distribution for mean = 0 and variance = 1
    mean = 0
    std = 1
    # plt.plot(x_axis, norm.pdf(x_axis, mean, std))
    # plt.title("Gaussian Distribution (Mean=0, variance=1)")
    # plt.savefig("distribution1.pdf", bbox_inches='tight')
    # plt.show()
    params = random.normal(loc=mean, scale=std, size=(1, K))
    print(params)

    i=0
    for N in N_list:
        print(K,N)

        # Plotting Guassian distribution for mean = 0 and variance = 4
        mean = 0
        std = 2
        # plt.plot(x_axis, norm.pdf(x_axis, mean, std))
        # plt.title("Gaussian Distribution (Mean=0, variance=4)")
        # plt.savefig("distribution2.pdf", bbox_inches='tight')
        # plt.show()
        x = random.normal(loc=mean, scale=std, size=(N, K))
        # print(x)

        # Plotting Guassian distribution for mean = 0 and variance = 0.1
        mean = 0
        std = math.sqrt(0.1)
        # plt.plot(x_axis, norm.pdf(x_axis, mean, std))
        # plt.title("Gaussian Distribution (Mean=0, variance=0.1)")

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

        # Calculate L2 distance (Loss Function)
        l1 = 0
        for k in range(K):
            l1 += (normal_results[k]-params[0][k])**2
        loss1 = l1

        print("Ordinary Regression Loss : ", loss1)
        normal_loss[i] = loss1
        i+=1

print(normal_loss)
plt.plot(N_list,normal_loss)
plt.xlabel("Smaple Size (N)")
plt.ylabel("L2 Distance")
plt.title("Ordinary Regression: L2 Norm vs N (K="+str(K)+")")
plt.savefig("K"+str(K)+"OR.pdf", bbox_inches='tight')
plt.show()