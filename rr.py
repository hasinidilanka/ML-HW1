import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import math


K = 3
N_list = [5,10,15,20,25,30]
lamda_list = [0.001, 0.01, 0.1, 1,2,10]
normal_loss = [0]*len(N_list)
# ridge_loss = [[0]*len(N_list)]*len(lamda_list)
ridge_loss = [[0 for i in range(len(N_list))] for j in range(len(lamda_list))]



i=0
for lamda in lamda_list:

    # Plot between -10 and 10 with .001 steps.
    x_axis = np.arange(-20, 20, 0.01)

     # Plotting Guassian distribution for mean = 0 and variance = 1
    mean = 0
    std = 1
    # plt.plot(x_axis, norm.pdf(x_axis, mean, std))
    # plt.savefig("distribution1.pdf", bbox_inches='tight')
    params = random.normal(loc=mean, scale=std, size=(1, K))
    print(params)

    j=0
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

      


        # Calculate Ridge Regression
        I = np.identity(K)
        # print(I)
        # A = x.T.dot(x)+(lamda**2).dot(I)
        theta_best_values = np.linalg.inv(x.T.dot(x)+(lamda**2)*(I)).dot(x.T).dot(y)
        ridge_results = theta_best_values.reshape(-1)
        print(ridge_results)

        # Calculate L2 distance (Loss Function)
        l2 = 0
        for k in range(K):
            l2 += (ridge_results[k]-params[0][k])**2
        loss2 = math.sqrt(l2)

        print("Ridge Regression Loss : ", loss2)
        ridge_loss[i][j] = loss2
        print(i,j)
        print(ridge_loss)
        j+=1
    i+=1

print(ridge_loss)
i=0
for loss in ridge_loss:
    print(loss)
    plt.plot(N_list,loss,label='lambda='+str(lamda_list[i]))
    i+=1
plt.xlabel("Smaple Size (N)")
plt.ylabel("L2 Distance")
plt.title("Ridge Regression: L2 Norm vs N (K=3)")
plt.savefig("K3RR.pdf", bbox_inches='tight')
plt.legend()
plt.show()