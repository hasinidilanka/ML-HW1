import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import math


K = 3
N = 10
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-20, 20, 0.01)
  
# Plotting Guassian distribution for mean = 0 and variance = 1
mean = 0
std = math.sqrt(1)
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.savefig("distribution1.pdf", bbox_inches='tight')
params = random.normal(loc=mean, scale=std, size=(1, K))
print(params)

# Plotting Guassian distribution for mean = 0 and variance = 4
mean = 0
std = 2
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.savefig("distribution2.pdf", bbox_inches='tight')
x = random.normal(loc=mean, scale=std, size=(N, K))
# print(x)

# Plotting Guassian distribution for mean = 0 and variance = 0.1
mean = 0
std = math.sqrt(0.1)
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.savefig("distribution3.pdf", bbox_inches='tight')
epsilon = random.normal(loc=mean, scale=std, size=(1, 1))
# print(epsilon)
# plt.show()

# Generating Data Set
y=[epsilon[0][0]]*N
# print(y)
for n in range(N):
    for k in range(K):
        y[n] = y[n] + x[n][k]*params[0][k]
        
# print(y)

# Calculate theta using OLS
theta_best_values = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
print(theta_best_values)
# for i in range(k):
#     print(i)
#     y = y+ params[0][i]*covariants[0][i]
# print(y)

# Calculate L2 distance (Loss Function)
l = x.dot(theta_best_values)-y
# print(l)
loss = l.T.dot(l)
print(loss)