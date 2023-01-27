import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import math

x = random.normal(loc=1, scale=2, size=(1, 3))

print(x)

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-20, 20, 0.01)
  
# Plotting Guassian distribution for mean = 0 and variance = 1
mean = 0
std = math.sqrt(1)
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.savefig("distribution1.pdf", bbox_inches='tight')

# Plotting Guassian distribution for mean = 0 and variance = 4
mean = 0
std = 2
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.savefig("distribution2.pdf", bbox_inches='tight')

# Plotting Guassian distribution for mean = 0 and variance = 0.1
mean = 0
std = math.sqrt(0.1)
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.savefig("distribution3.pdf", bbox_inches='tight')
plt.show()
