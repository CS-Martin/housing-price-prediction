#lin_reg.py

import numpy as np
import matplotlib.pyplot as plt

def load(file_name):
    data = np.loadtxt(open(file_name, "rb"), dtype="float", delimiter=",")
    data = data[: , [4, 7] ]
    
    return data
    
def hyp(theta, x):
    return theta[0] + theta[1] * x

def cost(theta, data):
    sum = 0.0
    
    m = data.shape[0]
    
    for i in range(0, m):
        sum += (hyp(theta, data[i, 0]) - data[i, 1]) ** 2
    
    sum = sum / (2.0 * m)
    
    return sum
    
def der0(theta, data):  
    sum = 0.0
    
    m = data.shape[0]
    
    for i in range(0, m):
        sum += hyp(theta, data[i, 0]) - data[i, 1]
    
    sum = sum / (1.0 * m)
    
    return sum  

def der1(theta, data):  
    sum = 0.0
    
    m = data.shape[0]
    
    for i in range(0, m):
        sum += ( hyp(theta, data[i, 0]) - data[i, 1] ) * data[i, 0] 
    
    sum = sum / (1.0 * m)
    
    return sum    
   
data = load("data/realestate.csv")

theta = [0, 0]
ntheta = [0, 0]
alpha = 0.002    
m, n = data.shape
c = []
loops = 10000

x = range(0, m)
learned_hypothesis = [hyp(theta, z) for z in x]

plt.plot(x, learned_hypothesis, label="Initial Hypothesis")

print("The unoptimized cost is: " + str(cost(theta, data)))

for i in range(0, loops):
    ntheta[0] = theta[0] - alpha * der0(theta, data)
    ntheta[1] = theta[1] - alpha * der1(theta, data)

    theta[0] = ntheta[0]
    theta[1] = ntheta[1]
    
    tc = cost(theta, data)
    c.append(tc)

    print("The optimized cost is: " + str(tc))

plt.plot(range(0, loops), c)


y = np.sort(data, axis=1)
print(y)

learned_hypothesis = [hyp(theta, z) for z in x] 

plt.scatter(y[:, 0], y[:, 1], label="Age of House vs Price")
plt.plot(x, learned_hypothesis, label="Learned Hypothesis")

print("The optimized cost is: " + str(tc))
print(theta)
plt.legend()

plt.show()