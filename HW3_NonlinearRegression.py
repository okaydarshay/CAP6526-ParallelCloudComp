#Darshay Blount
#COP6526 - Homework 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing Input data
data = pd.read_csv('/Users/darshayblount/Documents/1 MSDA/Parallel Cloud/20K_Datapoints.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
a = 0
b = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 10000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = a*X*X + b*X + c  # The current predicted value of Y
    D_a = (-2/n) * sum(X*X * (Y - Y_pred))  # Derivative wrt a
    D_b = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt b
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    a = a - L * D_a  # Update a
    b = b - L * D_b  # Update b
    c = c - L * D_c  # Update c

print (a, b, c)

# Making predictions
Y_pred = a*X*X + b*X + c

plt.scatter(X, Y)
plt.scatter(X, Y_pred , color='red') # predicted
plt.show()

###1 Multiprocessing
from multiprocessing import Pool, Process
import multiprocessing as mp
import time

def grad_desc():
    for i in range(epochs):
        Y_pred = a*X*X + b*X + c
        D_a = (-2/n) * sum(X*X * (Y - Y_pred))
        D_b = (-2/n) * sum(X * (Y - Y_pred))
        D_c = (-2/n) * sum(Y - Y_pred)
        a = a - L * D_a
        b = b - L *D_b
        c = c - L * D_C

if __name__ == '__main__':
    initialt = time.time()
    actions = []
    for i in range(0,2):#2cores, 1 processor
        proc = mp.Process(target = grad_desc, args = (0, 0, 0, X[i], Y[i], .0001, 10000)) #learning rate and epoch
        actions.append(proc)
        proc.start()
        
    for action in actions:
        action.join()


print('The total runtime for multiprocessing is {} seconds'.format(time.time() - initialt))
 
###2 Multithreading
import threading
if __name__ == '__main__':
    initial_thread = time.time()
    actions = []
    for i in range(0,2):#2cores, 1 processor
        proc_t = threading.Thread(target = grad_desc, args = (0, 0, 0, X[i], Y[i], .0001, 10000)) #learning rate and epoch
        actions.append(proc_t)
        proc_t.start()
        
    for action in actions:
        action.join()

print('The total runtime for multithreading is {} seconds'.format(time.time() - initial_thread))

###3 MPI4Py

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#print('My rank is ',rank) #rank 0, size 1

df = data.to_numpy() #df to numpy array
startt = time.time()

for i in range(epochs): 
    Y_pred = a*X*X + b*X + c  # The current predicted value of Y
    D_a = (-2/n) * sum(X*X * (Y - Y_pred))  # Derivative wrt a
    D_b = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt b
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    a = a - L * D_a  # Update a
    b = b - L * D_b  # Update b
    c = c - L * D_c  # Update c

if rank == 0:
    df = data.to_numpy()
else:
    df = None
comm.Bcast(df, root=0)

print('The total runtime for MPI is {} seconds'.format(time.time() - startt))


    
    