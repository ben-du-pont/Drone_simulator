import numpy as np

A = np.array([[1,0], [0,1], [0.3, 0.7]])
A_prime = np.array([[1,0], [0,1], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])

cond_A = np.linalg.cond(A)
cond_A_prime = np.linalg.cond(A_prime)
print("Condition number of A:", cond_A)
print("Condition number of A_prime:", cond_A_prime)


gdop_A = np.sqrt(np.trace(np.linalg.inv(A.T @ A)))
gdop_A_prime = np.sqrt(np.trace(np.linalg.inv(A_prime.T @ A_prime)))
print("Geometric Dilution of Precision of A:", gdop_A)
print("Geometric Dilution of Precision of A_prime:", gdop_A_prime)
