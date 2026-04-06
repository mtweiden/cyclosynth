from numpy import array
from sympy import sqrt
from sympy import symbols
from pprint import pprint
import numpy as np
import sympy as sp

r2 = 1/sqrt(2)
sigma_u = array([
    [1, r2, 0, -r2, 0, 0, 0, 0],
    [0, r2, 1, r2, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, r2, 0, -r2],
    [0, 0, 0, 0, 0, r2, 1, r2],
])

a1, b1, c1, d1, a2, b2, c2, d2 = symbols('a1 b1 c1 d1 a2 b2 c2 d2')
x = array([a1, b1, c1, d1, a2, b2, c2, d2])
M = sigma_u.transpose() @ sigma_u
prod = x @ M @ x
prod = prod.simplify()

sigma = array([
    [1, r2, 0, -r2, 0, 0, 0, 0],
    [0, r2, 1, r2, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, r2, 0, -r2],
    [0, 0, 0, 0, 0, r2, 1, r2],
    [1, -r2, 0, r2, 0, 0, 0, 0],
    [0, -r2, 1, -r2, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, -r2, 0, r2],
    [0, 0, 0, 0, 0, -r2, 1, -r2],
])
M = sigma_u.transpose() @ sigma_u
C = M - np.eye(8)
print(np.linalg.eig(C.astype(float))[0])
print(np.linalg.norm(C.astype(float), 2))
print(C)

sigma_bul = array([
    [1, -r2, 0, r2, 0, 0, 0, 0],
    [0, -r2, 1, -r2, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, -r2, 0, r2],
    [0, 0, 0, 0, 0, -r2, 1, -r2],
])

M_bul = sigma_bul.transpose() @ sigma_bul
M = sigma_u @ sigma_u.transpose()
# print(M)

# print(np.linalg.norm(sigma_bul.astype(float), 2))
# print(np.linalg.svd(sigma_bul.astype(float))[1])

# 
# a1, b1, c1, d1, a2, b2, c2, d2 = symbols('a1 b1 c1 d1 a2 b2 c2 d2')
# x = array([a1, b1, c1, d1, a2, b2, c2, d2])
# M = sigma_bul.transpose() @ sigma_bul
# prod = x @ M @ x
# prod = prod.simplify()
# pprint(M)
# 
# sigma_u = array([
#     [1, r2, 0, -r2, 0, 0, 0, 0],
#     [0, r2, 1, r2, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, r2, 0, -r2],
#     [0, 0, 0, 0, 0, r2, 1, r2],
# ])
# 
