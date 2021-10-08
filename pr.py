# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:30:39 2021

@author: RKorkin
"""

import numpy as np

def is_prime(x):
    if x != int(x):
        return False
    if x < 2:
        return False
    for i in range(2, int(np.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True

def gen_rand_prime(max_num = 200):
    while True:
        s = np.random.randint(max_num) + 4
        if is_prime(s):
            return s

Trials = 1000000
best_len = 2
P_best = [2, 3]
for case in range(Trials):
    s1, s2 = gen_rand_prime(), gen_rand_prime()
    P = [s1, s2]
    while True:
        P.append(P[-2] + 2 * P[-1])
        if not is_prime(P[-1]):
            break
    if len(P) > best_len:
        best_len = len(P)
        best_P = P
print(best_len, best_P)