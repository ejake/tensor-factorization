import string
import random
import itertools
import numpy as np

global C

def lcs(X,Y):
    global C
    C = np.ones((len(X)+1,len(Y)+1))*-1
    lcs3(X,Y,len(X),len(Y))
    return C[len(X)-1,len(Y)-1], C
    

def lcs3(x,y,i,j):
    global C
    if C[i,j] == -1:
        if i == 0 or j == 0:
            C[i,j] = 0
        else:
            if x[i-1]==y[j-1]:
                C[i,j] = lcs3(x,y,i-1,j-1)+1
            else:
                C[i,j] = max(lcs3(x,y,i-1,j),lcs3(x,y,i,j-1))
                
    return C[i,j]

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def lcs2(x,y,i,j):
    if i == 0 or j == 0:
        return 0
    if x[i-1]==y[j-1]:
        return lcs2(x,y,i-1,j-1)+1
    else:
        return max(lcs2(x,y,i-1,j),lcs2(x,y,i,j-1))
    


def lcs1(X,Y):
    #lst = list(itertools.product([0, 1], repeat=len(X)))
    lgst = ''
    for i in range(1,len(X)):
        for j in list(itertools.combinations(X, i)):
            if verificar(j,Y) == 1:
                lgst = j
    return ''.join(lgst)

#Procedimiento verificar
def verificar(subx, Y):
    idx_subx = 0; tam_subx = 0;
    if len(subx)>len(Y): return 0
    for i in Y:
        if subx[idx_subx] == i:
            idx_subx+=1
        if idx_subx == len(subx):
            return 1
    else:
        return 0