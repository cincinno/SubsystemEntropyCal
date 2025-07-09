'''
This file is used to calculate the data of BB code in Fig.2(b) and Fig.2(c) for ArXiv:2501.06407.
'''

import numpy as np
import copy as cp
import random
from tqdm import tqdm
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
from SubCal import code

def S(n):
    res=np.zeros([n,n])
    for i in range(n):
        res[i,(i+1)%n]=1
    return res
def repetition_code(n):
    """
    Parity check matrix of a repetition code with length n.
    """
    row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
    data = np.ones(2*n, dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)))
def toric_code_x_stabilisers(L):
    """
    Sparse check matrix for the X stabilisers of a toric code with
    lattice size L, constructed as the hypergraph product of
    two repetition codes.
    """
    Hr = repetition_code(L)
    H = hstack(
            [kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)],
            dtype=np.uint8
        )
    H.data = H.data % 2
    H.eliminate_zeros()
    return csc_matrix(H)

def BBLDPC(l,m,a1,a2,a3,b1,b2,b3):
    x=np.kron(S(l),np.eye(m))
    y=np.kron(np.eye(l),S(m))
    A=np.linalg.matrix_power(x,a1)+np.linalg.matrix_power(y,a2)+np.linalg.matrix_power(y,a3)
    B=np.linalg.matrix_power(y,b1)+np.linalg.matrix_power(x,b2)+np.linalg.matrix_power(x,b3)
    C=np.hstack([B.T,A.T])
    return C


l=[6,15,9,12,12,30,21]
m=[6,3,6,6,12,6,18]
a1=[3,9,3,3,3,9,3]
a2=[1,1,1,1,2,1,10]
a3=[2,2,2,2,7,2,17]
b1=[3,0,3,3,3,3,5]
b2=[1,2,1,1,1,25,3]
b3=[2,7,2,2,2,26,19]


for i in range(7):
    print(f'now cal i={i+1}')
    C=BBLDPC(l[i],m[i],a1[i],a2[i],a3[i],b1[i],b2[i],b3[i])
    bitnum=l[i]*m[i]
    diff=int(0.02*bitnum+1)
    if diff==0:diff+=1
    range1=np.arange(0,2*bitnum+1,diff)
    res=np.zeros(range1.shape[0])
    for idx,k in enumerate(tqdm(range1)):
        sum=0
        for _ in range(1000):
            new=code(C)
            choice=random.sample([i for i in range(2*bitnum)],k=k)
            se=set(choice)
            sum+=new.entropy_cal_neo(se)[0]
        res[idx]=sum/1000
    np.save(f'res_code_{i+1}_all.npy',res)

