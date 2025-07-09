'''
This file is used to calculate the data of toric code in Fig.2(b) and Fig.2(c) for ArXiv:2501.06407.
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

for d in [5,7,12,13,17,23,29]:
    a=toric_code_x_stabilisers(d)
    bitnum=2*d**2
    diff=int(0.07*bitnum+1)
    if diff==0:diff+=1
    range1=np.arange(0,2*d**2,diff)
    res=np.zeros(range1.shape[0])
    for idx,k in enumerate(tqdm(range1)):
        sum=0
        for _ in range(1000):
            new=code(a.toarray())
            choice=random.sample([i for i in range(2*d**2)],k=k)
            #print(choice)
            se=set(choice)
            #print(se)
            sum+=new.entropy_cal_neo(se)[0]
        res[idx]=sum/1000

    np.save(f'toricd={d}_all.npy',res)