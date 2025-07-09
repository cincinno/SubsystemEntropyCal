'''
This file is used to calculate the Fig.2(a) in ArXiv:2501.06407.
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

def expand(H,n,minus):
    res=set()
    nowlist=[0]
    for _ in range(n):
        qubit_list=set()
        stab_list=set()
        for i in nowlist:
            a=np.where(H[i,:])[0]
            for k in a:
                qubit_list.add(k)
            for j in qubit_list:
                b=np.where(H[:,j])[0]
                for k in b:
                    stab_list.add(k)
        nowlist=list(stab_list)
        if _==n-2:
            #print(minus,len(nowlist))
            for s_ in range(min(minus,len(nowlist))):
                nowlist.pop()
        res.update(qubit_list)
    return res

def disk_generate(w,h):
    res=[]
    for i in range(h):
        res.extend([j*6+i for j in range(w+1)])

    for i in range(w):
        res.extend([j+i*6+72 for j in range(h+1)])
    return res


C=BBLDPC(21,18,3,10,17,5,3,19)
numqubit=[]
entropy=[]
for i in [5,6]:
    for j in tqdm(range(30)):
        #print(j)
        qubits=expand(C,i+1,j)
        new=code(C)
        entropy.append(new.entropy_cal_neo(qubits)[0])
        numqubit.append(len(qubits))
npy=np.zeros([2,len(numqubit)])
npy[0,:]=np.array(numqubit)
npy[1,:]=np.array(entropy)
np.save('BB_expand.npy',npy)


numqubit=[]
entropy=[]
a=toric_code_x_stabilisers(20)
a=a.toarray()
for i in range(6):
    for j in tqdm(range(30)):
        qubits=expand(a,i+1,j)
        new=code(a)
        entropy.append(new.entropy_cal_neo(qubits)[0])
        numqubit.append(len(qubits))
npy=np.zeros([2,len(numqubit)])
npy[0,:]=np.array(numqubit)
npy[1,:]=np.array(entropy)
np.save('toric_expand.npy',npy)