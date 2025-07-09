'''
This file is used to provide tools to calculate the entanglement entropy in ArXiv:2501.06407. 
'''

import numpy as np
import copy as cp

def Gauss_Matrix(M,row,col):
    shape=M.shape
    for i in range(shape[0]):
        if i !=row:
            #print(i,col,M[i,col])
            if M[i,col]==1:
                M[i,:]=(M[i,:]+M[row,:])%2
                #print(M)
    return M 

def linear_independence(H,constrain):
    Hrow=H.shape[0]
    Hcol=H.shape[1]
    test=H
    independent_q=[]
    for i in range(Hcol):
        now_one=-1
        for j in range(Hrow):
            if test[j,i]==1 and (not (np.any(test[j,:i]==1))):
                now_one=j#第j行的是1
                break
        if now_one!=-1:#找到了1
            independent_q.append((i,now_one))
            for j in range(Hrow):       
                if j!=now_one and test[j,i]==1:
                    test[j,:]=(test[j,:]+test[now_one,:])%2
                    constrain[j]=(constrain[j]+constrain[now_one])%2
    return test,independent_q,constrain

def rankcal(test):
    Hrow=test.shape[0]
    Hcol=test.shape[1]
    rank=0
    for i in range(Hcol):
        now_one=-1
        for j in range(Hrow):
            if test[j,i]==1 and (not (np.any(test[j,:i]==1))):
                now_one=j#第j行的是1
                break
        if now_one!=-1:#找到了1
            rank+=1
            for j in range(Hrow):       
                if j!=now_one and test[j,i]==1:
                    test[j,:]=(test[j,:]+test[now_one,:])%2
    return rank 

def Gauss(test0):
    test=cp.deepcopy(test0)
    Hrow=test.shape[0]
    Hcol=test.shape[1]
    rank=0
    record_mat=np.eye(Hrow)
    for i in range(Hcol):
        now_one=-1
        for j in range(Hrow):
            if test[j,i]==1 and (not (np.any(test[j,:i]==1))):
                now_one=j#第j行的是1
                break
        if now_one!=-1:#找到了1
            rank+=1
            for j in range(Hrow):       
                if j!=now_one and test[j,i]==1:
                    test[j,:]=(test[j,:]+test[now_one,:])%2
                    record_mat[j,:]=(record_mat[j,:]+record_mat[now_one,:])%2
    return test,record_mat


def iterbin(n):
    if n==0:
        yield np.zeros(1)
        return 
    for i in range(2**n):
        s=bin(i)
        res=np.zeros(n)
        for j in range(-1,-len(s)+1,-1):
            res[j]=int(s[j])
        yield res

def codeword_cal(H,cons):
    M,l,constrain=linear_independence(H,cons)
    #print(M,l,constrain)
    #计算自由的比特
    shape=M.shape
    independent_q={i for i in range(shape[1])}
    dependent_q=[]
    for i in l:
        independent_q.remove(i[0])
        dependent_q.append(i[0])
    independent_q=list(independent_q)
    independent_n=len(independent_q)
    #计算结果
    res=[]
    for acti_q in iterbin(independent_n):
        res_one=np.zeros(shape[1])#创建一个解，之后再把数填进去
        for i in range(independent_n):#先填独立的
            res_one[independent_q[i]]=acti_q[i]
        #再计算非独立的
        for j in l:
            res_one[j[0]]=int((res_one@M[j[1],:]+constrain[j[1]])%2)
        res.append(res_one)
    return res

def bin2dec(array):
    l=array.shape[0]
    res=0
    for i in range(l-1,-1,-1):
        res+=array[i]*2**(l-1-i)
    return res

class code:

    def __init__(self,H) -> None:
        shape=H.shape
        self.qubits=shape[1]
        self.stab_num=shape[0]
        self.H_ori=H
        self.H=cp.deepcopy(H)
        self.activate_q=[i for i in range(self.qubits)]

    def partition_AB(self,A_zone):
        self.A_zone=A_zone
        B_zone=set(self.activate_q)-A_zone
        self.B_zone=B_zone

        self.A_stab=[]
        self.B_stab=[]
        self.bound_stab=[]
        H=self.H
        for i in range(H.shape[0]):
            in_A=0
            in_B=0
            for j in A_zone:
                idy=self.activate_q.index(j)
                if H[i,idy]==1:
                    in_A=1
                    break
            for j in B_zone:
                idy=self.activate_q.index(j)
                if H[i,idy]==1:
                    in_B=1
                    break
            if in_A==1 and in_B==0:
                self.A_stab.append(i)
            elif in_A==0 and in_B==1:
                self.B_stab.append(i)
            else:
                self.bound_stab.append(i)


    def delete_qb(self,qid,stab_id):

        H=self.H
        old_activate_q=self.activate_q
        #print(self.activate_q.index(qid),stab_id)
        H=Gauss_Matrix(H,stab_id,self.activate_q.index(qid))
        #print(H)
        new_actvate_q=cp.deepcopy(self.activate_q)
        new_actvate_q.pop(new_actvate_q.index(qid))

        new_H=np.zeros([H.shape[0]-1,H.shape[1]-1])
        stab_list=[]
        for i in range(H.shape[0]):
            if i !=stab_id: stab_list.append(i)

        for idx,i in enumerate(new_actvate_q):
            for jdx,j in enumerate(stab_list):
                new_H[jdx,idx]=H[j,self.activate_q.index(i)]
        #print(new_H)
        self.H=new_H
        self.activate_q=new_actvate_q
        return qid,H[stab_id,:],old_activate_q
    
    def delete_edge(self,A_zone):
        self.partition_AB(A_zone)
        if not (self.A_stab or self.B_stab):
            process=self.process_mid_delete()
            #print(process[1],self.H)
            self.H=(process[1]@self.H)%2
            print(f"A_zone:{A_zone}\nH:{self.H}\nactivate_q:{self.activate_q}")
            self.partition_AB(A_zone)
            print(f"A_stab:{self.A_stab}\nB_stab:{self.B_stab}\nbound_stab:{self.bound_stab}")
        self.deleted_q=[]
        while self.A_stab or self.B_stab:
            H=self.H
            if self.A_stab:
                delete_stab=list(self.A_stab)[0]
                for idx,i in enumerate(H[delete_stab,:]):
                    if i==1:
                        delete_qubit=self.activate_q[idx]
                        break
            elif self.B_stab:
                delete_stab=list(self.B_stab)[0]
                for idx,i in enumerate(H[delete_stab,:]):
                    if i==1:
                        delete_qubit=self.activate_q[idx]
                        break
            delete_info=self.delete_qb(delete_qubit,delete_stab)
            self.deleted_q.append(delete_info)
            A_zone= A_zone & set(self.activate_q)
            self.partition_AB(A_zone)
            if not (self.A_stab or self.B_stab):
                process=self.process_mid_delete()
                #print(process[1],self.H)
                self.H=(process[1]@self.H)%2
                #print(f"A_zone:{A_zone}\nH:{self.H}\nactivate_q:{self.activate_q}")
                self.partition_AB(A_zone)
                #print(f"A_stab:{self.A_stab}\nB_stab:{self.B_stab}\nbound_stab:{self.bound_stab}")



    def delete_q(self):
        self.A_remove=[]
        self.B_remove=[]
        H=self.H
        for i in self.A_stab:
            for j in self.A_zone:
                if H[i,j]==1 and (j in self.activate_q):
                    self.activate_q.remove(j)
                    self.A_remove.append((i,j))
                    H=Gauss_Matrix(H,i,j)
                    break
        for i in self.B_stab:
            for j in self.B_zone:
                if H[i,j]==1 and (j in self.activate_q) :
                    self.activate_q.remove(j)
                    self.B_remove.append((i,j))
                    H=Gauss_Matrix(H,i,j)
                    break
        self.H=H
        self.A_act_q=self.A_zone.intersection(self.activate_q)
        self.B_act_q=self.B_zone.intersection(self.activate_q)

    def cal_boundary_codeword(self):
        activate_A=list(self.A_act_q)
        activate_B=list(self.B_act_q)
        bound_stab=list(self.bound_stab)
        A_boundmatrix=np.zeros([len(bound_stab),len(activate_A)])
        B_boundmatrix=np.zeros([len(bound_stab),len(activate_B)])
        #构造AB矩阵
        for i in range(len(bound_stab)):
            for j in range(len(activate_A)):
                A_boundmatrix[i,j]=self.H[bound_stab[i],activate_A[j]]
        for i in range(len(bound_stab)):
            for j in range(len(activate_B)):
                B_boundmatrix[i,j]=self.H[bound_stab[i],activate_B[j]]

        #检查可能的约束并且直接计算boundary上的A子系统值
        constrains=set()
        res=[]
        for i in iterbin(len(activate_B)):
            constrain=(B_boundmatrix@i)%2
            #print(constrain)
            c_hash=bin2dec(constrain)#计算哈希值排除掉相同的约束
            
            if c_hash not in constrains:
                res_one=codeword_cal(A_boundmatrix,constrain)
                res.append(res_one)
                constrains.add(c_hash)
        print(constrains)

        #计算A系统剩下的值
        res_A=[]
        for ii in res:
            res_oo=[]
            for i in ii:
                res_o=np.zeros(self.qubits)#先把对应的值放进去
                for idx,j in enumerate(activate_A):
                    #print(i)
                    res_o[j]=i[idx]


                for j in self.A_remove:
                    res_o[j[1]]=(self.H[j[0],:]@res_o)%2 
                res_temp=np.zeros(len(self.A_zone))
                A_zone=list(self.A_zone)
                for idx,j in enumerate(A_zone):
                    res_temp[idx]=res_o[j]
                res_oo.append(res_temp)
            res_A.append(res_oo)
        return res_A,A_zone
        
    def entropy_cal_old(self):
        self.delete_q()
        activate_A=list(self.A_act_q)
        activate_B=list(self.B_act_q)
        bound_stab=list(self.bound_stab)
        A_boundmatrix=np.zeros([len(bound_stab),len(activate_A)])
        B_boundmatrix=np.zeros([len(bound_stab),len(activate_B)])
        #构造AB矩阵
        for i in range(len(bound_stab)):
            for j in range(len(activate_A)):
                A_boundmatrix[i,j]=self.H[bound_stab[i],activate_A[j]]
        for i in range(len(bound_stab)):
            for j in range(len(activate_B)):
                B_boundmatrix[i,j]=self.H[bound_stab[i],activate_B[j]]

        rank_a=rankcal(A_boundmatrix)
        rank_b=rankcal(B_boundmatrix)
        return min(rank_a,rank_b),rank_a,rank_b

    def process_mid_delete(self):
        activate_A=list(self.A_zone)
        activate_B=list(self.B_zone)
        bound_stab=list(self.bound_stab)
        A_boundmatrix=np.zeros([len(bound_stab),len(activate_A)])
        B_boundmatrix=np.zeros([len(bound_stab),len(activate_B)])
        #构造AB矩阵
        for i in range(len(bound_stab)):
            for idx,j in enumerate(activate_A):
                A_boundmatrix[i,idx]=self.H[bound_stab[i],self.activate_q.index(j)]
        for i in range(len(bound_stab)):
            for idx,j in enumerate(activate_B):
                B_boundmatrix[i,idx]=self.H[bound_stab[i],self.activate_q.index(j)]

        #print(A_boundmatrix,B_boundmatrix)
        process_A=Gauss(A_boundmatrix)
        process_B=Gauss(B_boundmatrix)
        rank_a=rankcal(A_boundmatrix)
        rank_b=rankcal(B_boundmatrix)
        #print(f"ra:{rank_a},rb:{rank_b}")
        if rank_a<=rank_b:
            return process_A
        else:
            return process_B


    def entropy_cal(self):
        activate_A=list(self.A_zone)
        activate_B=list(self.B_zone)
        bound_stab=list(self.bound_stab)
        A_boundmatrix=np.zeros([len(bound_stab),len(activate_A)])
        B_boundmatrix=np.zeros([len(bound_stab),len(activate_B)])
        #构造AB矩阵
        for i in range(len(bound_stab)):
            for idx,j in enumerate(activate_A):
                A_boundmatrix[i,idx]=self.H[bound_stab[i],self.activate_q.index(j)]
        for i in range(len(bound_stab)):
            for idx,j in enumerate(activate_B):
                B_boundmatrix[i,idx]=self.H[bound_stab[i],self.activate_q.index(j)]
        #print(f"act_A:{activate_A}\nA_mat:{A_boundmatrix}\nB_mat:{B_boundmatrix}")
        rank_a=rankcal(A_boundmatrix)
        rank_b=rankcal(B_boundmatrix)
        assert rank_a==rank_b,"rank not equal!"
        return rank_a
    

    def entropy_cal_neo(self,Azone):
        self.partition_AB(Azone)
        activate_A=list(self.A_zone)
        activate_B=list(self.B_zone)
        rowlen=self.H.shape[0]
        A_boundmatrix=np.zeros([rowlen,len(activate_A)])
        B_boundmatrix=np.zeros([rowlen,len(activate_B)])
        #构造AB矩阵
        for i in range(rowlen):
            for idx,j in enumerate(activate_A):
                A_boundmatrix[i,idx]=self.H[i,self.activate_q.index(j)]
        for i in range(rowlen):
            for idx,j in enumerate(activate_B):
                B_boundmatrix[i,idx]=self.H[i,self.activate_q.index(j)]
        rank_a=rankcal(A_boundmatrix)
        rank_b=rankcal(B_boundmatrix)
        rank_H=rankcal(cp.deepcopy(self.H))
        return rank_a+rank_b-rank_H,rank_a,rank_b,rank_H       
