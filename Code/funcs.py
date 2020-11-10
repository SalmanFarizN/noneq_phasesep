import numpy as np
from numba import jit,objmode

@jit(nopython=True)
def lj_force(pos,lj_params,shape_params,pbc_params):

    force=np.zeros((shape_params[0],shape_params[1]))
    r=np.zeros((shape_params[1]))




    for i in range(shape_params[0]):
        for j in range(i+1,shape_params[0]):
            
            #Periodic Boundary Conditions
            for k in range(shape_params[1]):
                r[k]=pos[i,k]-pos[j,k]
                if r[k] > pbc_params[1]:
                    r[k]=r[k]-pbc_params[0]
                if r[k] <= -pbc_params[1]:
                    r[k]=r[k]+pbc_params[0]
                
                
            #rnorm=np.linalg.norm(r)
            
            rnorm = 0
            for el in range(shape_params[1]):
                rnorm = rnorm+r[el]**2
            
            rnorm = np.sqrt(rnorm)
            
            
            if rnorm < lj_params[2]:
                #rnorm=np.linalg.norm(r)
                part=(lj_params[1]/rnorm)**6
                f=(-24*lj_params[0]/rnorm)*(2*part**2-part)
         
                for k in range(shape_params[1]):
                    force[i,k]=force[i,k]+(r[k]/rnorm)*f
                    force[j,k]=force[j,k]-(r[k]/rnorm)*f

    return(force)




def posini(shape_params,pbc_params):
    pos1=np.zeros((shape_params[0],shape_params[1]))
    sep=np.zeros((shape_params[0]))
    dimsep=np.zeros((shape_params[1]))
    
    for i in range(shape_params[0]):
        print(i)
        if i == 0:
            for j in range(shape_params[1]):
                pos1[i,j]=np.random.uniform(0.0,pbc_params[0])
        else:
            for j in range(shape_params[1]):
                pos1[i,j]=np.random.uniform(0.0,pbc_params[0])
            
            for k in range(i):
                if i!=k:
                    for l in range(shape_params[1]):
                        dimsep[l]=pos1[i,l]-pos1[k,l]
                        if dimsep[l] > pbc_params[1]:
                            dimsep[l]=dimsep[l]-pbc_params[0]
                        if dimsep[l] <= -pbc_params[1]:
                            dimsep[l]=dimsep[l]+pbc_params[0]
                        
                    #sep[k]=np.linalg.norm(dimsep)
                    sep[k]=0
                    for el in range(shape_params[1]):
                        sep[k]=sep[k]+dimsep[el]**2
                    
                    sep[k]=np.sqrt(sep[k])
                
            while(any(q < 0.6 and q!=0.0 for q in sep)):
                for j in range(shape_params[1]):
                    pos1[i,j]=np.random.uniform(0.0,pbc_params[0])
                for k in range(i):
                    if i!=k:
                        for l in range(shape_params[1]):
                            dimsep[l]=np.mod(pos1[i,l]-pos1[k,l],pbc_params[0])
                            if dimsep[l] > pbc_params[1]:
                                dimsep[l]=dimsep[l]-pbc_params[0]
                                
                            if dimsep[l] <= -pbc_params[1]:
                                dimsep[l]=dimsep[l]+pbc_params[0]
                            
                        #sep[k]=np.linalg.norm(dimsep)
                        sep[k]=0
                        for el in range(shape_params[1]):
                            sep[k]=sep[k]+dimsep[el]**2
                        sep[k]=np.sqrt(sep[k])
    
    return(pos1)
                        