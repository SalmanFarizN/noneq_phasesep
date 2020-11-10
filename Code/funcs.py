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