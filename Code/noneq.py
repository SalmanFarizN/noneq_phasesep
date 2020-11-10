import numpy as np
import h5py
from numba import jit,objmode
import time
import os
import funcs

#%% Defining model parameters
nparticles=32
split=int(nparticles/2)
ndims=2
mass=1

gamma1=1
gamma2=1

rho=1

ls=0.5

step_size=1e-3  #dt
length=np.sqrt((nparticles*3.14*ls**2)/rho)
kBT1=1
kBT2=5

tpoints=int(1e7)




tD1=(ls**2*gamma1)/kBT1
tD2=(ls**2*gamma2)/kBT2




tm1=mass/gamma1
tm2=mass/gamma2




D1=kBT1/gamma1

D2=kBT2/gamma2

tD=tD2



velratio1=tD/tm1
velratio2=tD/tm2

fratio1=tD**2/(tD1*tm1)
fratio2=tD**2/(tD2*tm2)


randstd1=np.sqrt((2.0*tD1)/(tD*step_size))
randstd2=np.sqrt((2.0*tD2)/(tD*step_size))


hlength=length/2.0


epsilon = 1
sigma = 0.5
rc=0.5




def write(pos,vel,t):
    position[:,t,:]=pos
    velocity[:,t,:]=vel
    #potf[:,t,:]=potentialf
    #randf[:,t,:]=random_force
    
 


                   
    
    
                  
    

@jit(nopython=True) 
def run():
    
    
    
    #%% Defining the matrices 
    pos=np.zeros((nparticles,ndims))
    vel=np.zeros((nparticles,ndims))
    random_force = np.zeros((nparticles,ndims))
    total_force = np.zeros((nparticles,ndims))
    potentialf = np.zeros((nparticles,ndims))

  
    

    lj_params=(epsilon,sigma,rc)
    shape_params=(nparticles,ndims)
    pbc_params=(length,hlength)

    



    
    
    
    #Initialising particle positions so that they are spread out randomly and do not overlap
    with objmode(pos ='float64[:,:]'):
        pos=funcs.posini(shape_params,pbc_params)
        


        
      
        
        
                        
                        
    #Initialising velocities to 0
    vel[:,:]=0   


    #Computing the random forces         
    random_force[0:split,:]=np.random.normal(0.0,randstd1,(split,ndims))
    random_force[split:nparticles,:]=np.random.normal(0.0,randstd2,(nparticles-split,ndims))
    
    #Computing the LJ potential

    potentialf=funcs.lj_force(pos,lj_params,shape_params,pbc_params)
    
    
    

    
    #Computing the total forces 
    total_force[0:split,:]=np.multiply(velratio1,-vel[0:split,:])+np.multiply(fratio1,(random_force[0:split,:]-potentialf[0:split,:]))
        
    total_force[split:nparticles,:]=np.multiply(velratio2,-vel[split:nparticles,:])+np.multiply(fratio2,(random_force[split:nparticles,:]-potentialf[split:nparticles,:]))
            
    
    
    #Writing the data point into the h5 file
    with objmode():
        write(pos,vel,0)


    
    # Integrating the positions and velocities
    count=1
    
    
    for t in range(1,tpoints):
    
        potentialf[:,:]=0.0
      

        pos[:,:]=np.add(pos[:,:],np.multiply(step_size,vel[:,:]))
        vel[:,:]=np.add(vel[:,:],np.multiply(step_size,total_force[:,:]))
       

            
        random_force[0:split,:]=np.random.normal(0.0,randstd1,(split,ndims))
        random_force[split:nparticles,:]=np.random.normal(0.0,randstd2,(nparticles-split,ndims)) 
        
        
        
        potentialf=funcs.lj_force(pos,lj_params,shape_params,pbc_params)

        
        
        
     
        
        total_force[0:split,:]=np.multiply(velratio1,-vel[0:split,:])+np.multiply(fratio1,(random_force[0:split,:]-potentialf[0:split,:]))
        
        total_force[split:nparticles,:]=np.multiply(velratio2,-vel[split:nparticles,:])+np.multiply(fratio2,(random_force[split:nparticles,:]-potentialf[split:nparticles,:]))


                    
        #Periodic Boundary Conditions
        for i in range(nparticles):
            for j in range(ndims):
                if pos[i,j]<0.0:
                    pos[i,j]+=length
                if pos[i,j]>length:
                    pos[i,j]-=length
        
        #pos[pos<0] = pos[pos<0] + length
        #pos[pos>length] = pos[pos>length] -length
    
    
        if t%1000 == 0:
            print((t/tpoints)*100,"%") 
            with objmode():
                write(pos,vel,count)
            count=count+1
            
        
            
    
    
    
    
    




os.chdir('/net/storage/salmanfan96/Data/')
hf=h5py.File('noneqpbc.h5', 'w')
spoints=int(tpoints/1000)
position=hf.create_dataset('pos', (nparticles,spoints,ndims))
velocity=hf.create_dataset('vel', (nparticles,spoints,ndims))
#potf=hf.create_dataset('potential', (nparticles,spoints,ndims))
#randf=hf.create_dataset('fr',(nparticles,spoints,ndims))



start=time.time()
run()


end=time.time()
simtime=end-start
print(simtime/60,"mints.")


hf.close()
