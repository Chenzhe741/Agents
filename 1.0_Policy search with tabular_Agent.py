# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:55:59 2020

@author: ChenZhe
"""
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import os
os.chdir('..')
from matplotlib import pyplot as plt
import numpy as np

import virl

env = virl.Epidemic(stochastic=True, noisy=True)
#env = virl.Epidemic(stochastic=False, noisy=False, problem_id=9)
"""
Args:
    stochastic (bool): Is the infection rate sampled from some distribution at the beginning of each episode (default: False)?
    
    
    noisy (bool): Is the state a noisy estimate of the true state (default: False)?
    
    
    problem_id (int): Deterministic parameterization of the epidemic (default: 0).
   
    
"""

r=0
act = 0
states = []   
rewards = []  
done = False   

s = env.reset()   
print('Reset s',s)

states.append(s)  
while not done:
    
    sp=float(s[0])
    ip=float(s[1])
    qp=float(s[2])
    rp=float(s[3])
    
    Sum=sp+ip+qp+rp
    
    R_sp=sp/Sum
    R_ip=ip/Sum
    R_qp=qp/Sum
    R_rp=rp/Sum
    
    print('Total',Sum)
    
    print('Sup People',R_sp*100,'%')
    print('Ifc People',R_ip*100,'%')
    print('Qrt People',R_qp*100,'%')
    print('Rcv people',R_rp*100,'%')
    
    """    
    if R_ip<0.01 and R_sp<0.01:
     at=0
    elif R_ip<=0.01 and R_sp>=0.25:
     at=2
    elif R_ip<=0.01 and R_sp<=0.25:
     at=1
    elif 0.01<=R_ip<=0.1 and R_sp<=0.01:
     at=3 
    elif R_ip>0.3 and R_sp>0.3:
        at=1
    """    
    
    if R_ip<=0.01:
        if R_sp>R_rp>0.2:
            act=3
        elif R_sp<=R_rp<0.2:
            act=2
        elif R_sp<=R_ip:
            act=1
    else :
        act=2
       
    s, r, done, i = env.step(action=act) # deterministic agent
    #s, r, done, i = env.step(action=np.random.choice(env.action_space.n)) #random agent
    states.append(s)
    
     
    print('====added s is',s)
    
    rewards.append(r) #add  new
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
states = np.array(states)

#print('==== states is',states,'states',type(states))

print('==== r is',r,type(r))
print('==== i is',i,type(i))


for i in range(4):
    axes[0].plot(states[:,i], label=labels[i]);
axes[0].set_xlabel('weeks since start of epidemic')
axes[0].set_ylabel('State s(t)')
axes[0].legend()
axes[1].plot(rewards);
axes[1].set_title('Reward')
axes[1].set_xlabel('weeks since start of epidemic')
axes[1].set_ylabel('reward r(t)')

print('total reward', np.sum(rewards))
