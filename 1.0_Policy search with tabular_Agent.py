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

env = virl.Epidemic(stochastic=False, noisy=True)
#env = virl.Epidemic(stochastic=False, noisy=False, problem_id=9)
"""
Args:
    stochastic (bool): Is the infection rate sampled from some distribution at the beginning of each episode (default: False)?
    
    noisy (bool): Is the state a noisy estimate of the true state (default: False)?
  
    problem_id (int): Deterministic parameterization of the epidemic (default: 0).
   
    
"""
at = 0 #action

states = []    #states numpy.ndarray
rewards = []   #reward numpy.float64
done = False   #finished or not

s = env.reset()   #reset the states
print('reset',s)

states.append(s)  #add new state
while not done:
    
    sp=float(s[0]) # suspectible people
    ip=float(s[1]) # infectious people
    qp=float(s[2]) # quarantined people
    rp=float(s[3]) # recovered people
    
    Sum=sp+ip+qp+rp #total people
    
    Ratio_sp=sp/Sum #ratio
    Ratio_ip=ip/Sum
    Ratio_qp=qp/Sum
    Ratio_rp=rp/Sum
    
    print('Infected r',float(Ratio_ip))
    print('Total',Sum)
    print('ip is',float(s[0]))
    print('ip is',float(s[1]),type(float(s[1])))
    print('qp is',float(s[2]))
    print('rp is',float(s[3]))
    
    

    
    if  0.01<Ratio_ip<0.1:
     at=2
    elif 0.1<Ratio_ip<0.3:
     at=1
    elif 0.3<Ratio_ip<0.5:
     at=3 
    
    s, r, done, i = env.step(action=at) # deterministic agent
    #s, r, done, i = env.step(action=np.random.choice(env.action_space.n)) #random agent
    states.append(s)
    
    print('====added s is',s)
    
    rewards.append(r) #add  new
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
states = np.array(states)

print('==== states is',states,'states')

print('==== r is',r)
print('==== i is',i)


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
