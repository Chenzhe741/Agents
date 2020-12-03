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
    随机，是否从随机分布中抽样 默认否
    
    noisy (bool): Is the state a noisy estimate of the true state (default: False)?
    噪音，是否有真实状态的评估 默认否
    
    problem_id (int): Deterministic parameterization of the epidemic (default: 0).
    病例ID
    
"""
at = 0

states = []    #状态
rewards = []   #奖励
done = False   #是否完成

s = env.reset()   #重置
print('重置s',s)

states.append(s)  #重置
while not done:
    
    sp=float(s[0])
    ip=float(s[1])
    qp=float(s[2])
    rp=float(s[3])
    
    Sum=sp+ip+qp+rp
    
    Ratio_sp=sp/Sum
    Ratio_ip=ip/Sum
    Ratio_qp=qp/Sum
    Ratio_rp=rp/Sum
    
    print('感染比例',float(Ratio_ip))
    print('总人数',Sum)
    print('疑似感染人数',float(s[0]))
    print('感染人数',float(s[1]),type(float(s[1])))
    print('隔离人数',float(s[2]))
    print('康复人数',float(s[3]))
    
    

    
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