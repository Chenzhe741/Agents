from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


import os
os.chdir('..')
from matplotlib import pyplot as plt
import numpy as np
import virl

#set environment

env = virl.Epidemic(stochastic=False, noisy= False, problem_id=(np.random.choice(10)))

states = []
rewards = []
done = False

s = env.reset()
states.append(s)

#set qlearnmethod

while not done:

    s, r, done, i = env.step(action=np.random.choice(env.action_space.n))
    states.append(s)
    rewards.append(r)
    print(rewards)


fig, axes = plt.subplots(1, 2, figsize=(20, 8))
labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
states = np.array(states)
for i in range(4):
    axes[0].plot(states[:,i], label=labels[i]);
axes[0].set_xlabel('weeks since start of epidemic')
axes[0].set_ylabel('State s(t)')
axes[0].legend()
axes[1].plot(rewards);
axes[1].set_title('Reward')
axes[1].set_xlabel('weeks since start of epidemic')
axes[1].set_ylabel('reward r(t)')

plt.show()

print('total reward', np.sum(rewards))



