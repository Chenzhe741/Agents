import numpy as np
import pickle, os, sys, itertools
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import virl
from collections import namedtuple
import pandas as pd

class TabularPolicy():
    """
    Q(s,a) tabular. # too gernerate the policy function

    """
 
    def __init__(self, env, gamma=0.9, Q_size=5000, lr=0.01): #initialization method of the class 
        self.n_states = env.observation_space.shape[0]    #return the shape of the array of observation space
        self.n_actions = env.action_space.n        #the action space, 0:none 1:full lockdown 2:track&trace 3:social distancing
        self.gamma = gamma               #gama is the discounted return/cumulative future reward, need to modify
        self.Q_size = Q_size      #for training size
        self.lr = lr            #learning rate
        self.Q = np.zeros([Q_size, self.n_states + self.n_actions])
        self.Q_nums = 0
    def get_index(self, s):
        last_distance = 10000
        index = 0
        for i in range(self.Q_size):
            distance = np.linalg.norm(self.Q[i][:self.n_states] - s)
            if distance < last_distance:
                index = i
                last_distance = distance
            if distance <= 0:
                return i
        return index
    
    #According to the input observation value, predict the output action value
    def predict(self, s, a=None):
        s_index = self.get_index(s)
        if a==None:
            return self.Q[s_index][self.n_states:]

        return self.Q[s_index][self.n_states:][a]
    
    
#store s,a,r into Q and update Q
    def store(self, s, a, r):
        index = self.Q_nums % self.Q_size
        self.Q[index, self.n_states + a] = r
        self.Q_nums += 1

    # method of updating Q-table 
    def update(self, s, a, target):
        s_index = self.get_index(s)
        r_index = self.n_states + a
        self.Q[s_index,r_index] += self.lr*(target - self.Q[s_index,r_index])

from utils import (
    exec_policy,
    get_fig,
    plt,
    q_learning,
    get_env,
    logging,
)

#training  and testing,save the results into file
def train(Q_size=2000, n_episodes=50, discount_factor=0.95, epsilon=0.05):
    print('policy_searh tabular training...')
    env = virl.Epidemic()
    tabular = TabularPolicy(env, Q_size)
    results_dir = './results/Policy_search_tabular'
    pkl_file = os.path.join(results_dir, 'ps_tabular_size{}_episodes{}.pkl'.format(Q_size, n_episodes))
    if os.path.exists(pkl_file):
        return
    q_learning(env, tabular, n_episodes, discount_factor, epsilon)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(pkl_file, 'wb')as f:
        pickle.dump(tabular, f)


 # number of training episodes
def get_eval(Q_size=2000, n_episodes=50, is_train=False, savefig=False):
    print('policy_searh tabular evaluating...')
    results_dir = './results/Policy_search_tabular'
    results_file = os.path.join(results_dir, 'policy_search_tabular.csv')
    log_file = os.path.join(results_dir, 'tabular.log')
    logger = logging(log_file)
    pkl_file = os.path.join(results_dir, 'ps_tabular_size{}_episodes{}.pkl'.format(Q_size, n_episodes))
    if not os.path.exists(pkl_file):
        train(Q_size, n_episodes)
    
    if os.path.exists(results_file) and not is_train and not savefig:
        results = pd.read_csv(results_file)
        results = results.sort_values(by=['noisy', 'problem_id'])
        return results
    else:
        if os.path.exists(results_file):
            os.remove(results_file)
        if os.path.exists(log_file):
            os.remove(log_file)
        with open(pkl_file, 'rb')as f:
           tabular = pickle.load(f)

    results = pd.DataFrame([], columns=['problem_id', 'noisy', 'Total_rewards', 'avg_reward_per_action'])
    for problem_id, noisy, env in get_env():
        states, rewards, actions = exec_policy(env, tabular,)
        result = {'problem_id':problem_id, 'noisy':noisy, 
                  'Total_rewards':sum(rewards),
                  'avg_reward_per_action':sum(rewards)/len(actions)}
        results = results.append(pd.DataFrame(result, index=[0]), ignore_index=0)
        logger(result)
        logger(actions)
        if savefig:
            get_fig(states, rewards)
            pic_name = os.path.join(results_dir, 'problem_id={} noisy={}.jpg'.format(problem_id, noisy))
            plt.savefig(dpi=300, fname=pic_name)
            plt.close()
        env.close()
    results = results.sort_values(by=['noisy', 'problem_id'])
    results.to_csv(results_file, index=0)
    return results

if __name__ == '__main__':
    train(n_episodes=10)
    print(get_eval(is_train=True, n_episodes=10))
    
