import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
from collections import deque
import virl
import sys
import matplotlib.pyplot as plt
#import pylab as pl


class DQN():
    # DQN Agent
    def __init__(self):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 4
        self.action_dim = 4
        self.create_Q_network()
        self.create_training_method()


        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim,128])
        b1 = self.bias_variable([128])
        W2 = self.weight_variable([128,128])
        b2 = self.bias_variable([128])
        W3 = self.weight_variable([128,128])
        b3 = self.bias_variable([128])
        W4 = self.weight_variable([128,self.action_dim])
        b4 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer1 = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        h_layer2 = tf.nn.relu(tf.matmul(h_layer1,W2) + b2)
        h_layer3 = tf.nn.relu(tf.matmul(h_layer2,W3) + b3)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer3,W4) + b4

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward/2,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        #rint (self.time_step)
        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

    def action(self,state):
         action = np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0])
         Q = np.max((self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            }))[0])
         return action, Q

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)



def state_gen(state):
        return state/6e8

if __name__ == "__main__":

    # Parameters for DQN
    GAMMA = 0.5  # discount factor for target Q
    INITIAL_EPSILON = 1  # starting value of epsilon
    FINAL_EPSILON = 0.1  # final value of epsilon
    REPLAY_SIZE = 10000  # experience replay buffer size
    BATCH_SIZE = 32  # size of minibatch
    s = np.array([0., 0, 0, 0])  # epidemic state
    c = 1.  # infection rate damping
    rewardss = []

    for i in range(1):

        try:
            episodes = int(sys.argv[1])
        except Exception as e:
            episodes = int(5)
        # is used to train several episodes,
        # random.seed(1)
        env = virl.Epidemic(noisy=False, problem_id=i)
        # define env class
        action_dim = env.action_space.n
        # state_dim = env.observation_space.n
        agent = DQN()

        # agent class in inrelated to env
        all_reward = []
        # all reward record list1
        for episode in range(episodes + 1):
            # circle episodes times
            state = env.reset()
            # restart env
            agent.epsilon -= 0.7/ episodes
            # 0.1-1, choose one number between this range
            reward_perepisode = []
            # clear the report record of individual episodes list2
            while True:
                # endless loop
                action = agent.egreedy_action(state_gen(state))
                # action
                state_, reward, done, _ = env.step(action)
                # action to env，return s',r,done
                agent.perceive(state_gen(state), action, reward, state_gen(state_), done)
                # agent learn，the Q table is updated iteratively
                state = state_
                # update state
                reward_perepisode.append(reward)
                # list2 record every step's reward
                if done:
                    break
                    # if finish one episode，break out of the loop
            reward_episode_avgr = sum(reward_perepisode) / len(reward_perepisode)
            # caculater episode's avg reward
            print('episode: {}, avg reward: {}'.format(episode, reward_episode_avgr))
            # output
            all_reward.append(reward_episode_avgr)

            #Plot
plt.plot(all_reward)
plt.xlabel('episodes')
plt.ylabel('avg_reward')
plt.show()
env.close()






