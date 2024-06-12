import numpy as np
import gymnasium as gym
import random
import time
from env.board import TicTacToe
import pandas as pd
from tqdm import tqdm

def state_key(state):
    sum = 0
    for i in range(len(state)):
        sum += state[i]*(3**i)
    return int(sum)

def get_best_action(epsilon, Q, key):
    exp_exp_tradeoff = random.uniform(0, 1)  
    if exp_exp_tradeoff > epsilon:
        action = np.argmax(Q[key,:]) 
    else:
        action = random.randint(0, env.num_actions-1)
    return action

env = TicTacToe(render_mode="human")
env.reset()

eta = 0.7 # exploration exploitation               
discount_factor = 0.618               
epsilon = 1                  
max_epsilon = 1
min_epsilon = 0.01         
decay = 0.01

train_episodes = 2000    
test_episodes = 100          
max_steps = 100

Q_p1 = np.zeros((3**10 - 3, env.num_actions))
Q_p2 = np.zeros((3**10 - 3, env.num_actions))

training_rewards_p1 = []
training_rewards_p2 = []  
epsilons = []

for episode in tqdm(range(train_episodes)):
    #Reseting the environment each time as per requirement
    state = env.reset()    
    #Starting the tracker for the rewards
    total_training_rewards_p1, total_training_rewards_p2 = 0,0
    for step in range(100):
        s_key = state_key(state)

        # exp_exp_tradeoff = random.uniform(0, 1) 
        # if exp_exp_tradeoff > epsilon:
        #     action_p1 = np.argmax(Q_p1[s_key,:]) 
        # else:
        #     action_p1 = random.randint(0, env.num_actions-1) 
        action_p1 = get_best_action(epsilon, Q_p1, s_key)
        while action_p1 in env.info['filled']:
            action_p1 = get_best_action(epsilon, Q_p1, s_key)
        new_state, reward, done, info = env.step_p1(action_p1)
        if reward == env.win_reward:
            total_training_rewards_p2 += env.loss_reward
            Q_p2[s_key, :] = eta*Q_p2[s_key, :]+ (1-eta)*(env.loss_reward+discount_factor*np.max(Q_p2[s1_key, :]))
        if episode%50 == 0:
            time.sleep(1)
        else:
            time.sleep(0.1)
        
        s1_key = state_key(new_state)
        Q_p1[s_key, action_p1] = eta*Q_p1[s_key, action_p1]+ (1-eta)*(reward+discount_factor*np.max(Q_p1[s1_key, :]))
        #Increasing our total reward and updating the state
        total_training_rewards_p1 += reward     

        if done:
            #code to display winner
            break

        state = new_state
        s_key = state_key(state)
        #print(state)
        # exp_exp_tradeoff = random.uniform(0, 1) 
        
        # if exp_exp_tradeoff > epsilon:
        #     action_p2 = np.argmax(Q_p2[s_key,:]) 
        # else:
        #     action_p2 = random.randint(0, env.num_actions-1) 
        action_p2 = get_best_action(epsilon, Q_p2, s_key)
        while action_p2 in env.info['filled']:
            action_p2 = get_best_action(epsilon, Q_p2, s_key)
        new_state, reward, done, info = env.step_p2(action_p2)
        if reward == env.win_reward:
            total_training_rewards_p1 += env.loss_reward
            Q_p1[s_key, :] = eta*Q_p1[s_key, :]+ (1-eta)*(env.loss_reward+discount_factor*np.max(Q_p1[s1_key, :]))
        if episode%50 == 0:
            time.sleep(1)
        else:
            time.sleep(0.1)
        s1_key = state_key(new_state)
        Q_p2[s_key, action_p2] = eta*Q_p2[s_key, action_p2]+ (1-eta)*(reward+discount_factor*np.max(Q_p2[s1_key, :]))
        #Increasing our total reward and updating the state
        total_training_rewards_p2 += reward    

        state = new_state    

        if done:
            #code to display winner
            break

    epsilon = min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay*episode)
    training_rewards_p1.append(total_training_rewards_p1)
    training_rewards_p2.append(total_training_rewards_p2)
    epsilons.append(epsilon)

cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
data_p1 = pd.DataFrame(Q_p1, columns=cols)
data_p2 = pd.DataFrame(Q_p2, columns=cols)
data_p1.to_csv('player1.csv')
data_p2.to_csv('player2.csv')