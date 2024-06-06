import numpy as np
import gymnasium as gym
import random
import time

env = '''Code to create obj of custom env we made'''
env.reset()
env.render()

eta = 0.7 # exploration exploitation               
discount_factor = 0.618               
epsilon = 1                  
max_epsilon = 1
min_epsilon = 0.01         
decay = 0.01

train_episodes = 2000    
test_episodes = 100          
max_steps = 100

Q_p1 = np.zeros((env.observation_space, env.action_space))
Q_p2 = np.zeros((env.observation_space, env.action_space))

training_rewards_p1 = []
training_rewards_p2 = []  
epsilons = []

for episode in range(train_episodes):
    #Reseting the environment each time as per requirement
    state = env.reset()    
    #Starting the tracker for the rewards
    total_training_rewards_p1, total_training_rewards_p2 = 0,0
    for step in range(100):
        exp_exp_tradeoff = random.uniform(0, 1) 
        
        if exp_exp_tradeoff > epsilon:
            action_p1 = np.argmax(Q_p1[state,:]) 
        else:
            action_p1 = random.randint(0, env.action_space-1) 
        new_state, reward, done, info = env.step_p1(action_p1)

        Q_p1[state, action_p1] = eta*Q_p1[state, action_p1]+ (1-eta)*(reward+discount_factor*np.max(Q_p1[new_state, :]))
        #Increasing our total reward and updating the state
        total_training_rewards_p1 += reward      

        if episode%50 == 0:
            env.render()
            time.sleep(5)

        if done:
            #code to display winner
            break

        exp_exp_tradeoff = random.uniform(0, 1) 
        
        if exp_exp_tradeoff > epsilon:
            action_p2 = np.argmax(Q_p1[new_state,:]) 
        else:
            action_p2 = random.randint(0, env.action_space-1) 
        new_state, reward, done, info = env.step_p2(action_p2)

        Q_p2[state, action_p2] = eta*Q_p2[state, action_p2]+ (1-eta)*(reward+discount_factor*np.max(Q_p2[new_state, :]))
        #Increasing our total reward and updating the state
        total_training_rewards_p2 += reward    

        state = new_state    
        
        if episode%50 == 0:
            env.render()
            time.sleep(5)

        if done:
            #code to display winner
            break

    epsilon = min_epsilon+(max_epsilon-min_epsilon)*np.exp(-decay*episode)
    training_rewards_p1.append(total_training_rewards_p1)
    training_rewards_p2.append(total_training_rewards_p2)
    epsilons.append(epsilon)
