import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class TicTacToe(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 16}

    def __init__(self, render_mode = None, size = 3, win_r = 10, loss_r = -10, draw_r = 1) -> None:
        '''
        Contructor. Defines the parameters of the board, the state space and action space.
        State Space: All possible variations of the sizexsize boards. each state is one particular board
        Action space: choosing to fill a particular cell in the board (for a 3x3 board, at every turn there are 9 options, though not all are valid)
        Reward: +10 for a win, -10 for a loss, +1 for a draw.

        Note: State space must be a list of discrete and valid states of the board, i.e. no cell in a state can hold 2 colors etc..
        '''
        self.board_size = (size, size)
        self.num_actions = size**2
        self.state_space = spaces.Discrete(3**(size**2))
        self.action_space = spaces.Discrete(self.num_actions)
        self.win_reward = win_r
        self.loss_reward = loss_r
        self.draw_reward = draw_r
        self.reset()
    
    def reset(self):
        '''
            New empty board. info stores the status of the palyers' moves.
        '''
        self.state = np.zeros(self.board_size)
        self.info = {"players": {1: {"actions": []}, 2: {"actions": []}}}
        return self.state.flatten(), self.info

    def step():
        '''
        Take turn.
        '''

    def render():
        '''
        render the pygame.
        '''
    
    def buttonClick():
        '''
        get input from the user who is one of the player and get s' (new state).
        '''
    
    def gameEndCheck(self, player):
        '''
        function to check if a move has caused the game to end i.e. win/ draw.
        '''
        done = False
        bool_matrix = self.state == player
        for ii in range(3):
            # check if three equal tokens are aligned (horizontal, verical or diagonal)
            if (
                # check columns
                np.sum(bool_matrix[:, ii]) == 3
                # check rows
                or np.sum(bool_matrix[ii, :]) == 3
                # check diagonal
                or np.sum([bool_matrix[0, 0], bool_matrix[1, 1], bool_matrix[2, 2]])
                == 3
                or np.sum([bool_matrix[0, 2], bool_matrix[1, 1], bool_matrix[2, 0]])
                == 3
            ):
                done = True
                break
        return done