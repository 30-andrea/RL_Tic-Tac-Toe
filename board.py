import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class TicTacToe(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 16}

    def __init__(self, render_mode = None, size = 3) -> None:
        '''
        Contructor. Defines the parameters of the board, the state space and action space.
        State Space: All possible variations of the sizexsize boards. each state is one particular board
        Action space: choosing to fill a particular cell in the board (for a 3x3 board, at every turn there are 9 options, though not all are valid)
        Reward: +10 for a win, -10 for a loss, +1 for a draw.

        Note: State space must be a list of discrete and valid states of the board, i.e. no cell in a state can hold 2 colors etc..
        '''
    
    def reset():
        '''
        Emptying the board after a game is over
        '''

    def step():
        '''
        Take turn
        '''

    def render():
        '''
        render the pygame
        '''
    
    def buttonClick():
        '''
        get input from the user who is one of the player and get s' (new state)
        '''
    
    def gameEndCheck():
        '''
        function to check if a move has caused the game to end i.e. win/ draw.
        '''