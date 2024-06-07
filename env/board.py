import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from tabulate import tabulate

class TicTacToe(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 16}

    def __init__(self, render_mode = None, size = 3, win_r = 10, loss_r = -10, draw_r = 1) -> None:
        """
            Defines the parameters of the board, the state space and action space.
            State Space: All possible variations of the sizexsize boards. each state is one particular board
            Action space: choosing to fill a particular cell in the board (for a 3x3 board, at every turn there are 9 options, though not all are valid)
            Reward: +10 for a win, -10 for a loss, +1 for a draw for player 1, and a win/loss reward for player 2

            Note: State space must be a list of discrete and valid states of the board, i.e. no cell in a state can hold 2 colors etc..
        """
        self.board_size = (size, size)
        self.num_actions = size**2
        state_space = spaces.Discrete(3**(size**2))
        self.observation_space = '''Code''' # calculate the discrete combinations of all valid tic tac toe boards. Need to cacluate for any board size n.
        self.action_space = spaces.Discrete(self.num_actions)
        self.win_reward = win_r
        self.loss_reward = loss_r
        self.draw_reward = draw_r
        self.reset()
    
    def reset(self):
        """
            New empty board. info stores the status of the palyers' moves.
        """
        self.state = np.zeros(self.board_size)
        self.info = {"players": {1: {"actions": []}, 2: {"actions": []}}}
        return self.state.flatten(), self.info

    def step_p1(self, action):
        """
            Doc String
        """
        terminated = truncated = False
        reward = 0

        # convert action to row and col indices
        row, col = divmod(action, self.board_size[1])
    
        # check if the action is valid 
        if self.state[row, col] != 0:
            raise ValueError("Invalid action: Cell already filled.")

        # update board state
        self.state[row, col] = 1
        self.info["players"][1]["actions"].append(action)
        
        # check if the game is over
        if self.gameEndCheck(player = 1):
            terminated = True
            reward = self.win_reward
        else:
            if len(self.info['players'][1]['actions']) >= 9:
                truncated = True
                reward = self.draw_reward
            else:
                terminated = True
                reward = self.loss_reward
        self.render()
        return self.state.flatten(), reward, terminated or truncated, self.info

    
    def step_p2(self, action):
        """
        Doc String
        """
        terminated = truncated = False
        reward = 0

        # convert action to row and col indices
        row, col = divmod(action, self.board_size[1])
    
        # check if the action is valid 
        if self.state[row, col] != 0:
            raise ValueError("Invalid action: Cell already filled.")

        # update board state
        self.state[row, col] = 2
        self.info["players"][2]["actions"].append(action)
        
        # check if the game is over
        if self.gameEndCheck(player = 2):
            terminated = True
            reward = self.win_reward
        else:
            if len(self.info['players'][1]['actions']) >= 9:
                truncated = True
                reward = self.draw_reward
            else:
                terminated = True
                reward = self.loss_reward
        self.render()
        return self.state.flatten(), reward, terminated or truncated, self.info

    def render(self, mode="human") -> None:
        """render the board

        The following charachters are used to represent the fields,
            '-' no stone
            'O' for player 1
            'X' for player 2

        example:
            ╒═══╤═══╤═══╕
            │ O │ - │ - │
            ├───┼───┼───┤
            │ - │ X │ - │
            ├───┼───┼───┤
            │ - │ - │ - │
            ╘═══╧═══╧═══╛
        """
        board = np.zeros((3, 3), dtype=str)
        for ii in range(3):
            for jj in range(3):
                if self.state[ii, jj] == 0:
                    board[ii, jj] = "-"
                elif self.state[ii, jj] == 1:
                    board[ii, jj] = "X"
                elif self.state[ii, jj] == 2:
                    board[ii, jj] = "O"

        if mode == "human":
            board = tabulate(board, tablefmt="fancy_grid")

        # instead of return, use pygame to view it.
        return board
    
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
    
if __name__ == "__main__":
    env = gym.envs.make("TTT-v0")
