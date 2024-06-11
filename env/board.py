import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from utils.pygame_draw import draw_lines, draw_figures
from env.env_properties import PHYSICAL_ATTRIBUTES

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
        self.action_space = spaces.Discrete(self.num_actions)
        self.win_reward = win_r
        self.loss_reward = loss_r
        self.draw_reward = draw_r
        self.render_mode = render_mode
        self.reset()

        # pygame setup
        if self.render_mode == "human":
            self.pygame_init()
        
    def pygme_init(self):
        pygame.init()
        pygame.display.set_caption('Tic Tac Toe')
        self.screen = pygame.display.set_mode((PHYSICAL_ATTRIBUTES.WIDTH, PHYSICAL_ATTRIBUTES.HEIGHT))
        self.screen.fill(PHYSICAL_ATTRIBUTES.BG_COLOR)
        self.board = [[0 for _ in range(PHYSICAL_ATTRIBUTES.BOARD_COLS)] for _ in range(PHYSICAL_ATTRIBUTES.BOARD_ROWS)]

    def reset(self):
        """
            New empty board. info stores the status of the palyers' moves.
        """
        self.state = np.zeros(self.num_actions)
        self.done = False
        self.pygme_init()
        return self.state

    def step_p1(self, action):
        """
            Doc String
        """
        terminated = truncated = False
        reward = 0

        # update to next state.
        # if the agent picks a square already selected, give a large -ve reward to train it to not do that again.
        if self.state[action] == 0:
            self.state[action] = 1
        else:
            reward = -100
            terminated = True
        
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

        # update to next state.
        # if the agent picks a square already selected, give a large -ve reward to train it to not do that again.
        if self.state[action] == 0:
            self.state[action] = 1
        else:
            reward = -100
            terminated = True
        
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

    def render(self, action, player, mode="human"):
        draw_lines(self.screen, PHYSICAL_ATTRIBUTES.LINE_COLOR, PHYSICAL_ATTRIBUTES.SQUARE_SIZE, PHYSICAL_ATTRIBUTES.WIDTH, PHYSICAL_ATTRIBUTES.HEIGHT, PHYSICAL_ATTRIBUTES.LINE_WIDTH)
        draw_figures(self.screen, action, player, self.board, PHYSICAL_ATTRIBUTES.CIRCLE_COLOR, PHYSICAL_ATTRIBUTES.SQUARE_SIZE, PHYSICAL_ATTRIBUTES.CIRCLE_RADIUS, PHYSICAL_ATTRIBUTES.CIRCLE_WIDTH, PHYSICAL_ATTRIBUTES.CROSS_COLOR, PHYSICAL_ATTRIBUTES.SPACE, PHYSICAL_ATTRIBUTES.CROSS_WIDTH)
        # for ii in range(3):
        #     for jj in range(3):
        #         if player == 1:
        #             pygame.draw.circle(self.screen, (239, 231, 200), (jj * 100 + 50, ii * 100 + 50), 40, 15)
        #         elif player == 2:
        #             pygame.draw.line(self.screen, (66, 66, 66), (jj * 100 + 15, ii * 100 + 85), (jj * 100 + 85, ii * 100 + 15), 25)
        #             pygame.draw.line(self.screen, (66, 66, 66), (jj * 100 + 15, ii * 100 + 15), (jj * 100 + 85, ii * 100 + 85), 25)

        pygame.display.update()
    
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
