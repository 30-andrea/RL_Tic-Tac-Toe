import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from utils.pygame_draw import draw_lines, draw_figures
from env.env_properties import PHYSICAL_ATTRIBUTES
import time

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
        self.win_reward = win_r
        self.loss_reward = loss_r
        self.draw_reward = draw_r
        self.render_mode = render_mode
        self.info = {}
        self.reset()

        # pygame setup
        if self.render_mode == "human":
            self.pygame_init()
        
    def pygame_init(self):
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
        self.info = {
            'players': {
                1: {
                    'actions': []
                },
                2: {
                    'actions': []
                }
            }
        }
        self.pygame_init()
        return self.state

    def step_p1(self, action):
        """
            Doc String
        """
        terminated = truncated = False
        reward = 0
        if len(self.info['players'][1]['actions']) >= 9:
            truncated = True
            reward = self.draw_reward
            self.render(None, 1)
        else:
            # update to next state.
            # if the agent picks a square already selected, give a large -ve reward to train it to not do that again.
            if self.state[action] == 0:
                self.state[action] = 1
                self.info['players'][1]['actions'].append(action)
            else:
                reward = -100
                terminated = True
                return self.state.flatten(), reward, terminated or truncated, self.info

            # check if the game is over
            if self.gameEndCheck(player = 1):
                terminated = True
                reward = self.win_reward
            else:
                terminated = True
                reward = self.loss_reward
            self.render(action, 1)
        return self.state.flatten(), reward, terminated or truncated, self.info

    
    def step_p2(self, action):
        """
        Doc String
        """
        terminated = truncated = False
        reward = 0
        if len(self.info['players'][2]['actions']) >= 9:
            truncated = True
            self.render(None, 2)
        else:
            # update to next state.
            # if the agent picks a square already selected, give a large -ve reward to train it to not do that again.
            if self.state[action] == 0:
                self.state[action] = 2
                self.info['players'][2]['actions'].append(action)
            else:
                reward = -100
                terminated = True
                return self.state.flatten(), reward, terminated or truncated, self.info
            
            # check if the game is over
            if self.gameEndCheck(player = 2):
                terminated = True
                reward = self.win_reward
            else:
                terminated = True
                reward = self.loss_reward
            self.render(action, 2)
        return self.state.flatten(), reward, terminated or truncated, self.info

    def render(self, action, player, mode="human"):
        if action == None:
            return
        draw_lines(self.screen, PHYSICAL_ATTRIBUTES.LINE_COLOR, PHYSICAL_ATTRIBUTES.SQUARE_SIZE, PHYSICAL_ATTRIBUTES.WIDTH, PHYSICAL_ATTRIBUTES.HEIGHT, PHYSICAL_ATTRIBUTES.LINE_WIDTH)
        draw_figures(self.screen, action, player, self.board, PHYSICAL_ATTRIBUTES.CIRCLE_COLOR, PHYSICAL_ATTRIBUTES.SQUARE_SIZE, PHYSICAL_ATTRIBUTES.CIRCLE_RADIUS, PHYSICAL_ATTRIBUTES.CIRCLE_WIDTH, PHYSICAL_ATTRIBUTES.CROSS_COLOR, PHYSICAL_ATTRIBUTES.SPACE, PHYSICAL_ATTRIBUTES.CROSS_WIDTH)
        # for ii in range(3):
        #     for jj in range(3):
        #         if player == 1:
        #             pygame.draw.circle(self.screen, (239, 231, 200), (jj * 100 + 50, ii * 100 + 50), 40, 15)
        #         elif player == 2:
        #             pygame.draw.line(self.screen, (66, 66, 66), (jj * 100 + 15, ii * 100 + 85), (jj * 100 + 85, ii * 100 + 15), 25)
        #             pygame.draw.line(self.screen, (66, 66, 66), (jj * 100 + 15, ii * 100 + 15), (jj * 100 + 85, ii * 100 + 85), 25)
        time.sleep(3)
        pygame.display.update()
    
    def gameEndCheck(self, player):
        '''
        function to check if a move has caused the game to end i.e. win/ draw.
        '''
        rows, cols = self.board_size
        grid = self.state.reshape(-1, self.board_size[0])
        num_winning = rows
        for r in range(rows):
            for c in range(cols):
                value = grid[r][c]
                if value == player:

                    # left, top, right, bottom, top-left, top-right, bottom-right, bottom-left
                    check_ver_list = [0, -1, 0, 1, -1, -1, 1, 1]
                    check_hor_list = [-1, 0, 1, 0, -1, 1, 1, -1]

                    for i in range(len(check_ver_list)):
                        row_current = r
                        col_current = c

                        check_ver = check_ver_list[i]
                        check_hor = check_hor_list[i]

                        for line in range(num_winning - 1):
                            row_current = row_current + check_ver
                            col_current = col_current + check_hor

                            if row_current >= rows or col_current >= cols or row_current < 0 or col_current < 0:
                                break

                            value_current = grid[row_current][col_current]
                            if value_current != player:
                                break

                            if (line + 1) == (num_winning - 1):
                                return True

        return False
    
if __name__ == "__main__":
    env = gym.envs.make("TTT-v0")
