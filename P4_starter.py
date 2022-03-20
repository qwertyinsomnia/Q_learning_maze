import tkinter as tk
import numpy as np
import pandas as pd
import time
import os
from collections import defaultdict
import pickle

UNIT = 80
ENABLE_VISULIZE = 0
ENABLE_Learning = 1

class Maze(tk.Tk, object):
    def __init__(self, actions, current_location, end_location, maze):
        super(Maze, self).__init__()
        self.actions = actions
        self.n_actions = len(self.actions)
        self.current_location = current_location
        self.start_location = self.current_location
        self.end_location = end_location
        self.title('maze')
        self.height = len(maze)
        self.width = len(maze[0])
        self.maze = maze
        self.geometry('{0}x{1}'.format(self.height * UNIT, self.width * UNIT))
        self.canvas = tk.Canvas(self, bg='white', height=self.height * UNIT, width=self.width * UNIT)
        self.bind("<Key>", self.manual_move)
        self.build_maze()
        self.canvas.pack()
        self.check_allowed_states()

    def check_allowed_states(self):
        self.allowed_states = defaultdict(list)
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i][j] == 1:
                    continue
                if j > 0  and self.maze[i][j - 1] != 1:
                    self.allowed_states[(i, j)].append("L")
                if j < self.width - 1  and self.maze[i][j + 1] != 1:
                    self.allowed_states[(i, j)].append("R")
                if i > 0  and self.maze[i - 1][j] != 1:
                    self.allowed_states[(i, j)].append("U")
                if i < self.height - 1  and self.maze[i + 1][j] != 1:
                    self.allowed_states[(i, j)].append("D")
        
    
    def build_maze(self):
        for c in range(0, self.width * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.width * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.height * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.height * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        for i in range(self.height):
            for j in range(self.width):
                if (j, i) == self.current_location:
                    self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='red')
                    continue
                if (j, i) == self.end_location:
                    self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='blue')
                    continue
                if self.maze[j][i] == 1:
                    self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='black')
                else:
                    self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='white')
        

    def reset(self):
        self.current_location = self.start_location
        self.build_maze()
        return self.current_location
    
    def valid_moves(self, action):
        y = self.current_location[0]
        x = self.current_location[1]
        if y >= self.height or y < 0 or x >= self.width or x < 0 or self.maze[y][x] == 1:
            self.current_location =  (self.current_location[0] - self.actions[action][0], self.current_location[1] - self.actions[action][1])
        else:
            self.current_location = (y, x)

    def step(self, action):
        self.current_location =  (self.current_location[0] + self.actions[action][0], self.current_location[1] + self.actions[action][1])
        self.valid_moves(action)
        self.build_maze()

        s_ = self.current_location

        if s_ == self.end_location:
            reward = 1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def manual_move(self, event):
        if event.char == "w":
            self.step("U")
        elif event.char == "s":
            self.step("D")
        elif event.char == "a":
            self.step("L")
        elif event.char == "d":
            self.step("R")

    def q_table_visulize(self, q_table):
        for item in q_table.index:
            if item == "terminal":
                continue
            y = int(item[1:item.index(",")])
            x = int(item[item.index(",") + 1:-1])
            up_length = np.log(1 + q_table.loc[item]["U"])
            down_length = np.log(1 + q_table.loc[item]["D"])
            left_length = np.log(1 + q_table.loc[item]["L"])
            right_length = np.log(1 + q_table.loc[item]["R"])
            temp = [up_length, down_length, left_length, right_length]
            dirct = temp.index(max(temp))
            fill_color = ['black', 'black', 'black', 'black']
            fill_color[dirct] = 'red'

            if up_length > 0.3:
                up_length = UNIT / 2
            elif up_length > 0:
                up_length = UNIT / 4

            if down_length > 0.3:
                down_length = UNIT / 2
            elif down_length > 0:
                down_length = UNIT / 4
                
            if left_length > 0.3:
                left_length = UNIT / 2
            elif left_length > 0:
                left_length = UNIT / 4
                
            if right_length > 0.3:
                right_length = UNIT / 2
            elif right_length > 0:
                right_length = UNIT / 4
            
            self.canvas.create_line(x * UNIT + UNIT / 2, y * UNIT + UNIT / 2, x * UNIT + UNIT / 2, 
                                    y * UNIT + UNIT / 2 - up_length, fill=fill_color[0], arrow = "last")
            self.canvas.create_line(x * UNIT + UNIT / 2, y * UNIT + UNIT / 2, x * UNIT + UNIT / 2,
                                    y * UNIT + UNIT / 2 + down_length, fill=fill_color[1], arrow = "last")
            self.canvas.create_line(x * UNIT + UNIT / 2, y * UNIT + UNIT / 2, x * UNIT + UNIT / 2 - left_length,
                                    y * UNIT + UNIT / 2, fill=fill_color[2], arrow = "last")
            self.canvas.create_line(x * UNIT + UNIT / 2, y * UNIT + UNIT / 2, x * UNIT + UNIT / 2 + right_length,
                                    y * UNIT + UNIT / 2, fill=fill_color[3], arrow = "last")
        
        self.update


    def print_maze(self, maze):
        print("█", end='')
        for col in maze[0]:
            print("█", end='')
        print("█")
        for row in maze:
            print("█", end='')
            for col in row:
                if (col == 0):
                    print(' ', end='')
                elif (col == 1):
                    print('█', end ='')
                elif (col == 2):
                    print('O', end='')
            print("█")
        print("█", end='')
        for col in maze[0]:
            print("█", end='')
        print("█")


class QLearningTable:
    def __init__(self, actions, allowed_states, learning_rate=0.01, reward_decay=0.9, e_greedy=0.75):
        self.actions = list(actions.keys())
        self.allowed_states = allowed_states
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        if os.path.isfile("Qtable.json"):
            self.q_table = pd.read_json("Qtable.json")
        else:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exist(str(state))
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[str(state), :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action and explore
            # action = np.random.choice(self.actions)
            action = np.random.choice(self.allowed_states[state])
        return action

    def learn(self, s, a, r, s_):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[(s), a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state))
        
def learning_func():
    step_list = []
    for episode in range(500):
        state = env.reset()
        step_cnt = 0
        while True:
            step_cnt += 1
            action = RL.choose_action(state)
            state_, reward, done = env.step(action)
            if ENABLE_VISULIZE:
                env.update()
                time.sleep(0.1)
            RL.learn(state, action, reward, state_)
            state = state_
            if step_cnt > 1000:
                print("step exceeding... you have lost...")
                break
            if done:
                print(env.current_location, "total step:  ", step_cnt)
                break
        
        step_list.append(step_cnt)
        with open('step.txt', 'wb') as text:
            pickle.dump(step_list, text)
    
    RL.q_table.to_json("Qtable.json")
    RL.q_table.to_csv("Qtable.csv")
    env.q_table_visulize(RL.q_table)
    print("Game Over")

# Set up important variables
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
current_location = (0, 0)
end_location = (7, 7)
maze = [[0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0]]
# maze = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
#         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
#         [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
#         [0, 0, 1, 1, 0, 0, 1, 0, 1, 0]]


# Construct allowed states
# allowed_states = {}

env = Maze(actions, current_location, end_location, maze)
allowed_states = env.allowed_states
# print(allowed_states)
RL = QLearningTable(actions, allowed_states)
if ENABLE_Learning:
    env.after(100, learning_func)

# The main loop where you navigate to the end
# while (current_location != end_location):
#     print_maze(maze)
#     break # Temporary break to avoid infinite loop

# print_maze(maze)
# print("Congratulations! You made it to the end of the maze!")

tk.mainloop()