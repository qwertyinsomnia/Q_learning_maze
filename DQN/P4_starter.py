import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import deque
import torch
import torch.nn as nn
import random

UNIT = 60
ENABLE_LEARNING = 1
TRAIN_TO_TEST_SETUP = 0
BATCH_SIZE = 32
TARGET_REPLACE_ITER = 20   # iterations when Q target net update
MEMORY_CAPACITY = 200      # size of memory


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
        self.grid = self.maze
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
                    self.allowed_states[(i, j)].append([False, False, False, False])
                if i > 0  and self.maze[i - 1][j] != 1:
                    self.allowed_states[(i, j)].append(True)
                else:
                    self.allowed_states[(i, j)].append(False)
                if i < self.height - 1  and self.maze[i + 1][j] != 1:
                    self.allowed_states[(i, j)].append(True)
                else:
                    self.allowed_states[(i, j)].append(False)
                if j > 0  and self.maze[i][j - 1] != 1:
                    self.allowed_states[(i, j)].append(True)
                else:
                    self.allowed_states[(i, j)].append(False)
                if j < self.width - 1  and self.maze[i][j + 1] != 1:
                    self.allowed_states[(i, j)].append(True)
                else:
                    self.allowed_states[(i, j)].append(False)
        
    
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
        self.grid[self.current_location[0]][self.current_location[1]] = 2
        self.grid[self.end_location[0]][self.end_location[1]] = 3
        return self.grid
        # return self.current_location
    
    def valid_moves(self, action):
        y = self.current_location[0]
        x = self.current_location[1]
        if y >= self.height or y < 0 or x >= self.width or x < 0 or self.maze[y][x] == 1:
            self.current_location =  (self.current_location[0] - self.actions[action][0], self.current_location[1] - self.actions[action][1])
            return 1
        else:
            self.current_location = (y, x)
            return 0

    def step(self, action):
        self.current_location =  (self.current_location[0] + self.actions[action][0], self.current_location[1] + self.actions[action][1])
        valid_flag = self.valid_moves(action)
        self.build_maze()

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[j][i] == 2:
                    self.grid[j][i] = 0
        self.grid[self.current_location[0]][self.current_location[1]] = 2
        self.grid[self.end_location[0]][self.end_location[1]] = 3
        s_ = self.grid

        if self.current_location == self.end_location:
            reward = 20
            done = True
            # s_ = 'terminal'
        else:
            if valid_flag == 0:
                reward = -1
            else:
                reward = -5
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




class Net(nn.Module):
    def __init__(self, num_input):
        super(Net, self).__init__()
        self.f0 = nn.Linear(num_input, 25)
        self.f1 = nn.Linear(25, 16)
        self.f2 = nn.Linear(16, 4)
        self.net2 = True

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.f0(x)
        x = nn.functional.relu(x)
        x = self.f1(x)
        x = nn.functional.relu(x)
        x = self.f2(x)
        return x

class DQN(object):
    def __init__(self,num_input):
        self.epsilon = 0.5
        self.gamma = 0.8
        self.learning_rate = 0.001
        self.learn_step_counter = 0
        self.memory_counter = 0
        
        self.memory = list(np.zeros((MEMORY_CAPACITY, 4)))
        self.eval_net = Net(num_input)
        self.target_net = Net(num_input)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        # print(state)
        if np.random.uniform() < self.epsilon:
            action = self.choose_best_action(state)
        else:
            action = np.array([np.random.randint(0, 4)])
        return action
        
    def choose_best_action(self, state):
        actions_value = self.eval_net.forward(state)
        action = torch.max(actions_value, 1)[1].data.numpy()
        return action

    def remember_memory(self, s, a, r, s_):
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = [s, a, r, s_]
        self.memory_counter += 1

    def replay_experience(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_state_ = []
        for i in sample_index:
            # print(self.memory[i])
            batch_state.append(self.memory[i][0])
            batch_action.append(np.array(self.memory[i][1], dtype=np.int32))
            batch_reward.append(np.array([self.memory[i][2]], dtype=np.int32))
            batch_state_.append(self.memory[i][3])
        batch_state = torch.FloatTensor(np.array(batch_state))
        batch_action = torch.LongTensor(np.array(batch_action))
        batch_reward = torch.FloatTensor(np.array(batch_reward))
        batch_state_ = torch.FloatTensor(np.array(batch_state_))
        
        # training
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_state_).detach()
        q_target = batch_reward + self.gamma * torch.unsqueeze(q_next.max(1)[0], 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def training_func(episodes=300, times=256):
    alpha = 0.9
    actions_list = ['U', 'D', 'L', 'R']
    time_visualize_list = []
    dict_a = {0:'up',1:'down',2:'left',3:'right'}

    finishe_step_cnt = 25
    success_timer_cnt = 0
    step_cnt = 0
    good_test_cnt = 0

    for episode in range(episodes):
        times = 0
        s = env.reset()
        s = np.divide(s, 3)
        a_lib = []
        if episode % TARGET_REPLACE_ITER == 0:
            dql_agent.epsilon = 1 - (1 - dql_agent.epsilon) * alpha
        while True:
            times += 1
            a = dql_agent.choose_action(s)
        
            a_lib.append(dict_a[int(a)])
            s_, r, done = env.step(actions_list[int(a)])
            s_ = np.divide(s_, 3)
            
            dql_agent.remember_memory(s, a, r, s_)
            if dql_agent.memory_counter > MEMORY_CAPACITY:
                dql_agent.replay_experience()

            if done:
                print('epoch', episode, r, '成功')
                print(a_lib)
                time_visualize_list.append(times)
                success_timer_cnt += 1
                break
            if times > 1000:
                print('epoch', episode, r, '失败')
                time_visualize_list.append(times)
                success_timer_cnt = 0
                break
            s = s_
    
        if TRAIN_TO_TEST_SETUP:
            if success_timer_cnt == 3:
                success_timer_cnt = 0
                print('begin testing......')
                while good_test_cnt < 3:
                    step_cnt = testing_func()
                    if step_cnt <= finishe_step_cnt:
                        good_test_cnt += 1
                    else:
                        good_test_cnt = 0
                        break
                    if good_test_cnt == 3:
                        print('training success after', episode, 'episode')
                        return
        
    if not TRAIN_TO_TEST_SETUP:
        plt.figure(figsize=(32, 18))
        x = [i for i in range(episodes)]
        plt.plot(x, time_visualize_list, color="red", linewidth=2)
        plt.xlabel("episode", fontsize='large')
        plt.ylabel("steps", fontsize='large')
        plt.title("steps visualization")
        plt.savefig("steps.jpg")


def testing_func(): # Exploit Only
    actions_list = ['U', 'D', 'L', 'R']
    dict_a = {0:'up',1:'down',2:'left',3:'right'}
    times = 0
    s = env.reset()
    s = np.divide(s, 3)
    a_lib = []

    while True:
        times += 1
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        a = dql_agent.choose_best_action(s)
    
        a_lib.append(dict_a[int(a)])
        s_, r, done = env.step(actions_list[int(a)])
        s_ = np.divide(s_, 3)

        if done:
            print('成功', a_lib)
            break
        if times > 1000:
            print('失败')
            break
        s = s_
    return times

# Set up important variables
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
current_location = (0, 0)
end_location = (7, 7)
# maze = [[0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 1],
#         [0, 0, 0, 0, 1, 0, 0, 0]]
maze = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1, 0]]


# Construct allowed states
# allowed_states = {}

env = Maze(actions, current_location, end_location, maze)
allowed_states = env.allowed_states
# print(allowed_states)


# dql_agent = DQN_agent()
dql_agent = DQN(env.height * env.width)
if ENABLE_LEARNING:
    env.after(100, training_func)

tk.mainloop()