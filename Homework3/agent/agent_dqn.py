import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.agent import Agent
from environment import Environment


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.get_action_space().n

        # Build target, online network.
        self.target_net = DQN(self.input_channels, self.num_actions).to(device)
        self.online_net = DQN(self.input_channels, self.num_actions).to(device)
        self.target_net.eval()

        if args.test_dqn:
            self.load('dqn')

        # discounted reward
        self.gamma = 0.99
        self.batch_size = 32

        # training hyperparameters
        self.learning_start = 10000  # before we start to update our network, we wait a few steps first to fill the replay.
        self.num_timesteps = 3000000  # total training steps

        self.target_update_freq = 1000  # frequency to update target network
        self.train_freq = 4  # frequency to train the online network
        self.save_freq = 200000  # frequency to save the model
        self.display_freq = 10  # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0  # num. of passed steps

        # replay buffer
        self.buffer_size = 10000
        self.replay_pre = torch.empty(self.buffer_size, 4, 84, 84, device=device)
        self.replay_next = torch.empty(self.buffer_size, 4, 84, 84, device=device)
        self.replay_action = torch.empty(self.buffer_size, dtype=torch.long, device=device)
        self.replay_reward = torch.empty(self.buffer_size, device=device)
        self.replay_done = torch.empty(self.buffer_size, dtype=torch.long, device=device)


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.pt')
        torch.save(self.target_net.state_dict(), save_path + '_target.pt')


    def load(self, load_path):
        print('load model from', load_path)
        self.online_net.load_state_dict(torch.load(load_path + '_online.pt', map_location=device))
        self.target_net.load_state_dict(torch.load(load_path + '_target.pt', map_location=device))


    def init_game_setting(self):
        pass


    def make_action(self, state, test=False):
        state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to(device)
        actions_prob = self.online_net(state).squeeze(0)
        best_action = torch.argmax(actions_prob, dim=0).item()
        # epsilon greedy
        if not test and random.random() < 0.1 * (1 / (self.steps + 1)):
            action = torch.randint(0, self.num_actions, (1, ))[0].item()
            # Explore an action differnet from the best one.
            while action == best_action:
                action = torch.randint(0, self.num_actions, (1, ))[0].item()
            return action

        return best_action

    def update(self):
        # Sample from replay buffer.
        index = torch.randint(0, self.buffer_size, (self.batch_size,))
        state = self.replay_pre[index]
        action = self.replay_action[index]
        next_state = self.replay_next[index]
        reward = self.replay_reward[index]
        done = self.replay_done[index]

        # Temporal difference.
        Q_value = torch.empty(self.batch_size, device=device)
        for i, qs in enumerate(self.online_net(state)):
            Q_value[i] = qs[action[i]]
        # Target value. (without updating target network)
        with torch.no_grad():
            Q_hat_value, _ = torch.max(self.target_net(next_state), dim=1)
        loss = torch.sum((reward + self.gamma * Q_hat_value * (1 - done) - Q_value) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0
        while True:
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)

            done = False
            while not done:
                # Select and perform action.
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # Update replay buffer.
                index = self.steps % self.buffer_size
                self.replay_pre[index] = torch.from_numpy(state).permute(2, 0, 1).to(device)
                self.replay_next[index] = torch.from_numpy(next_state).permute(2, 0, 1).to(device)
                self.replay_action[index] = action
                self.replay_reward[index] = reward
                self.replay_done[index] = 1 if done else 0

                # Move to the next state.
                state = next_state

                # Perform one step of the optimization.
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # Update target network.
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # Save the model.
                if self.steps % self.save_freq == 0:
                    self.save('dqn')

                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn')
