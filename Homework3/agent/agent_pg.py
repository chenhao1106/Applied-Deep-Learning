import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from agent.agent import Agent
from environment import Environment


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim=self.env.get_observation_space().shape[0],
                               action_num=self.env.get_action_space().n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('./pg.pt')
            self.model.eval()

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        self.num_episodes = 10000  # total training episodes (actually too large...)
        self.display_freq = 10  # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and probability of actions
        self.rewards, self.action_prob = [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)


    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))


    def init_game_setting(self):
        self.rewards, self.action_prob = [], []


    def make_action(self, state, test=False):
        state = torch.tensor(state).unsqueeze(0)
        action_probs = self.model(state).squeeze(0)

        distribution = distributions.Categorical(probs=action_probs)
        action = distribution.sample()
        if not test:
            self.action_prob.append(distribution.log_prob(action))
        return action.item()


    def update(self):
        discounted_rewards = 0.
        for i in range(len(self.rewards) - 1, -1, -1):
            discounted_rewards = self.rewards[i] + self.gamma * discounted_rewards

        loss = -discounted_rewards * torch.sum(torch.stack(self.action_prob))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self):
        avg_reward = None
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            
            done = False
            while not done:
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.rewards.append(reward)

            # update model
            self.update()

            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))

            if avg_reward > 50:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                self.save('./pg.pt')
                break

