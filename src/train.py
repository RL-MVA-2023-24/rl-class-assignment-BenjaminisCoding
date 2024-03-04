from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
# from main import seed_everything

import random
import torch
import torch.nn as nn
import numpy as np

import os
from tqdm import tqdm 
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled

        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    
    def __len__(self):
        return len(self.data)
    

class ProjectAgent:

    def act(self, observation, use_random=False): #greedy

        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"

        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path, custom = False):
        '''
        save as model.pt
        if custom is set to True, than save at the patch variable
        '''

        self.path = path + "/model.pt"
        if custom:
            torch.save(self.model.state_dict(), path)
            return
        torch.save(self.model.state_dict(), self.path)

    def load(self):

        device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.path = os.getcwd() + "/model.pt"
        self.path = os.getcwd() + "/best_model.pt"

        self.model = self.network({}, device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()

    def network(self, config, device):

        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 
        nb_neurons=256 

        DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, n_action)).to(device)
        return DQN

    ## UTILITY FUNCTIONS
    
    def greedy_action(self, network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def gradient_step_v2(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            Q_target_Ymax = self.target_model(Y).max(1)[0].detach()
            Q_Ymax = self.model(Y).max(1)[0].detach()
            next_Q = torch.min(Q_target_Ymax, Q_Ymax)
            update = torch.addcmul(R, 1-D, next_Q, value=self.gamma)
            Q_target_XA = self.target_model(X).gather(1, A.to(torch.long).unsqueeze(1))
            Q_XA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            
            loss = self.criterion(Q_target_XA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            
            loss = self.criterion(Q_XA, update.unsqueeze(1))
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step() 
    
    def train(self, config = None):

        if config is None : #baseline config 
            config = {'nb_actions': env.action_space.n,
                    'criterion': torch.nn.SmoothL1Loss(),
                    'learning_rate': 0.001,
                    'batch_size': 800,
                    'gamma': 0.98,
                    'buffer_size': 100000,
                    'epsilon_min': 0.02,
                    'epsilon_max': 1.,
                    'epsilon_decay_period': 20000,
                    'epsilon_delay_decay': 100,
                    'gradient_steps': 3,
                    'update_target_freq': 400,
                    'update_target_tau': 0.005,
                    'update_target_strategy': 'replace'}
            
        else:
            ### add extra parameters not tuned 
            config['nb_actions'] = env.action_space.n
            config['criterion'] = torch.nn.SmoothL1Loss()

        # architecture
        device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', device)
        self.model = self.network(config, device)
        self.target_model = deepcopy(self.model).to(device)

        ### set the parameters for the experiment

        # memory buffer
        self.memory = ReplayBuffer(config['buffer_size'], device)


        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']

        # epsilon greedy strategy
        epsilon_max = config['epsilon_max']
        epsilon_min = config['epsilon_min']
        epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        epsilon_step = (epsilon_max-epsilon_min)/epsilon_stop

        # learning parameters (loss, lr, optimizer, gradient step)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer2 = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=88, gamma=0.1)

        nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1

        # target network
        update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005


        val_before = 0
        ## INITIATE NETWORK

        # set max episode. Could include this parameter as an argument of the train function. 
        max_episode = 220

        episode_return = []
        episode = 0
        state, _ = env.reset()
        epsilon = epsilon_max
        step = 0
        episode_cum_reward = 0


        ## TRAIN NETWORK
        pbar = tqdm(total=max_episode, desc="Progress", position=0) #to see the progress 
        while episode < max_episode:
            # update epsilon
            if step > epsilon_delay:
                epsilon = max(epsilon_min, epsilon-epsilon_step)
            # select epsilon-greedy action
                
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
                
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train 
            for _ in range(nb_gradient_steps): 
                self.gradient_step()

            if update_target_strategy == 'replace':
                if step % update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)

            # next transition
            step += 1
            if done or trunc:
                episode += 1
                pbar.update(1) 
                if episode >= 100: #compute a val score 

                    seed_everything(seed=42)
                    val_score= evaluate_HIV(agent=self, nb_episode=1)
                    print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", episode return ", '{:.2e}'.format(episode_cum_reward),
                      # evaluation score 
                      ", val score ", '{:.2e}'.format(val_score),
                      sep='')
                else :
                    val_score= 0

                state, _ = env.reset()

                # save if model improved 
                if val_score > val_before:
                    print("better model")
                    val_before = val_score
                    self.best_model = deepcopy(self.model).to(device)
                    self.save(os.getcwd())
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0

            else:
                state = next_state


        self.model.load_state_dict(self.best_model.state_dict())
        path = os.getcwd()
        self.save(os.getcwd())
        print('Model saved !, at: ', path)
        return episode_return


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


### to train the agent
if __name__ == "__main__":

    seed_everything(seed=42)
    agent = ProjectAgent()
    agent.train()