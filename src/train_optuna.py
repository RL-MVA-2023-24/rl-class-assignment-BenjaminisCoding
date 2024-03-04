import optuna
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
import json 

import argparse
import os
import random 

from main import main
from train import ProjectAgent

if torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = torch.cuda.FloatTensor
else:
    device = torch.device('cpu')
    dtype = torch.FloatTensor
# device = torch.device('cpu')

# from models import Discriminator, Generator


# If you don't want to bother with the device, stay on cpu:
# device = torch.device('cpu')
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


# Define an objective function to optimize
def objective(trial):

    seed_everything(seed=42)
    config = {'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                    'gamma': trial.suggest_float('gamma', 0.9, 0.99),
                    'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 150000]),  
                    'epsilon_min': trial.suggest_float('epsilon_min', 0.01, 0.1),
                    'epsilon_max': 1.,
                    'epsilon_decay_period': trial.suggest_int('epsilon_decay_period', 10000, 30000), # go plus haut? plus bas ?
                    'epsilon_delay_decay': trial.suggest_int('epsilon_delay_decay', 50, 200),
                    'batch_size': 800,
                    'gradient_steps': 3,
                    'update_target_strategy': trial.suggest_categorical('update_target_strategy', ['replace', 'ema']), # or 'ema'
                    'update_target_freq': trial.suggest_int('update_target_freq', 100, 1000),
                    'update_target_tau': 0.005}
                    # 'criterion': torch.nn.SmoothL1Loss()}


    agent = ProjectAgent()
    agent.train(config = config) ### train the agent 

    main() ### compute the score
    with open(file="score.txt", mode="r") as R:
            lines = R.readlines()
            scores = lines[0].strip().split('\n')
            score_agent = float(scores[0])
            score_agent_dr = float(lines[1])

    if not os.path.exists('score_best.txt'):
        with open(file="score_best.txt", mode="w") as W:
            W.write(f"{score_agent}\n{score_agent_dr}")
        with open(file='config.txt', mode='w') as W:
            config_json = json.dumps(config, indent=4)
            W.write(config_json)       

    with open(file="score_best.txt", mode="r") as R:
            lines = R.readlines()
            scores = lines[0].strip().split('\n')
            score_agent_best = float(scores[0])
            score_agent_dr_best = float(lines[1])    

    if score_agent_dr >= score_agent_dr_best:
        with open(file="score_best.txt", mode="w") as W:
            W.write(f"{score_agent}\n{score_agent_dr}")   
        agent.save('best_model.pt', custom = True) #and give a path

        with open(file='config.txt', mode='w') as W:
            config_json = json.dumps(config, indent=4)
            W.write(config_json)        

    return score_agent_dr


if __name__ == "__main__":

    # Create an Optuna study
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_exp', required=True)
    parser.add_argument('--n_trials', default = 500)
    # parser.add_argument('--num_epochs', default = 200)
    args = parser.parse_args()
    storage_url = f"sqlite:///{args.name_exp}.db"
    study = optuna.create_study(direction='maximize', study_name=args.name_exp, storage=storage_url, load_if_exists=True)


    # train_loader, train_data  = get_train_data() # your code to create the DataLoader
    # Optimize the objective function
    study.optimize(lambda trial: objective(trial), n_trials=int(args.n_trials), n_jobs = 1, catch=(optuna.exceptions.TrialPruned, ))

    # Get the best hyperparameters and their values
    best_params = study.best_params
    best_value = study.best_value

    print("Best Hyperparameters:", best_params)
    print("Best Wasserstein:", best_value)
    