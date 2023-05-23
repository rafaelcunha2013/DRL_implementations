import numpy as np
import pandas as pd



gamma = 0.99

all_rewards = []
rewards = [(8, 0.5), (16, 1), (26, 0.5), (30, 1), (38, 1), (41, 0.5), (46, 1), (51, 0.5), (68, 1)]
all_rewards.append(rewards)
rewards = [(8, 0.5), (16, 1), (62, 0.5), (24, 1), (32, 1), (35, 0.5), (40, 1), (45, 0.5), (74, 1)]
all_rewards.append(rewards)
rewards = [(8, 0.5), (16, 1), (28, 0.5), (24, 1), (38, 1), (41, 0.5), (46, 1), (51, 0.5), (68, 1)]
all_rewards.append(rewards)
rewards = [(8, 0.5), (16, 1), (12, 0.5), (56, 1), (24, 1), (27, 0.5), (32, 1), (37, 0.5), (72, 1)]
all_rewards.append(rewards)
rewards = [(8, 0.5), (12, 0.5), (16, 1), (24, 1), (27, 0.5), (32, 1), (37, 0.5), (56, 1), (72, 1)]
all_rewards.append(rewards)



for rewards in all_rewards:
    cum_reward = 0
    for rw_pair in rewards:
        cum_reward += (gamma ** (rw_pair[0])) * rw_pair[1]

    print(cum_reward)

# name = "logs_habrok_four-room-multiagent-v0_ddqn_100000_run_2023_04_25__11_33_53__78.csv"
# df = pd.read_csv(name)
# # print(df.head())
# print(df['Value'].max())

# name = "logs_habrok_four-room-multiagent-v0_ddqn_100000_run_2023_04_25__11_33_53__80.csv"
# df = pd.read_csv(name)
# # print(df.head())
# print(df['Value'].max())