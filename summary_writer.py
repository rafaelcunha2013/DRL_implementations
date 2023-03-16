from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tensorboard(folder_path, tag, save_dir, split_term, file_name='_'):

    # create a summary writer object to read events from the file
    # file_path = "./tutorial_pytorch/logs2/ddqn_run_2023_03_14__00_57_31__47/"
    # summary_writer = SummaryWriter(log_dir=file_path)

    # folder_path = './tutorial_pytorch/logs2/'
    subfolders = [f.name for f in os.scandir(folder_path) if (f.is_dir() and (file_name in f.name))]

    # tag = 'reward'
    df = pd.DataFrame(columns=['run', 'step', tag ])
    for name in subfolders:
        file_path = folder_path + name + '/'
        # Create an EventAccumulator object for a run
        ea = EventAccumulator(file_path)
        ea.Reload()

        # Get a list of available tags for this run
        # tags = ea.Tags()

        # Get the scalar data for a specific tag
        # tag = "reward"
        scalar_data = ea.Scalars(tag)

        # Convert to a pandas dataframe
        # df = pd.DataFrame([(s.step, s.value) for s in scalar_data])
        df = pd.concat([df, pd.DataFrame([(name, s.step, s.value) for s in scalar_data], columns=['run', 'step', tag])])

    df.reset_index()
    # split_term = '_'
    df_alg = df.run.apply(lambda run: run.split(split_term)[0])

    plt.figure(1, figsize=(16, 6))
    plt.clf()
    sns.lineplot(data=df, x="step", y=tag, hue=df_alg).set_title(f"DQN {tag.capitalize()}")
    plt.grid()
    # save_dir = './tutorial_pytorch/Dqn_comp_same_plot'
    plt.savefig(save_dir, dpi=200)

if __name__ == '__main__':
    # folder_path = './tutorial_pytorch/logs2/'
    # save_dir = './tutorial_pytorch/Dqn_comp_same_plot3'
    # file_name = '_'
    # tag = 'reward'  
    # split_term = '_'

    folder_path = '/data/p285087/drl_alg/logs/'
    #file_name = 'LunarLander-v2'
    file_name = 'four-room-multiagent-v0'
    split_term = '_run'

    tag = 'reward'  
    save_dir = f'/data/p285087/drl_alg/{file_name}_{tag}'   
    plot_tensorboard(folder_path, tag, save_dir, split_term, file_name)

    tag = 'disc_reward'  
    save_dir = f'/data/p285087/drl_alg/{file_name}_{tag}' 
    plot_tensorboard(folder_path, tag, save_dir, split_term, file_name)

    # tag = 'loss'  
    # save_dir = f'/data/p285087/drl_alg/{file_name}_{tag}' 
    # plot_tensorboard(folder_path, tag, save_dir, split_term, file_name)
    # print('Done')


