import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

"""
Generates a plot visualization with mean and +/-1 standard deviation
"""

# tensorboard dev upload --logdir './tutorial_pytorch/logs2/'

# experiment_id = "KEdW6yKDSOOHbZU76DzvMg"
experiment_id = "EtWlV14tQbK6zsLozhm2rQ"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
# print(df["run"].unique())
# print(df["tag"].unique())

df = df[df['tag']=='reward']
csv_path = './tutorial_pytorch/dqn_comparison.csv'
df.to_csv(csv_path, index=False)


# dfw = experiment.get_scalars(pivot=True) 


# # Filter the DataFrame to only validation data, which is what the subsequent
# # analyses and visualization will be focused on.
# dfw_validation = df[dfw.run.str.endswith("/validation")]
# # Get the optimizer value for each row of the validation DataFrame.
# optimizer_validation = dfw_validation.run.apply(lambda run: run.split(",")[0])
df_alg = df.run.apply(lambda run: run.split("_")[0])
plt.figure(2, figsize=(16, 6))
sns.lineplot(data=df, x="step", y="value", hue=df_alg).set_title("DQN Reward")
plt.savefig('./tutorial_pytorch/Dqn_comp_same_plot', dpi=200)

plt.figure(1, figsize=(16, 6))
plt.subplot(1, 2, 1)
# sns.lineplot(data=df, x="step", y="value", hue=optimizer_validation).set_title("reward")
df_dqn = df[df.run.str.startswith("dqn_")]
sns.lineplot(data=df_dqn, x="step", y="value").set_title("DQN Reward")
plt.grid()


plt.subplot(1, 2, 2)
df_ddqn = df[df.run.str.startswith("ddqn_")]
sns.lineplot(data=df_ddqn, x="step", y="value").set_title("DDQN Reward")
plt.grid()

plt.savefig('./tutorial_pytorch/Dqn_alg_comparison.png', dpi=200)

# plt.subplot(1, 2, 2)
# df_ddqn2 = df[df.run.str.startswith("ddqn2")]
# sns.lineplot(data=df[df.run.str.startswith("ddqn2")], x="step", y="value").set_title("DDQN2 Reward")
# plt.grid()
