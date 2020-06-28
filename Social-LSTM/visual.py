import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(pred_traj, true_traj):
    fig_width = 10
    fig_height = 10

    fig = plt.figure(figsize=(fig_width, fig_width))

    plt.plot(true_traj[:, 0], true_traj[:, 1], color = 'G', linestyle = '-.', linewidth = 3,
        marker = 'p', markersize = 15, markeredgecolor = 'g', markerfacecolor = 'g')

    plt.plot(pred_traj[:, 0], pred_traj[:, 1], color = 'R', linestyle = '-.', linewidth = 3,
        marker = 'p', markersize = 10, markeredgecolor = 'r', markerfacecolor = 'r')

    plt.show()

# true_traj = np.array([[1,2], [3,4]])
# pred_traj = np.array([[1,3], [3,6]])

# plot_trajectories(true_traj, pred_traj)