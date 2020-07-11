import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import multivariate_normal

def draw_heatmap(mux, muy, sx, sy, rho, plt = None, bound = 0.1):
    x, y = np.meshgrid(np.linspace(mux - bound, mux + bound, 200),
                       np.linspace(muy - bound, muy + bound, 200))
    
    mean = [mux, muy]

    # Extract covariance matrix
    cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    
    gaussian = multivariate_normal(mean = mean, cov = cov)
    d = np.dstack([x, y])
    z = gaussian.pdf(d)

    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    plt.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max, alpha = 0.5)

def visual():
    data_file = "./pred_results.pkl"

    f = open(data_file, "rb")
    visual_data = pickle.load(f)
    f.close()

    pred_trajs = visual_data[0]
    truth_trajs = visual_data[1]
    gauss_params = visual_data[2]

    traj_num = len(pred_trajs)

    for index in range(traj_num):
        visual_trajectories(pred_trajs[index], truth_trajs[index], gauss_params[index])



def visual_trajectories(pred_traj, true_traj, gauss_param):
    fig_width = 10
    fig_height = 10

    fig = plt.figure(figsize=(fig_width, fig_width))

    plt.plot(true_traj[:, 0], true_traj[:, 1], color = 'G', linestyle = '-.', linewidth = 3,
        marker = 'p', markersize = 15, markeredgecolor = 'g', markerfacecolor = 'g')

    plt.plot(pred_traj[:, 0], pred_traj[:, 1], color = 'R', linestyle = '-.', linewidth = 3,
        marker = 'p', markersize = 10, markeredgecolor = 'r', markerfacecolor = 'r')

    plt.show()


visual()