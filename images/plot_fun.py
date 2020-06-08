import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rcParams
import numpy as np

def plot_3d(r):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(111,projection="3d")
    
    n = int(r.shape[1]/3)
    colors = ['r','b','g']
    
    for i in range(n):
        r_plot = r[:,i*3:i*3+3]
        ax.plot(r_plot[:,0],r_plot[:,1],r_plot[:,2], c=colors[i], alpha=0.8)
        ax.scatter(r_plot[-1,0],r_plot[-1,1],r_plot[-1,2],marker="o", s=100, c=colors[i])

    ax.set_xlabel('X', color='w')
    ax.set_ylabel('Y', color='w')
    ax.set_zlabel('Z', color='w')
    
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    
#     for line in ax.xaxis.get_ticklines():
#         line.set_visible(False)
#     for line in ax.yaxis.get_ticklines():
#         line.set_visible(False)
#     for line in ax.zaxis.get_ticklines():
#         line.set_visible(False)
        
    fig.set_facecolor('black')
    ax.set_facecolor('black')
#     ax.xaxis.pane.set_edgecolor('w')
#     ax.yaxis.pane.set_edgecolor('w')
#     ax.zaxis.pane.set_edgecolor('w')
#     ax.set_axis_off()
#     ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
    
    
def animate(i, r, lines, scatters, speed=20):
    for line, scatter, instant in zip(lines, scatters, r):
        line.set_data(instant[:speed*i,0], instant[:speed*i,1])
        line.set_3d_properties(instant[:speed*i, 2])
        
        scatter.set_data(instant[speed*i,0], instant[speed*i,1])
        scatter.set_3d_properties(instant[speed*i, 2])
    return lines


def plot_3d_animate(r, path=None, speed=20, interval=1):
    n = int(r.shape[1]/3)
    colors = ['r','b','g']
    idx = [[], [], [[0,3],[1,4],[2,5]],[[0,3,6],[1,4,7],[2,5,8]]]
    xlim = [np.min(np.min(r, axis=0)[idx[n][0]])*1.1, np.max(np.max(r, axis=0)[idx[n][0]])*1.1]
    ylim = [np.min(np.min(r, axis=0)[idx[n][1]])*1.1, np.max(np.max(r, axis=0)[idx[n][1]])*1.1]
    zlim = [np.min(np.min(r, axis=0)[idx[n][2]])*1.1, np.max(np.max(r, axis=0)[idx[n][2]])*1.1]
    
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection="3d")
    
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    ax.set_xlabel('X', color='w')
    ax.set_ylabel('Y', color='w')
    ax.set_zlabel('Z', color='w')
    
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
        
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
#     ax.set_axis_off()
#     ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    
    r_plot = [r[:,i*3:i*3+3] for i in range(n)]
    lines = [ax.plot(b[0:1, 0], b[0:1, 1], b[0:1, 2], c=colors[i], alpha=0.5)[0] for i, b in enumerate(r_plot)]
    scatters = [ax.plot(b[0:1, 0], b[0:1, 1], b[0:1, 2], marker='o', ms=10, c=colors[i])[0] for i, b in enumerate(r_plot)]

    line_ani = animation.FuncAnimation(fig=fig, 
                                       func=animate, 
                                       fargs=(r_plot, lines, scatters, speed), 
                                       frames=int(r.shape[0]/speed), 
                                       interval=interval)
    
    if path:     
        writer = animation.PillowWriter(fps=60)
        line_ani.save(path, writer=writer)
        
    return line_ani


def animate_sbs(i, r_true, r_pred, lines_true, lines_pred, scatters_true, statters_pred, speed=20):
    
    for line_true, line_pred, scat_true, scat_pred, i_true, i_pred in zip(lines_true, lines_pred, scatters_true, statters_pred, r_true, r_pred):
        line_true.set_data(i_true[:speed*i,0], i_true[:speed*i,1])
        line_true.set_3d_properties(i_true[:speed*i, 2])
        
        scat_true.set_data(i_true[speed*i,0], i_true[speed*i,1])
        scat_true.set_3d_properties(i_true[speed*i, 2])
        
        line_pred.set_data(i_pred[:speed*i,0], i_pred[:speed*i,1])
        line_pred.set_3d_properties(i_pred[:speed*i, 2])
        
        scat_pred.set_data(i_pred[speed*i,0], i_pred[speed*i,1])
        scat_pred.set_3d_properties(i_pred[speed*i, 2])
    return lines_true

def plot_3d_animate_sbs(r_true, r_pred, path=None, speed=20, interval=1, steps=0):
    n = int(r_true.shape[1]/3)
    colors = ['r','b','g']
    idx = [[], [], [[0,3],[1,4],[2,5]],[[0,3,6],[1,4,7],[2,5,8]]]
    xlim = [np.min(np.min(r_true, axis=0)[idx[n][0]])*1.1, np.max(np.max(r_true, axis=0)[idx[n][0]])*1.1]
    ylim = [np.min(np.min(r_true, axis=0)[idx[n][1]])*1.1, np.max(np.max(r_true, axis=0)[idx[n][1]])*1.1]
    zlim = [np.min(np.min(r_true, axis=0)[idx[n][2]])*1.1, np.max(np.max(r_true, axis=0)[idx[n][2]])*1.1]
    
    fig=plt.figure(figsize=(16,8))
    ax1=fig.add_subplot(1,2,1, projection="3d")
    ax2=fig.add_subplot(1,2,2, projection="3d")
    axs = [ax1, ax2]
    
    for ax in axs:
        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(zlim)

        ax.set_xlabel('X', color='w')
        ax.set_ylabel('Y', color='w')
        ax.set_zlabel('Z', color='w')

        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)

        fig.set_facecolor('black')
        ax.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor('k')
        ax.yaxis.pane.set_edgecolor('k')
        ax.zaxis.pane.set_edgecolor('k')
#         ax.set_axis_off()
#         ax.grid(False)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.1))
    
    t = f'Predicted Orbits: {steps} Time Steps'
    axs[0].set_title('Simulated Orbits', fontsize=20, color='w')
    axs[1].set_title(t, fontsize=20, color='w')
    
    r_plot_true = [r_true[:,i*3:i*3+3] for i in range(n)]
    r_plot_pred = [r_pred[:,i*3:i*3+3] for i in range(n)]
    
    lines_true = [axs[0].plot(b[0:1, 0], b[0:1, 1], b[0:1, 2], c=colors[i], alpha=0.5)[0] for i, b in enumerate(r_plot_true)]
    lines_pred = [axs[1].plot(b[0:1, 0], b[0:1, 1], b[0:1, 2], c=colors[i], alpha=0.5)[0] for i, b in enumerate(r_plot_pred)]
    
    scatters_true = [axs[0].plot(b[0:1, 0], b[0:1, 1], b[0:1, 2], marker='o', ms=10, c=colors[i])[0] for i, b in enumerate(r_plot_true)]
    scatters_pred = [axs[1].plot(b[0:1, 0], b[0:1, 1], b[0:1, 2], marker='o', ms=10, c=colors[i])[0] for i, b in enumerate(r_plot_pred)]
    
    plt.subplots_adjust(wspace=0.0)

    line_ani = animation.FuncAnimation(fig=fig, 
                                       func=animate_sbs, 
                                       fargs=(r_plot_true, r_plot_pred, lines_true, lines_pred, scatters_true, scatters_pred, speed), 
                                       frames=int(r_true.shape[0]/speed), 
                                       interval=interval)
    
    if path:     
        writer = animation.PillowWriter(fps=60)
        line_ani.save(path, writer=writer)
        
    return line_ani
