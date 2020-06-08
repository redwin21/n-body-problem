import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rcParams
import numpy as np

def plot_3d(r):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection="3d")
    
    n = int(r.shape[1]/3)
    colors = ['r','b','g']
    
    for i in range(n):
        r_plot = r[:,i*3:i*3+3]
        ax.plot(r_plot[:,0],r_plot[:,1],r_plot[:,2], c=colors[i], alpha=0.8)
        ax.scatter(r_plot[-1,0],r_plot[-1,1],r_plot[-1,2],marker="o", s=100, c=colors[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
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
    ax.xaxis.pane.set_edgecolor('b')
    ax.yaxis.pane.set_edgecolor('b')
    ax.zaxis.pane.set_edgecolor('b')
    ax.set_axis_off()
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    
    
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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
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
    ax.xaxis.pane.set_edgecolor('b')
    ax.yaxis.pane.set_edgecolor('b')
    ax.zaxis.pane.set_edgecolor('b')
    ax.set_axis_off()
    ax.grid(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    
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