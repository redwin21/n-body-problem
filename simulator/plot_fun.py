import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rcParams

def plot_3d(n, r):
    fig=plt.figure(figsize=(15,15))
    ax=fig.add_subplot(111,projection="3d")
    
    for i in range(n):
        r_plot = r[:,i*3:i*3+3]
        ax.plot(r_plot[:,0],r_plot[:,1],r_plot[:,2])
        label = f'body {i+1}'
        ax.scatter(r_plot[-1,0],r_plot[-1,1],r_plot[-1,2],marker="o",s=100, label=label)

    ax.set_xlabel("x-coordinate",fontsize=14)
    ax.set_ylabel("y-coordinate",fontsize=14)
    ax.set_zlabel("z-coordinate",fontsize=14)
    ax.set_title("Visualization of orbits of stars in a two-body system\n",fontsize=14)
    ax.legend(loc="upper left",fontsize=14)
    
    
def animate(i, r, lines, speed=20):
    for line, instant in zip(lines, r):
        line.set_data(instant[:speed*i,0], instant[:speed*i,1])
        line.set_3d_properties(instant[:speed*i, 2])
    return lines


writer = animation.PillowWriter(fps=60)


def plot_3d_animate(n, r, path=None, speed=20):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection="3d")
    ax.set_xlim3d([-1.5, 0.5])
    ax.set_ylim3d([0.0, 0.3])
    ax.set_zlim3d([-2.5, 0.0])
    r_plot = [r[:,i*3:i*3+3] for i in range(n)]
    lines = [ax.plot(b[0:1, 0], b[0:1, 1], b[0:1, 2])[0] for b in r_plot]

    line_ani = animation.FuncAnimation(fig, 
                                       animate, 
                                       fargs=(r_plot, lines, speed), 
                                       frames=int(r.shape[0]/speed), 
                                       interval=1)
    if path:
        line_ani.save(path, writer=writer)