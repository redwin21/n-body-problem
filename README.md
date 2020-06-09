<p align="center">
<img align="center" width="1000" src="images/stars.gif">
</p>

# N-Body Can Solve This Problem

<p align="left">
<img align="center" src="https://img.shields.io/badge/last%20modified-june%20%202020-success">
|
<img align="center" src="https://img.shields.io/badge/status-in%20progress-yellow">
</p>

[Eddie Ressegue](http://github.com/redwin21)

The N-Body Problem is an analytically unsolvable astrophysics model that describes the motion of celestial bodies. This project attempts to capture the physics with machine learning. 

Presentation slides for this project can be found [here]().

---

## Table of Contents

- <a href="https://github.com/redwin21/n-body-problem#description-of-the-n-body-problem">Description of the N-Body Problem</a>
- <a href="https://github.com/redwin21/n-body-problem#simulation-and-data-generation">Simulation and Data Generation</a>
- <a href="https://github.com/redwin21/n-body-problem#Modeling">Modeling</a>
- <a href="https://github.com/redwin21/n-body-problem#predictions">Predictions</a>
- <a href="https://github.com/redwin21/n-body-problem#sources">Sources</a>
- <a href="https://github.com/redwin21/n-body-problem#implications">Implications</a>

---

## Description of the N-Body Problem

The N-Body problem is the problem of taking the position and velocity of point masses and solving for their motion according to Newton's Laws of motion and gravity. The *n* denotes the number of bodies in the system that is being modeled. The 2-body problem has an analytical solution, meaning there is a closesd-form set of equations to describe the motion of the point masses. However, the 3-body problem, and all number of bodies *n* greater than that, are not solvable the same way. There is a power series solution that describes teh motion, but is not solvable for practical purposes. The only way to determine the subsequent motion of those *n* point masses is to perform numerical integration, iterating over discrete time steps to determine some future state. This process is complex, and for reasonable models of distant future states, a lot of computing power is required.

Some of the equations that describe the motion of 3 bodies (where <a href="https://www.codecogs.com/eqnedit.php?latex=G" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G" title="G" /></a> is the universal gravitational constant, <a href="https://www.codecogs.com/eqnedit.php?latex=m_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_n" title="m_n" /></a> is the mass of a body, <a href="https://www.codecogs.com/eqnedit.php?latex=r_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_n" title="r_n" /></a> is the position, and <a href="https://www.codecogs.com/eqnedit.php?latex=\ddot{r}_n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ddot{r}_n" title="\ddot{r}_n" /></a> is the acceleration):

<p align="center">
<img align="center" width="300" src="images/3-body-eq.svg">
</p>

The high computing power needed comes form the fact that the problem is inherently chaotic. This means that minor issues with roundoff error can propogate as the simulation progresses and completely change the results. Therefore, high precision is required, which means very small time steps and a lot of computing power.

Machine learning uses pattern recognition to capture a relationship with data, and the laws of motion are well defined relationships. The idea behind this project is to train a machine learning model on the data from simulations, with the "predictor" being the current state of position and velocity, and the "target" being some future state after some number of time steps. This application could be used to greatly speed up or reduce computing time for generating models of n-body systems, improving scientific discoveries and experiments.

One of the disadvantages of approaching this with machine learning is the chaotic nature. The machine learning predictions have to be very precise to match the simulated model over time. For this reason, the machine learning predictions were never compounded, meaning no prediction was made from a prior prediction, but was instead made from a prior simulated state.

---

## Simulation and Data Generation

### Simulation Setup

The data used for modeling was generating by running 1000 simulations for various system configurations. The configurations addressed are:

- 2 bodies, same masses
- 2 bodies, different masses
- 3 bodies, same masses
- 3 bodies, different masses

Each simulation was created using a python class called `NBSimulator`, as seen in the [simulator](https://github.com/redwin21/n-body-problem/tree/master/simulator) folder. The object is initiated with initial positions, velocities, and masses for each body in the system. The simulation then uses an Ordinary Differential Equations (ODE) solver in the *Sci-Kit Learn* Python library to numerically integrate and solve for the subsequent positions and velocities for discrete time steps.

The data was generated by instantiating the 1000 simulations with random starting positions and velocities and masses (or masses of 1 if it was a "same mass" configuration). All positions for all times steps are stored in the simulation object when the simulation is run. All positions and velocites are relative to the center of mass of the system as a coordinate reference frame. The data from each run was dumped into a database for later processing.

An example plot of the a simulation can be seen below, where the lines trace the path of the orbit and the dot is the final state when plotted.

<p align="center">
<img align="center" width="400" src="./images/simulation.png">
</p>

### Data Converted for Modeling

The data provided by the simulation is the position and velocity of each body at each time step. To be used in a machine learning model, the data was restructured into having columns for each initial position and velocity paired with positions and velocity of some future time state. Each of the 1000 simulations was sampled randomly for an initial state and then the corresponding future state.

For the models in this study, time horizons of 10, 100, and 1000 time steps were used.

Here is a sample of what the data from a 2-body system would look like after the conversion: 

<p align="center">
<img align="center" width="800" src="./images/data_structure.png">
</p>

An example key for the data column names is as follows (and is consistent for all columns):

```
sim_id:     the arbitrary id of the simulation run
m_1:        mass of the first body
m_2:        mass of the second body
rx_1_0:     x position of the first body at time 0 (predictor)
vx_2_100:   y velocity of the second body at time 100 (target)
rz_2_1000:  z position of the second body at time 1000 (target)
```

All positions and velocities in these columns are relative to the center of mass of the system. 

---

## Modeling

#### Data Transformation for Modeling

Predicting future states of the n-body system requires a model with multiple outputs. For each configuration, there were the following number of inputs and outputs when using all of the data features:

| Configuration | Number of Inputs to the Model | Number of Outputs from the Model |
|---------------|:------------------:|:-------------------:|
| 2 bodies, same masses | 12 | 12 |
| 2 bodies, different masses | 14 | 12 |
| 3 bodies, same masses | 18 | 18 |
| 3 bodies, different masses | 21 | 18 |

Training a machine learning model with this many features, especially this many to predict, requires a lot of computing power and is prone to low accuracy. It turns out there is actually some redundant information in these features, however. The center of mass of the system, as a reference point, is arbitrary. If the reference frame was transformed to the first body, then the position and velocity of the first body would always be 0, so that body could be removed, allowing for some dimensionality reduction for the model. This is done by simply subtracting the position and velocity of the first body from the rest and reduces 6 columns of the data. A similar transformation can be done to convert the positions and velocites back to refering to the center of mass, so no information is actually lost. The number of features after the transformations are:

| Configuration | Number of Inputs to the Model | Number of Outputs from the Model |
|---------------|:------------------:|:-------------------:|
| 2 bodies, same masses | 6 | 6 |
| 2 bodies, different masses | 8 | 6 |
| 3 bodies, same masses | 12 | 12 |
| 3 bodies, different masses | 15 | 12 |

#### Model Performance

<div class="text-white bg-gray-dark mb-2">
  .text-white on .bg-gray-dark
</div>

---

## Predictions

<p align="center">
<img align="center" width="1200" src="images/gifs/3_bodies_100_steps_2.gif">
</p>

#### A note on scale and predictive capability

Time and position scales have been intentionally left off of these plots. As stated earlier, unit dimensions were used, so the plot animation doesn't reflect any reality. However, it is all relative, so it could reflect whatever reality you scale it to. For some clarity, these plots were generated witha. simulation of 5000 time steps, and take about 50 seconds to run. Therefore, each 1 second is about 100 time steps. With a model that predicts 100 time steps in the future, a 1 second prediction on this plot does not seem to be very good. A 10 second prediction makes it clear why the predictions for the 1000 step time horizon and beyond are so challenging.

---

## Implications

---

## Sources

This project would not be possible without the guidance and insight from various sources.

- Inspiration for this project: [Newton vs the machine: solving the chaotic three-body problem using deep neural networks](https://arxiv.org/abs/1910.07291)
- A tutorial for creating the n-body simulation: [Modelling the Three Body Problem in Classical Mechanics using Python](https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767)
- A textbook referenced for generalizing to *n* bodies: [The Three Body Problem](https://jfuchs.hotell.kau.se/kurs/amek/prst/04_3bdy.pdf)
- The source for the description information: [Three-body problem](https://en.wikipedia.org/wiki/Three-body_problem)
- Header gif: [Astronomers Observe Two Young Suns Collecting Matter in a Binary System](https://scitechdaily.com/astronomers-observe-two-young-suns-collecting-matter-in-a-binary-system/)

---