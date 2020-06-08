# Simulating the N-Body Problem

The simulation of the problem is built using the physics described in [this](https://jfuchs.hotell.kau.se/kurs/amek/prst/04_3bdy.pdf) paper, generalized for more than 3 bodies.

The `simulator.py` file provies a class that can be used to simulate the n-body problem given the number of bodies as well as the starting mass, position, and velocity. It uses an ODE solver to simulate each successive set of positions and velocities for the system given a time series.

The workflow for generating the data used for modeling is as follows:

1. An EC2 instance on AWS was launched to handle the large amount of computing power necessary to run the simulator for thousands of simulations.

2. A MongoDB container was launched on Docker inside the EC2 instance to temporarily store teh large amounts of generated data.

3. The `data_generation.py` file was run to create several thousand random initial states of position and velocity to then run the n-body simulation and generate the data. This was run for 2 and 3 body systems, with and without the same starting mass (4 database tables total) and ended with about 10 GB of data.

4. The `table_data.py` file was run to convert the simulation data into a table structure with "predictors" and "targets" that could be used for modeling. More on the modeling can be found in the `README` in the `model` folder of this repository.

Examples of the final data tables can be found in the `data` folder of this repository.

These scripts can be run to generate data and repeat this experiment.