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

- <a href="https://github.com/redwin21/n-body-problem#history-of-the-n-body-problem">History of the N-Body Problem</a>
- <a href="https://github.com/redwin21/n-body-problem#simulation-and-data-generation">Simulation and Data Generation</a>
- <a href="https://github.com/redwin21/n-body-problem#Modeling">Modeling</a>
- <a href="https://github.com/redwin21/n-body-problem#predictions">Predictions</a>
- <a href="https://github.com/redwin21/n-body-problem#sources">Sources</a>
- <a href="https://github.com/redwin21/n-body-problem#implications">Implications</a>

---

## History of the N-Body Problem

---

## Simulation and Data Generation

---

## Modeling

#### Data Transformation for Modeling

#### Model Performance

---

## Predictions

<p align="center">
<img align="center" width="1000" src="images/gifs/3_bodies_100_steps_2.gif">
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
- A textbook referenced for generalizing to n-bodies: [The Three Body Problem](https://jfuchs.hotell.kau.se/kurs/amek/prst/04_3bdy.pdf)
- The source for historical information: [Three-body problem](https://en.wikipedia.org/wiki/Three-body_problem)
- Header gif: [Astronomers Observe Two Young Suns Collecting Matter in a Binary System](https://scitechdaily.com/astronomers-observe-two-young-suns-collecting-matter-in-a-binary-system/)

---