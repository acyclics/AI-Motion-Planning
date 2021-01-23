# AI Motion Planning
This repository is a WIP that aims to combine reachability analysis and optimal control to allow an agent, trained via reinforcement learning in simulation, to directly deploy its learned policy in the real world. The overarching idea is to deal with the generalization gap between simulation and real-world dynamics with reachability analysis and optimal control. 

Reachability analysis ideas and tools came from "FaSTrack: a Modular Framework for Fast and Guaranteed Safe Motion Planning".
The method of state-space decomposition of the AI robot came from "Decomposition of Reachable Sets and Tubes for a Class of Nonlinear Systems".

# Diagrams from reachability analysis

<img src="https://github.com/acyclics/AI-Motion-Planning/blob/master/examples/aimp_1.png" width="500" height="500"> <img src="https://github.com/acyclics/AI-Motion-Planning/blob/master/examples/aimp_2.png" width="500" height="500">

<img src="https://github.com/acyclics/AI-Motion-Planning/blob/master/examples/aimp_3.png" width="500" height="500">

# Using the computed optimal control from reachability analysis to track a target
![](https://github.com/acyclics/AI-Motion-Planning/blob/master/examples/optctrl.gif)

# Next step: train a reinforcement learning agent to motion-plan in simulation
![](https://github.com/acyclics/AI-Motion-Planning/blob/master/examples/aimp_4.png)
