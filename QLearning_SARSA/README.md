# Reinforcement Learning: Q-Learning and SARSA on Taxi-v3

## 1. Project Description

(a) This project implements and compares two reinforcement learning algorithms — Q-Learning and SARSA, in the Taxi-v3 environment from OpenAI Gym.<br>
(b) It analyzes convergence behavior and exploration efficiency under different strategies including ε-greedy, Softmax, and Upper Confidence Bound (UCB).<br>
(c) The project further explores hyperparameter sensitivity through a grid search of the learning rate (α) and discount factor (γ), visualized with heatmaps.<br>

## 2. Tech Stack / Tools Used

(a) Python 3.10<br>
(b) Gymnasium (Taxi-v3 environment)<br>
(c) NumPy, Pandas for data handling and numerical computation<br>
(d) Matplotlib, Seaborn for visualization<br>

## 3. Objectives / Tasks

### 3.1 Algorithm Implementation

(a) Implement Q-Learning and SARSA algorithms from scratch using tabular methods.<br>
(b) Compare their convergence performance and policy stability.<br>

### 3.2 Exploration Strategy Analysis

(a) Integrate ε-greedy, Softmax, and UCB exploration strategies into the Q-Learning framework.<br>
(b) Assess how different exploration mechanisms influence training efficiency.<br>

### 3.3 Parameter Optimization

(a) Perform grid search over α (learning rate) and γ (discount factor).<br>
 (b) Analyze performance trends with visual heatmaps to identify optimal combinations.<br>

## 4. Implementation / Methods

### 4.1 Environment Setup

(a) Initialize the Taxi-v3 environment with controlled random seeds to ensure reproducibility.<br>
(b) Define environment state and action spaces for agent-environment interaction.<br>

### 4.2 Q-Learning Implementation

(a) Initialize a Q-table with zeros representing state-action values.<br>
(b) Use ε-greedy exploration to balance exploration and exploitation.<br>
(c) Update Q-values using the Bellman optimality equation with maximum next-state value.<br>
(d) Optionally apply ε decay for more stable convergence.<br>

### 4.3 SARSA Implementation

(a) Initialize a Q-table for on-policy learning.<br>
(b) Select actions using ε-greedy and update values based on the next action actually taken.<br>
(c) Compare to Q-Learning to demonstrate the difference between on-policy and off-policy learning.<br>

### 4.4 Extended Exploration Strategies

(a) Implement Softmax exploration to select actions probabilistically based on Q-values.<br>
(b) Implement Upper Confidence Bound (UCB) exploration to balance uncertainty and reward estimates.<br>
(c) Visualize the reward trends of each method over episodes.<br>

### 4.5 Visualization and Evaluation

(a) Apply moving-average smoothing to reward curves for better convergence visualization.<br>
(b) Plot performance curves comparing different learning and exploration strategies.<br>
(c) Generate a heatmap showing the effect of α and γ combinations on average reward.<br>

## 5. Results / Outputs

<p align="center">
  <img src="outputDemos/learningCurve.png" alt="Learning Curve" width="50%">
</p>

<p align="center">
  <img src="outputDemos/heatMap.png" alt="Heat Map" width="50%">
</p>


## 6. Conclusion / Insights

(a) Q-Learning converges faster but may be less stable under high exploration settings.<br>
(b) SARSA demonstrates safer convergence through on-policy updates.<br>
(c) Softmax and UCB strategies improve exploration efficiency and reduce variance when tuned properly.<br>
(d) Mid-range α and γ values tend to yield optimal learning performance, balancing exploration depth and update stability.<br>

## 7. Acknowledgements / References

(a) OpenAI Gymnasium for the Taxi-v3 environment.<br>
(b) Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction (2nd Edition).<br>
(c) NumPy, Pandas, Matplotlib, and Seaborn official documentation.<br>

