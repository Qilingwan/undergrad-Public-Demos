# main.py: Q-Learning & SARSA on Taxi-v3
# Author: Chen
# Description: Implements Q-Learning and SARSA on Taxi-v3 to learn taxi navigation, compares convergence, explores strategies (ε-greedy, Softmax, UCB), and tunes α/γ via grid search with heatmap visualization.

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import hashlib
import random

id = input("Set id seed as:")
idHash = int(hashlib.sha256(id.encode()).hexdigest(), 16) % (2**31)

np.random.seed(idHash)
random.seed(idHash)

print(f"Environment seed is {idHash}")

env = gym.make("Taxi-v3", render_mode="rgb_array")
env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
nStates = env.observation_space.n
nActions = env.action_space.n
print(f"States: {nStates}, Actions: {nActions}")

def renderEpisode(Q, maxSteps=500):
    state, _ = env.reset()
    steps = 0
    done = False
    frames = []
    while not done and steps < maxSteps:
        action = np.argmax(Q[state])
        state, reward, done, _, _ = env.step(action)
        frames.append(env.render())
        steps += 1
    return frames, steps

def trainQLearning(episodes, alpha, gamma, epsilon, epsilonDecay=None):
    Q = np.zeros((nStates, nActions))
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        totalReward = 0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            nextState, reward, done, _, _ = env.step(action)
            totalReward += reward
            Q[state, action] += alpha * (reward + gamma * np.max(Q[nextState]) - Q[state, action])
            state = nextState
        rewards.append(totalReward)
        if epsilonDecay:
            epsilon = max(0.01, epsilon * epsilonDecay)
    return Q, rewards

def trainSarsa(episodes, alpha, gamma, epsilon):
    Q = np.zeros((nStates, nActions))
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        totalReward = 0
        done = False
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])
        while not done:
            nextState, reward, done, _, _ = env.step(action)
            totalReward += reward
            nextAction = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[nextState])
            Q[state, action] += alpha * (reward + gamma * Q[nextState, nextAction] - Q[state, action])
            state = nextState
            action = nextAction
        rewards.append(totalReward)
    return Q, rewards

alpha = 0.3
gamma = 0.9
epsilon = 0.3
episodes = 2000

Q_q, rew_q = trainQLearning(episodes, alpha, gamma, epsilon)
Q_s, rew_s = trainSarsa(episodes, alpha, gamma, epsilon)

def smoothRewards(rewards, window=100):
    return np.convolve(rewards, np.ones(window) / window, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(smoothRewards(rew_q), label='Q-Learning')
plt.plot(smoothRewards(rew_s), label='SARSA')
plt.xlabel('Episodes')
plt.ylabel('Smoothed Reward')
plt.title('Convergence Curves')
plt.legend()
plt.show()

def evalPolicy(Q, numEps=100):
    lengths = []
    for _ in range(numEps):
        state, _ = env.reset()
        steps = 0
        done = False
        while not done and steps < 500:
            action = np.argmax(Q[state])
            state, _, done, _, _ = env.step(action)
            steps += 1
        lengths.append(steps)
    return np.mean(lengths)

print(f"Q-Learning Mean Episode Length: {evalPolicy(Q_q)}")
print(f"SARSA Mean Episode Length: {evalPolicy(Q_s)}")

def softmaxExplore(Q, state, tau=1.0):
    probs = np.exp(Q[state] / tau) / np.sum(np.exp(Q[state] / tau))
    return np.random.choice(range(nActions), p=probs)

def ucbExplore(Q, state, counts, t, c=1.0):
    ucb = Q[state] + c * np.sqrt(np.log(t+1) / (counts[state] + 1e-5))
    return np.argmax(ucb)

def trainQLSoftmax(episodes, alpha, gamma, tau=1.0):
    Q = np.zeros((nStates, nActions))
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        totalReward = 0
        done = False
        while not done:
            action = softmaxExplore(Q, state, tau)
            nextState, reward, done, _, _ = env.step(action)
            totalReward += reward
            Q[state, action] += alpha * (reward + gamma * np.max(Q[nextState]) - Q[state, action])
            state = nextState
        rewards.append(totalReward)
    return Q, rewards

def trainQLUCB(episodes, alpha, gamma, c=1.0):
    Q = np.zeros((nStates, nActions))
    counts = np.zeros((nStates, nActions))
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        totalReward = 0
        done = False
        t = 0
        while not done:
            action = ucbExplore(Q, state, counts, t, c)
            nextState, reward, done, _, _ = env.step(action)
            totalReward += reward
            counts[state, action] += 1
            Q[state, action] += alpha * (reward + gamma * np.max(Q[nextState]) - Q[state, action])
            state = nextState
            t += 1
        rewards.append(totalReward)
    return Q, rewards

Q_soft, rew_soft = trainQLSoftmax(episodes, alpha, gamma)
Q_ucb, rew_ucb = trainQLUCB(episodes, alpha, gamma)

plt.figure(figsize=(10, 6))
plt.plot(smoothRewards(rew_q), label='ε-Greedy')
plt.plot(smoothRewards(rew_soft), label='Softmax')
plt.plot(smoothRewards(rew_ucb), label='UCB')
plt.xlabel('Episodes')
plt.ylabel('Smoothed Reward')
plt.title('Exploration Strategies')
plt.legend()
plt.show()

alphaVals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gammaVals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for a in alphaVals:
    for g in gammaVals:
        Q, rew = trainQLearning(episodes, a, g, epsilon)
        avg = np.mean(rew[-100:])
        results.append({'alpha': a, 'gamma': g, 'reward': avg})

df = pd.DataFrame(results)
pivot = df.pivot(index='alpha', columns='gamma', values='reward')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.1f')
plt.title('Average Reward over Last 100 Episodes')
plt.xlabel('Gamma (γ)')
plt.ylabel('Alpha (α)')
plt.show()