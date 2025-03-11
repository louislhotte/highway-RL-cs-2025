# **Project Overview: Highway-env Reinforcement Learning**

## **Problem Overview**
The objective of this project is to implement and train reinforcement learning (RL) agents in the **Highway-env** environment, specifically focusing on the **Racetrack** environment. This environment simulates a race track where autonomous vehicles must navigate efficiently while avoiding collisions. The goal is to develop agents that can drive optimally under different conditions. The project consists of three main implementation tasks and an additional experimental analysis.

---

## **Project Breakdown**

### **Task 1: Training a DQN Agent on the Racetrack Environment**
- Implement a **Deep Q-Network (DQN) agent from scratch** using the provided configuration file (`config.py`).
- The configuration file will be made available on **March 16** and will define the parameters of the **Racetrack** environment.
- The implementation can use code from lab sessions but must be developed independently.
- The training process should be documented, including:
  - Observations of different learning phases.
  - Performance evaluation of the trained agent.
  - Analysis of achieved rewards and behavior.

---

### **Task 2: Implementing an RL Agent with Continuous Actions in the Racetrack Environment**
- Modify the **Racetrack** environment to use **continuous action spaces**.
- Implement an RL algorithm that supports continuous actions (e.g., **DDPG, PPO, SAC**).
- Compare the performance of this agent with the discrete-action DQN agent from Task 1.
- Document how continuous control influences driving behavior and learning efficiency.

---

### **Task 3: Using Stable-Baselines for Training in the Racetrack Environment**
- Use **Stable Baselines3**, a library containing pre-implemented RL algorithms, to train an agent in the **Racetrack** environment.
- Train the agent using an existing RL algorithm (e.g., **PPO, A2C, TD3**).
- Compare its performance with the agents trained in the previous tasks.

---

### **Task 4: Additional Experiment**
- Conduct an **experiment** within the **Racetrack** environment.
- The experiment should explore an interesting RL aspect, such as:
  - The impact of hyperparameters on agent performance.
  - The generalization ability of the trained agent in modified environments.
  - Safe learning techniques to reduce collisions.
  - Multi-agent training effects (e.g., driving two cars simultaneously).
  - Whether the trained model can predict the outcomes of an MDP.
- Negative results (i.e., failed attempts) are acceptable as long as the approach and findings are well-documented.

---

## **Final Goal**
By the end of the project, the group should:
1. **Implement and train three different agents** in the **Racetrack** environment.
2. **Compare performance** between discrete and continuous action spaces.
3. **Conduct an additional experiment** to investigate an RL-related research question.
4. **Document findings and observations** in an individual report, analyzing agent behavior, training efficiency, and environment interactions.

---

## **Important Deadlines**
- **April 1st**: Group registration (max 4 members per group) and submission of two environment configuration files.
- **April 22nd**: Submission of individual reports detailing contributions and findings.

---

For further reference, visit the **Highway-env documentation**: [Highway-env Racetrack](https://highway-env.farama.org/environments/racetrack/)

