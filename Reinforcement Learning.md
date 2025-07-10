# Reinforcement Learning
Markov decision process (MDP) is a tuple $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$.  
A partially observable Markov decision process (POMDP) is a generalization of a Markov decision process (MDP); tuple $(\mathcal{S}, \mathcal{A}, T, R, \mathcal{O}, Z, \gamma)$.

| Symbol         | Meaning                                                                 |
|----------------|-------------------------------------------------------------------------|
| $\mathcal{S}$  | Set of states                                                           |
| $\mathcal{A}$  | Set of actions                                                          |
| $T$            | Transition model: $T(s' \mid s, a)$, probability of next state          |
| $R$            | Reward function: $R(s, a, s') \in \mathbb{R}$                           |
| $\gamma$       | Discount factor: $\gamma \in [0, 1)$, weighting future rewards          |
| $\mathcal{O}$  | Set of observations (POMDP only since State is hidden for it)           |
| $Z$            | Observation model: $Z(o \mid s', a)$, probability of observation        |

Process
1. The environment is in a state $s \in \mathcal{S}$.
2. The agent takes an action $a \in \mathcal{A}$.
3. The environment transitions to a new state $s' \in \mathcal{S}$ with probability $T(s' \mid s, a)$.
4. The agent receives an observation $o \in \mathcal{O}$ based on $s'$ and $a$ with probability $Z(o \mid s', a)$. An MDP does not include the observation set, because the agent always knows with certainty the environment's current state
5. The agent receives a reward $r = R(s, a)$.
6. The process repeats from the new state $s'$.
7. The agent's objective is to choose actions that maximize expected cumulative discounted rewards: $\mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$
   where:
   - $r_t$ is the reward at time $t$,
   - $\gamma \in [0,1)$ is the discount factor.
     - $\gamma = 0$: only immediate rewards matter.
     - $\gamma \to 1$: future rewards are weighted almost equally to immediate rewards.
8. After taking action $a$ and receiving observation $o$, the agent updates its belief over the state space $\mathcal{S}$. Since the system is Markovian, the updated belief $b'(s')$ only depends on the previous belief $b(s)$, the action $a$, and the observation $o$:

$${b'(s') = \eta \, Z(o \mid s', a) \sum_{s \in \mathcal{S}} T(s' \mid s, a) \, b(s)}$$

where:

- $b(s)$ is the prior belief (probability the environment is in state $s$)
- $T(s' \mid s, a)$ is the transition probability from $s$ to $s'$ under action $a$
- $Z(o \mid s', a)$ is the probability of observing $o$ in state $s'$ after taking action $a$
- $\eta$ is a normalizing constant ensuring that $\sum_{s'} b'(s') = 1$

The normalizing constant is given by:

$${\eta = \frac{1}{\Pr(o \mid b, a)}}$$

where:

$${\Pr(o \mid b, a) = \sum_{s' \in \mathcal{S}} Z(o \mid s', a) \sum_{s \in \mathcal{S}} T(s' \mid s, a) \, b(s)}$$


## Terms 
|  | **Offline RL** | **Online RL** |
|:-:|:-:|:-:|
| **Data source** | Fixed dataset of (state, action, reward, next state) | Live interaction with environment |
| **Learning** | Learns policy/value from the static dataset | Learns while collecting new data |
| **Interaction** | No further interaction allowed | Environment queried continuously |
| **Exploration** | Impossible | Essential (e.g., ε-greedy, UCB) |
| **Stability** | Safer, but requires careful distribution handling | Potentially unstable but more adaptive |



### Bootstrapping Formulas in Reinforcement Learning
Bootstrapping uses an estimate of future value to update the current estimate.

Let:  
- $s_t$: current state  
- $a_t$: current action  
- $r_t$: reward received after taking $a_t$ in $s_t$  
- $s_{t+1}$: next state  
- $\gamma$: discount factor  
- $\alpha$: learning rate  
- $Q(s, a)$: action-value function  
- $V(s)$: state-value function  
- $\pi(a \mid s)$: policy  

| # | Method                        | Update Rule / Formula |
|---|-------------------------------|------------------------|
| 1 | Temporal Difference (TD(0))   | $V(s_t) \leftarrow V(s_t) + \alpha \left[ r_t + \gamma V(s_{t+1}) - V(s_t) \right]$ |
| 2 | SARSA (On-Policy)             | $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$ |
| 3 | Q-Learning (Off-Policy)       | $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$ |
| 4 | Expected SARSA                | $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \sum_{a'} \pi(a' \mid s_{t+1}) Q(s_{t+1}, a') - Q(s_t, a_t) \right]$ |
| 5 | n-Step TD                     | $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$ |
| 6 | TD($\lambda$) (Forward View) | $G_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$, $V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t^\lambda - V(s_t) \right]$ |
| 7 | Bootstrapped Target Annotated | $\text{Target} = r_t + \gamma V(s_{t+1})$, with $V(s_{t+1})$ bootstrapped estimate |

