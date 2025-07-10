# Reinforcement learning
Markov decision process (MDP) is a tuple $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$.  
A partially observable Markov decision process (POMDP) is a generalization of a Markov decision process (MDP); tuple $(\mathcal{S}, \mathcal{A}, T, R, \mathcal{O}, Z, \gamma)$.

| Symbol         | Meaning                                                                 |
|----------------|-------------------------------------------------------------------------|
| $\mathcal{S}$  | Set of states                                                           |
| $\mathcal{A}$  | Set of actions                                                          |
| $\mathcal{O}$  | Set of observations (POMDP only, state is hidden)                       |
| $s$            | Current state                                                           |
| $a$            | Action taken by the agent                                               |
| $s'$           | Next state reached after applying action $a$ in state $s$               |
| $o$            | Observation received after reaching state $s'$                          |
| $T$            | Transition model: $T(s' \mid s, a)$, probability of next state          |
| $Z$            | Observation model: $Z(o \mid s', a)$, probability of observation        |
| $R$            | Reward function: $R(s, a, s') \in \mathbb{R}$                           |
| $\gamma$       | Discount factor: $\gamma \in [0, 1)$, weighting future rewards          |
| $b(s)$         | Prior belief: probability environment is in state $s$                   |
| $b'(s')$       | Updated belief: probability environment is in $s'$ after $a$, $o$       |
| $\eta$         | Normalizing constant: ensures $\sum_{s'} b'(s') = 1$                    |

The normalizing constant is:

$\eta = \frac{1}{\Pr(o \mid b, a)}$

Where:

$\Pr(o \mid b, a) = \sum_{s' \in \mathcal{S}} Z(o \mid s', a) \sum_{s \in \mathcal{S}} T(s' \mid s, a) b(s)$

### How it works:
1. The environment is in a state $s \in \mathcal{S}$.
2. The agent takes an action $a \in \mathcal{A}$.
3. The environment transitions to a new state $s' \in \mathcal{S}$ with probability $T(s' \mid s, a)$.
4. The agent receives an observation $o \in \mathcal{O}$ based on $s'$ and $a$ with probability $Z(o \mid s', a)$. (In MDPs, this step is skipped.)
5. The agent receives a reward $r = R(s, a, s')$.
6. The process repeats from the new state $s'$.
7. The agent's objective is to maximize expected cumulative discounted rewards:

   $\mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$

   - $\gamma = 0$: only immediate rewards matter.
   - $\gamma \to 1$: future rewards are weighted almost equally.

8. In a POMDP, the agent does not observe $s'$ directly and must update its belief: $b'(s') = \eta \, Z(o \mid s', a) \sum_{s \in \mathcal{S}} T(s' \mid s, a) \, b(s)$
9. The belief MDP is defined as the tuple $(\mathcal{B}, \mathcal{A}, \tau, r, \gamma)$.

   The belief transition function is:

   $\tau(b, a, b') = \sum_{o \in \mathcal{O}} \Pr(b' \mid b, a, o) \Pr(o \mid b, a)$

   where:
   - $\Pr(b' \mid b, a, o) = 1$ if $b'$ is the result of belief update from $(b, a, o)$, else $0$

   The reward in the belief MDP is the expected reward over belief:

   $r(b, a) = \sum_{s \in \mathcal{S}} b(s) R(s, a)$

   Since $b$ is known, the belief MDP is fully observable and standard MDP solvers can be applied.
10. In the belief MDP, all belief states allow all actions because belief distributions are nonzero over all of $\mathcal{S}$.  
A policy $\pi$ maps any belief $b$ to an action $a = \pi(b)$.  
The expected return from a policy $\pi$ starting from $b_0$ is:  

$$
V^\pi(b_0) = \sum_{t=0}^{\infty} \gamma^t r(b_t, a_t) = \sum_{t=0}^{\infty} \gamma^t \mathbb{E}[ R(s_t, a_t) \mid b_0, \pi ]
$$

The optimal policy $\pi^*$ is:  

$$
\pi^* = \arg\max_\pi V^\pi(b_0)
$$

The value function satisfies the Bellman optimality equation:  

Let $\tau(b, a, o)$ be the updated belief after observing $o$ following action $a$ in belief $b$. Then:

 ```math
 V^{*}(b) = \max_{a \in A} \Bigl[ r(b, a) + \gamma \sum_{o \in \Omega} \Pr(o \mid b, a) V^{*}(\tau(b, a, o)) \Bigr]
 ```

In finite-horizon problems, $V^*(b)$ is piecewise-linear and convex. For infinite-horizon, it can be approximated arbitrarily closely with piecewise-linear convex functions.
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

