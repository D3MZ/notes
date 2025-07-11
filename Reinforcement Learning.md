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

| #  | Method                                   | Category                       | Update Rule / Formula |
|----|------------------------------------------|--------------------------------|------------------------|
| 1  | Reinforcement Learning                   | Framework                      | $`J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T \gamma^t r_t \right]`$ |
| 2  | Deep Q-Learning (DQN)                    | Value-based, Off-policy        | $`L(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right]^2`$ |
| 3  | Prioritized Experience Replay (PER)      | Sampling strategy              | $`p_i \propto |\delta_i|^\alpha`, $\delta_i = r + \gamma \max_{a'} Q(s', a') - Q(s, a)`, weighted by $w_i = \left( \frac{1}{N p_i} \right)^\beta`$ |
| 4  | Soft Q-Learning                          | Entropy-regularized, Off-policy | $`Q(s, a) \leftarrow r + \gamma \mathbb{E}_{s'} [ V(s') ]`, $V(s) = \alpha \log \sum_a \exp ( Q(s, a)/\alpha )`$ |
| 5  | REINFORCE                                | Policy-gradient, On-policy     | $`\nabla_\theta J(\theta) = \mathbb{E} [ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t ]`, $G_t = \sum_{\tau=t}^T \gamma^{\tau - t} r_\tau`$ |
| 6  | Proximal Policy Optimization (PPO)       | Policy-gradient, On-policy     | $`L^{\mathrm{CLIP}} = \mathbb{E} [ \min ( r_t(\theta) \hat{A}_t, \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t ) ]`$ |
| 7  | Lagrange-Constrained PPO                 | Safe RL                        | $`L = \mathbb{E} [ L^{\mathrm{CLIP}} - \lambda (c - d) ]`, constraint: $\mathbb{E}[c] \le d`$ |
| 8  | Advantage Actor-Critic (A2C)             | Actor-critic, On-policy        | Actor: $`\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi(a \mid s) A(s,a)]`$, Critic: $`\min (r + \gamma V(s') - V(s))^2`$ |
| 9  | Deep Deterministic Policy Gradient (DDPG)| Actor-critic, Off-policy       | Critic: $`\min ( r + \gamma Q(s', \mu(s')) - Q(s, a) )^2`$, Actor: $`\nabla_\theta J = \mathbb{E}[\nabla_a Q(s,a) \nabla_\theta \mu(s)]`$ |
| 10 | Twin Delayed DDPG (TD3)                  | Actor-critic, Off-policy       | $`y = r + \gamma \min_i Q_i(s', \mu(s'))`$, with clipped noise and delayed actor update |
| 11 | Soft Actor-Critic (SAC)                  | Entropy-regularized, Off-policy| Critic: $`\min (Q(s,a) - y)^2`, $y = r + \gamma \mathbb{E}_{a'}[Q(s', a') - \alpha \log \pi(a'|s')]`$<br>Actor: $`\min \mathbb{E}[\alpha \log \pi(a|s) - Q(s,a)]`$ |
| 12 | Behavioral Cloning                       | Imitation, Supervised          | $`\min_\theta \sum_i -\log \pi_\theta(a_i \mid s_i)`$ |
| 13 | GAIL (On-policy / Off-policy)            | Adversarial Imitation          | $`\min_\pi \max_D \mathbb{E}_\pi[\log D(s,a)] + \mathbb{E}_{\pi_E}[\log (1 - D(s,a))] - \lambda H(\pi)`$ |
| 14 | Adversarial Value Moment IL (AdVIL)      | Adversarial Imitation          | $`\mathbb{E}_\pi[\phi(s,a)] = \mathbb{E}_{\pi_E}[\phi(s,a)]`$ (match feature moments via adversary) |
| 15 | Adversarial Reinforced IL (AdRIL)        | Adversarial Imitation          | Variant of AdVIL with value-based critic in adversarial loop |
| 16 | SQIL                                     | Imitation-augmented RL         | Expert transitions get $`+1`$ reward, agent transitions $`0`$; Q-learning used normally |
| 17 | ASAF (Adversarial Soft Advantage Fitting)| Adversarial Imitation          | Match expert advantage via adversarial fitting |
| 18 | Inverse Q-Learning (IQLearn)             | Inverse RL                     | $`\pi_E(a \mid s) \propto \exp(Q(s,a)/\tau)`$ (recover Q from expert policy) |
| 19 | Batch SAC                                | Offline RL                     | SAC applied to fixed dataset with added conservatism |
| 20 | Conservative Q-Learning (CQL)            | Offline RL                     | $`\min_\theta \mathbb{E}_{\mathcal{D}}[(Q(s,a) - y)^2] + \alpha (\mathbb{E}_{a \sim \pi}[Q(s,a)] - \mathbb{E}_{a \sim \mathcal{D}}[Q(s,a)])`$ |
| 21 | Robust Adversarial RL (RARL)             | Adversarial RL                 | $`\max_\pi \min_\xi \mathbb{E}_{\pi, \xi} [ \sum_t r(s_t, a_t, \xi_t) ]`$, $\xi$ is adversarial perturbation |
