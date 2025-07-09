# Reinforcement Learning

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
