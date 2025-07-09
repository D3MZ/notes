# Reinforcement Learning

## Terms 
|  | **Offline RL** | **Online RL** |
|:-:|:-:|:-:|
| **Data source** | Fixed dataset of (state, action, reward, next state) | Live interaction with environment |
| **Learning** | Learns policy/value from the static dataset | Learns while collecting new data |
| **Interaction** | No further interaction allowed | Environment queried continuously |
| **Exploration** | Impossible | Essential (e.g., ε-greedy, UCB) |
| **Stability** | Safer, but requires careful distribution handling | Potentially unstable but more adaptive |

Bootstrapping uses an estimate of future value to update the current estimate. Central to **Temporal Difference (TD) learning**.
\text{Target} = r_t + \gamma \cdot \underbrace{V(s_{t+1})}_{\text{bootstrapped estimate}}
