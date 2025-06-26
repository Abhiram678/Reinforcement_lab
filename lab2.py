import math
import matplotlib.pyplot as plt
import os
import numpy as np

np.random.seed(0)

class Environment:
    def __init__(self, probs):
        self.probs = probs

    def step(self, action):
        return 1 if (np.random.random() < self.probs[action]) else 0

class Agent:
    def __init__(self, nActions, eps):
        self.nActions = nActions
        self.eps = eps
        self.n = np.zeros(nActions, dtype=np.int64)
        self.Q = np.zeros(nActions, dtype=np.float64)

    def update_Q(self, action, reward):
        self.n[action] += 1
        self.Q[action] += (1 / self.n[action]) * (reward - self.Q[action])

    def get_action(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.nActions)
        else:
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))

def experiment(probs, N_episodes, eps):
    env = Environment(probs)
    agent = Agent(len(env.probs), eps)
    actions, rewards = [], []
    for episode in range(N_episodes):
        action = agent.get_action()
        reward = env.step(action)
        agent.update_Q(action, reward)
        actions.append(action)
        rewards.append(reward)
    return np.array(actions), np.array(rewards)

# Parameters
probs = [0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65]
N_experiments = 50000
N_steps = 500
eps = 0.2
save_fig = True
output_dir = os.path.join(os.getcwd(), "output")

print("Running multi-armed bandits with nActions = {}, eps = {}".format(len(probs), eps))

R = np.zeros((N_steps,))
A = np.zeros((N_steps, len(probs)))

for i in range(N_experiments):
    actions, rewards = experiment(probs, N_steps, eps)
    if (i + 1) % (N_experiments // 100) == 0:
        print("[Experiment {}/{}] n_steps = {}, reward_avg = {:.3f}".format(
            i + 1, N_experiments, N_steps, np.sum(rewards) / len(rewards)))

    R += rewards
    for j, a in enumerate(actions):
        A[j][a] += 1

R_avg = R / np.float64(N_experiments)

# Plot average rewards
plt.plot(R_avg, ".")
plt.xlabel("Step")
plt.ylabel("Average reward")
plt.grid()
plt.xlim([1, N_steps])

if save_fig:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "rewards.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()

# Plot action percentages over time
for i in range(len(probs)):
    A_pct = 100 * A[:, i] / N_experiments
    steps = list(np.array(range(len(A_pct))) + 1)
    plt.plot(steps, A_pct, "-", linewidth=2.5, label="Arm {} ({:.0f}%)".format(i + 1, 100 * probs[i]))

plt.xlabel("Step")
plt.ylabel("Action selection percentage (%)")
leg = plt.legend(loc="upper left", shadow=True)
plt.xlim([1, N_steps])
plt.ylim([0, 100])

# Fixing linewidth for legend lines
for legobj in leg.get_lines():
    legobj.set_linewidth(4)

if save_fig:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "actions.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()
