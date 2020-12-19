import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


# NN class that outputs logit (instead of softmax probs for better accuracy, softmax can be added when required) 
class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# A Function to calculate the list of q value(discounted reward) for each state the agent has passed through. Pass the rewards list for the whole episode which contains reward for each step
def calc_qvals(rewards):
    # Initialize few helper variables
    res = []
    sum_r = 0.0

    # Loop through the rewards, in the reverse order 
    for r in reversed(rewards):
        # The q value corresponding to the particular state for which reward 'r' is obtained is (r + gamma * sum_of_future rewards as per policy
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    # Initialize the environment, the tensorboard summary writer and network
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    # Initalize our PolicyAgent, passing apply_softmax=True to apply softmax to network output 
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    # Initialze Experience Source that generates trajectories as per the agent (and does much more)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Define few helper variables
    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []

    # Loop through the trajectories generated 
    for step_idx, exp in enumerate(exp_source):
        # Append the state, action and reward obtained
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        # In case we reached the end of the episode 
        if exp.last_state is None:
            # Calculate the q values for each state passed through using the calc_qvals function
            batch_qvals.extend(calc_qvals(cur_rewards))
           
            cur_rewards.clear()
            # Increment the episode counter
            batch_episodes += 1

        # Report and write current metrics to Tensorboard 
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        # Ensure enough # of episodes are present to start training. If not obtain more episodes
        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        # Convert the states, actions and q-values into Tensors.
        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        # Pass the state value through the nn and apply softmax to get normalized probabilities
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)

        # Obtain the gradient as per the REINFORCE method. We iterate through the batch states to obtain another list
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        # Take the mean to calculate the gradient
        loss_v = -log_prob_actions_v.mean()

        # Perform the backpropagation and perform the gradient ascent step
        loss_v.backward()
        optimizer.step()

        # Clear required variables to work on the new batch 
        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    # Close the tensorboard writer
    writer.close()
