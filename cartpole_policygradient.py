"""
The major changes adopted from the REINFORCE policy method is:
1. Subtracting baseline value from Q to counter high gradient variance
2. To avoid the full episode requirement, we perform Bellman equation by unrolling 10 steps ahead by observing that value contribution is very small and negligible(though this may not be work for some other environments)
3. We subtract the entropy of policy gradient method from loss function, punishing agent for being too certain about the action to take
"""
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8

REWARD_STEPS = 10


# A class to create the nn
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


def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val


if __name__ == "__main__":
    # Create the environment, the Tensorboard writer and the network
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg")
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    # Create our Policy Agent passing apply_softmax=True to apply softmax to network output 
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    # Initialze Experience Source that generates trajectories as per the agent (and does much more, such as unrolling etc)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Few helper variables 
    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions, batch_scales = [], [], []

    # Loop through the trajectories generated 
    for step_idx, exp in enumerate(exp_source):
        # Calculate the moving average of the discounted reward which will be our baseline
        # Note that exp.reward will automatically return discounted reward. Check debugginy.py module for more info
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)

        # Append the state, action and the 'scaled reward' obtained
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

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

        if len(batch_states) < BATCH_SIZE:
            continue

        
        # Convert the states, actions and scaled rewards into Tensors.
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        
        optimizer.zero_grad()
        
        # Pass the state value through the nn and apply softmax to get normalized probabilities
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        
        # Obtain the gradient as per the REINFORCE method. We iterate through the batch states to obtain another list
        # THe only difference is how the Q value is obtained by unrolling
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        # Take the mean to obtain the gradient
        loss_policy_v = -log_prob_actions_v.mean()

        # Calculate the entropy value and then add it (actually subtracting as we use '-' in the penultimate equation) to our loss equation
        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_loss_v

        # Perform the backpropagation and perform the gradient ascent step
        loss_v.backward()
        optimizer.step()

        # calc KL-divergence between the new policy and the old policy 
        # The two policies are the one returned by the model before and after the optimization step
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        # Finding the gradient(l2-mean and max) and writing it to tensorboard.
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        bs_smoothed = smooth(bs_smoothed, np.mean(batch_scales))
        entropy = smooth(entropy, entropy_v.item())
        l_entropy = smooth(l_entropy, entropy_loss_v.item())
        l_policy = smooth(l_policy, loss_policy_v.item())
        l_total = smooth(l_total, loss_v.item())

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy, step_idx)
        writer.add_scalar("loss_entropy", l_entropy, step_idx)
        writer.add_scalar("loss_policy", l_policy, step_idx)
        writer.add_scalar("loss_total", l_total, step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)
        writer.add_scalar("batch_scales", bs_smoothed, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
