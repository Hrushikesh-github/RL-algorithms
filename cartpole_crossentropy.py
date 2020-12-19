import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# softmax is not applied to increase the numerical stability of the training process. Rather than calculating softmax and then calculating cross-entropy loss, we can use the PyTorch class nn.CrossEntropyLoss which combines both softmax and cross-entropy in a single, more numerically stable expression

# Define a single episode stored as total undiscounted reward
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
# Define a single step that our agent makes in an episode. It stires the observation from env and what action the agent completed
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# Define a function that generates batches of episodes
def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    # Create a softmax layer that will be used later to convert NN's output to probability distributions of actions, which are then sampled as per probability
    sm = nn.Softmax(dim=1)

    while True:
        # Convert the observations into PyTorch Tensor and pass it to NN
        obs_v = torch.FloatTensor([obs]).cuda()
        act_probs_v = sm(net(obs_v))
        act_probs_v = act_probs_v.cpu()
        act_probs = act_probs_v.data.numpy()[0]
        
        # Choose an action with the probabilities obtained from the nn, perform the step, update the rewards
        action = np.random.choice(len(act_probs), p=act_probs)
        (next_obs, reward, is_done, _) = env.step(action)
        episode_reward += reward

        # Write down the step into the EpisodeStep variable and append it to the series of episode_steps
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)

        # If the episode(i.e task) has been completed, append all the episode_steps into the episode variable, which are appended into the batch variable.
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            
            # If the batch size equals the mentioned batch_size, then yield, else reset the environment, reward and go again
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


# Create a function that filters the episodes whose net reward is less than 75 percentile
def filter_batch(batch, percentile):
    # Get the rewards and a bound based on percentile value. Also obtain the mean of rewards
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for (reward, steps) in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs).cuda()
    train_act_v = torch.LongTensor(train_act).cuda()

    return (train_obs_v, train_act_v, reward_bound, reward_mean)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    #env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]

    # Store the number of actions in the env, based on which our nn is made
    n_actions = env.action_space.n

    # Initialize the network and pass the network to cuda to use gpu
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    net = net.cuda()
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # Initialize the TensorBoard summary writer
    writer = SummaryWriter(comment="-cartpole")

    # loop over the batches
    for (iter_no, batch) in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        (obs_v, acts_v, reward_b, reward_m) = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("{:.0f} loss={:.3f}, reward_mean={:.1f}, rw_bound={:.1f}".format(iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()
