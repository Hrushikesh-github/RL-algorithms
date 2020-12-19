# Q-learning for Frozen Lake
import numpy as np
import collections
import gym
from tensorboardX import SummaryWriter

steps = 100
GAMMA = 0.9
ENV_NAME = "FrozenLake-v0"
#ENV_NAME = "FrozenLake8x8-v0"
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        # initialise the environment and reset the state
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()

        # define the data structures that will be used
        # reward_table is dictionary with composite key "source_state", "action", "target_state". Value obtained is reward
        # transition_table is dictionary keeping counters of the transitions. key is compositie "state", "action". Value is another dictionary that maps target state with count of times that we have seen it
        # value_table maps state into it's calculated value of the state
        # The advantage of using defaultdict is that we don't need to initialize the dict keys
        self.reward_table = collections.defaultdict(float)
        self.transition_table = collections.defaultdict(collections.Counter)
        self.value_table = collections.defaultdict(float)

        # create a method that will play n random steps. This is required to populate reward and transition tables
    def play_n_random_steps(self, count):
        for _ in range(count):
            # obtain an action from the action space of the environment and perform the action
            action = self.env.action_space.sample()
            (new_state, reward, is_done, _) = self.env.step(action)
            
            # Update the reward and transition table
            self.reward_table[(self.state, action, new_state)] = reward
            self.transition_table[(self.state, action)][new_state] += 1
            
            # Update the state to the new_state and let the loop run. If episode ends, reset the environment
            self.state = self.env.reset() if is_done else new_state


    # Perform the value iteration algorithm and update the value table of best action for the current state
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transition_table[(state, action)]
                total = sum(target_counts.values())
                for (tgt_state, count) in target_counts.items():
                    reward = self.reward_table[(state, action, tgt_state)]
                    best_action = self.select_best_action(tgt_state)
                    val = reward + GAMMA * self.value_table[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.value_table[(state, action)] = action_value


    # For a given state, find the best action ( it is assumed all the actions in action spaces have a value in transition table, otherwise we get error, thus enough steps must be taken to ensure non zero/non existant number )
    def select_best_action(self, state):
        (best_action, best_value) = (None, None)
        for action in range(self.env.action_space.n):
            action_value = self.value_table[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_action

    # A method that plays one full episode, taking the best action and returns the total reward
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_best_action(state)
            (new_state, reward, is_done, _)  = env.step(action)
            self.reward_table[(state, action, new_state)] = reward
            self.transition_table[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

        
if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    # NOTE that there are two environments now, test_env and env initialized by agent. test_env is required so that we don't reset the main environment which is used to gather random data
    writer = SummaryWriter(comment="-v-iteration")
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        # Perform 100 random steps to fill our reward and transition tables with fresh data and then run value iteration over all states.
        agent.play_n_random_steps(100)
        agent.value_iteration()

        # Using value table as our policy, we play the test episodes.
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)

        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated {:.3f} -> {:.3f}".format(best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in {:.0f} iterations!".format(iter_no))
            break
    writer.close()


