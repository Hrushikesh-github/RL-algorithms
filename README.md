# RL algorithms such as Cross Entropy, Sarsa, Expected Sarsa, Q-learning, DQN, Policy gradient, REINFORCE, Actor critic and Hill Climbing on Environments: [Cartpole](https://gym.openai.com/envs/CartPole-v0/), [FrozenLake](https://gym.openai.com/envs/FrozenLake-v0/), [Taxi](https://gym.openai.com/envs/Taxi-v2/) and [BlackJack](https://gym.openai.com/envs/Blackjack-v0/) 

More about DQN and it's variants in [my other repository](https://github.com/Hrushikesh-github/DQN-and-Extensions)

### Cross-Entropy
Cross Entropy is a on-policy method. We play N-episodes using our current model, then smaple only those episodes based on a criteria and then train using these episodes on our Neural Network, taking observations as input and desired action as output. Cross Entropy on Cartpole gives good and quick results. Reward_bound here is the boundary that we use to winnow episodes.
![Screenshot from 2020-12-19 22-55-47](https://user-images.githubusercontent.com/56476887/102695542-0ee98780-424e-11eb-9ea0-6cef1c9ccfaf.png)
![Screenshot from 2020-12-19 22-55-57](https://user-images.githubusercontent.com/56476887/102695550-1315a500-424e-11eb-97ff-b69bea26bffd.png)
![Screenshot from 2020-12-19 22-56-15](https://user-images.githubusercontent.com/56476887/102695554-1577ff00-424e-11eb-9e9e-b5f98f03bc37.png)

However using Cross Entropy with the same parameters on slighter tougher environment of Frozen Lake doesn't give us solution. The major reason for this is the reward system. In cartpole we obtain reward of +1/-1 for every step, while in Frozen Lake, reward is only obtained once the episode finishes. Failed episodes dominate at the start of training and there are high chances of training not improving. This is because our neural network learns predominantly from bad experiences.

![Screenshot from 2020-12-19 22-57-11](https://user-images.githubusercontent.com/56476887/102695556-17da5900-424e-11eb-96c4-e39498a2306b.png)
![Screenshot from 2020-12-19 22-48-52](https://user-images.githubusercontent.com/56476887/102695512-db0e6200-424d-11eb-8c46-ad3e25c03ba6.png)
![Screenshot from 2020-12-19 22-49-02](https://user-images.githubusercontent.com/56476887/102695539-0bee9700-424e-11eb-89b1-e3254226e8bb.png)

### Q-learning and V-learning
Sarsa, Expected Sarsa have been implemented on a Jupyter Notebook on the BlackJack Environment. 

Q-learning is an off-policy method. While using Q(s,a) to improve the agent, frozenlake was solved within 20 iterations(trajectories) whereas using the value of state, the environment is solved in 504 iterations. THis difference is because the agent just requires the Q values to solve the environment. However V-values are also powerful especially in the actor-critic family of algorithms. The graphs here are the rewards vs episodes obtained for Q-learning and V-learning respectively

![Screenshot from 2020-12-19 23-10-00](https://user-images.githubusercontent.com/56476887/102696418-1f9cfc00-4254-11eb-80ab-226b7ca31b4f.png)
![Screenshot from 2020-12-19 23-22-15](https://user-images.githubusercontent.com/56476887/102696420-23c91980-4254-11eb-8482-b9be8f59680e.png)
![Screenshot from 2020-12-19 23-22-29](https://user-images.githubusercontent.com/56476887/102696423-26c40a00-4254-11eb-9c1d-e335f01d11bb.png)


### DQN
More about DQN and it's variations in [my other repository](https://github.com/Hrushikesh-github/DQN-and-Extensions)
![Screenshot from 2020-12-19 23-31-22](https://user-images.githubusercontent.com/56476887/102696428-30e60880-4254-11eb-84b9-f6acab674a60.png)
![Screenshot from 2020-12-19 23-32-51](https://user-images.githubusercontent.com/56476887/102696429-33486280-4254-11eb-9150-c308670df3d1.png)
![Screenshot from 2020-12-19 23-33-10](https://user-images.githubusercontent.com/56476887/102696430-35aabc80-4254-11eb-8f1e-abcd6dcfa1e9.png)

### Reinforce
In Reinforce policy gradient algorithm, we train Neural network to minimize the loss function L = - Q(s,a)*log(neural_network(s|a)). This method is similar to cross entropy in few ways, such as we obtain transitions from our current model, for every transition of episode, we calculate the discounted reward, which would be our Q(s,a). Then we obtain the loss and perform gradient descent. This way we don't require explicit exploration. 
![Screenshot from 2020-12-19 23-47-50](https://user-images.githubusercontent.com/56476887/102696478-a9e56000-4254-11eb-86e5-e916c62b08f5.png)
![Screenshot from 2020-12-19 23-47-59](https://user-images.githubusercontent.com/56476887/102696480-ace05080-4254-11eb-8a02-8cd62ae9d0ea.png)
![Screenshot from 2020-12-19 23-40-15](https://user-images.githubusercontent.com/56476887/102696494-c71a2e80-4254-11eb-8209-bbea816fb832.png)


However, there are various drawbacks such as Full episode requirements, High gradients variance and exploration problem-chances of converging to a local optimal policy(explicit exploration is not required). The High gradients arise because of the variance of taking different. 

### Policy Gradient
The major changes adopted from the REINFORCE policy method is:
1. Subtracting baseline value from Q to counter high gradient variance. Baseline chosen was the moving average reward
2. To avoid the full episode requirement, we perform Bellman equation by unrolling 10 steps ahead by observing that value contribution in future steps is very small and thus can be negligible(though this may not be work for some other environments)
3. We subtract the entropy bonus of policy gradient method from loss function, punishing agent for being too certain about the action to take, to avoid falling in a local maxima. Entropy bonus = sum[p(a|s) * log(p(a|s))] 

![Screenshot from 2020-12-19 23-40-26](https://user-images.githubusercontent.com/56476887/102696497-ced9d300-4254-11eb-813d-98ea8955684d.png)
![Screenshot from 2020-12-19 23-47-50](https://user-images.githubusercontent.com/56476887/102696543-41e34980-4255-11eb-86be-533f63b9e1ec.png)
![Screenshot from 2020-12-19 23-49-42](https://user-images.githubusercontent.com/56476887/102696547-4871c100-4255-11eb-8d3b-a407ce6d18ed.png)
![Screenshot from 2020-12-19 23-51-57](https://user-images.githubusercontent.com/56476887/102696554-50316580-4255-11eb-9d97-5123146289e6.png)

### Gradients 
To visualize the gradients, the finding_gradients_cartpole_pg.py module was made. We visualize the gradients when we include and don't include the baseline values. Red denotes the value of gradients when baseline is not subtracted and blue is when baseline is subtracted from the loss function. We can notice the huge decrease in gradients when baseline is subtracted. And this gets better when we use the actor-ctric algorithm to obtain the ideal value of baseline.

![Screenshot from 2020-12-19 23-54-30](https://user-images.githubusercontent.com/56476887/102696616-a4d4e080-4255-11eb-8dc0-3853c1e5e691.png)
![Screenshot from 2020-12-19 23-54-39](https://user-images.githubusercontent.com/56476887/102696619-a7373a80-4255-11eb-8d5a-022dcd876ec9.png)
![Screenshot from 2020-12-19 23-54-47](https://user-images.githubusercontent.com/56476887/102696622-aa322b00-4255-11eb-9e20-bffb139f7f1a.png)
