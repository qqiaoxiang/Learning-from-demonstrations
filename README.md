# Learning-from-demonstrations

## Hindsight Experience Replay 
HER is used to enhance sample efficiency in reinforcement learning. It trains agents more efficiently by introducing additional target samples in the replay buffer and reutilizing the failure experience to achieve the goal.


### Interpretation of code
1. ddpg.py
   
2. her.py
   Hindsight Experience Replay (HER) was used to improve the efficiency of the sample data.
   - def learn(*, network, env, total_timesteps, ...)
     It is the core part of the training, initiating the environment, the policy network, etc. and then calling the function train()to train.
   - def train(*, policy, rollout_worker, evaluator,..., demo_file, **kwargs)
     Train a policy network for reinforcement learning in a distributed environment. It generates trajectory data by interacting with the environment and then trains in batches with these data, while logging and saving the trained models.

3. 
  
  
4. actor_critic.py
   It implements an Actor-Critic network and training. The policy network (Actor) generates actions, and the Q-value network (Critic) evaluates the Q-value for a given combination of observation, goal and action. These networks were utilised to train a RL algorithm to maximise the cumulative rewards of the policy.

5. normalizer.py
   Two classes, Normalizer and IdentityNormalizer, are implemented to perform ‘Normalisation’ on the input data. 
   For Normalizer, normalise the input data to an approximate standard normal distribution, while supporting data synchronisation in a multi-process environment. It improves the training efficiency and stability of deep learning models. 
   For IdentityNormalizer, mapping the input data to a constant is simple. It divides the input data by a predefined standard deviation or multiplying the standardised data by the standard deviation to achieve data reduction.






   
