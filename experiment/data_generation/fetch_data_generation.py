import gym
import numpy as np


"""Data generation for the case of a single block pick and place in Fetch Env"""

actions = []
observations = []
infos = []


# Execute the data generation process and save the demo dataset to the file
def main():
    env = gym.make('FetchPickAndPlace-v1')
    # Number of iterations of data generation, i.e., 100 demo trajectories generated
    numItr = 100
    initStateSpace = "random"
    env.reset()
    print("Reset!")
    while len(actions) < numItr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)


    fileName = "data_fetch"
    fileName += "_" + initStateSpace
    fileName += "_" + str(numItr)
    fileName += ".npz"

    np.savez_compressed(fileName, acs=actions, obs=observations, info=infos) # save the file


# Simulate the movement of a robotic arm from its current state to a target position, 
# and record the action, observation and environment information
def goToGoal(env, lastObs):
    
    # goal: the target position of the robotic arm pick and place task.
    goal = lastObs['desired_goal']
    # the position of the object held by the robotic arm.
    objectPos = lastObs['observation'][3:6]
    # the position of the object that the agent needs to grasp and place
    object_rel_pos = lastObs['observation'][6:9]
    # Action, observation and environmental information for the current demo trajectory
    episodeAcs = []
    episodeObs = []
    episodeInfo = []

    object_oriented_goal = object_rel_pos.copy()
    # first make the gripper go slightly above the object on the z-axis
    object_oriented_goal[2] += 0.03 

    # count the total number of timesteps
    timeStep = 0 
    episodeObs.append(lastObs)

    # Pick
    while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03

        for i in range(len(object_oriented_goal)):
            # The robotic arm is controlled to perform an action to move toward the target
            # and multiplying by 6 is to adjust the amplitude of the action.
            action[i] = object_oriented_goal[i]*6

        # open the gripper for picking objects
        action[len(action)-1] = 0.05

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

   # Similar to the first while loop, 
   # but after picking the object, continue to control the robot arm toward the target
    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(object_rel_pos)):
            action[i] = object_rel_pos[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    # After placing the object, continue to control the robotic arm toward the target.
    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0]
        for i in range(len(goal - objectPos)):
            action[i] = (goal - objectPos)[i]*6

        action[len(action)-1] = -0.005

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

    # Limit the number of steps in the demo trajectory to avoid infinite loops. 
    # Keep the gripper closed and do not perform other actions.
    while True: #limit the number of timesteps in the episode to a fixed duration
        env.render()
        action = [0, 0, 0, 0]
        action[len(action)-1] = -0.005 # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeObs.append(obsDataNew)

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

        if timeStep >= env._max_episode_steps: break

    actions.append(episodeAcs)
    observations.append(episodeObs)
    infos.append(episodeInfo)


if __name__ == "__main__":
    main()
