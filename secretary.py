import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class secretary(gym.Env):
    '''
    Secretary environment:

    Actions:
    Type: Discrete(3)
    Num    Action
    0         Continue
    1         Halt
    2         Reverse 1 step back with a loss

    Observation:
    Type: Box(4)
    Num Observation
    0       scaled time [0, 1)
    1       max score of the candidates seen till now [0, 1]
    2       the score of the current candidate
    3       the score of the previous candidate
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, N=20):
        super(secretary, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(4,), dtype=np.float32)
        # The number of steps till end of the episode
        self.N = N
        # The current time step (time < N)
        self.time = 0
        # state
        self.state = None
        self.seed()
        self.steps_beyond_done = None

    def seed(self, seed=None):
        '''
        create the random generator

        copy-pasted from the cartpole
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        score = self.np_random.uniform(low=0, high=1)
        self.state = np.array([0, 0, score, score])
        self.time = 0
        self.steps_beyond_done = None
        return np.array(self.state)

    def backtrack_loss_function(self, score):
        return 0.2*score

    def step(self, action):
        '''
        one step in secretary problem

        it stops if the action is 1 or 2 and continues if the action is 0

        argument:
        action -- the chosen action

        returns:

        observation -- the new state
        done -- if the process is terminated. The process terminates if the actions is 1
        reward -- the reward is 0 if the process continues and the reward is the state if the halt action is chosen.
        info -- forget about it!
        '''
        _, max_score, current_score, previous_score = self.state

        # increase the internal clock by one step
        self.time += 1

        if action == 0:
            # the new candidate
            the_new_score = np.random.rand()

            # the new state
            self.state[0] = self.time / (self.N + 0.0)
            self.state[1] = np.max([max_score, current_score])
            self.state[2] = the_new_score
            self.state[3] = current_score

            # reward and done for this action
            reward = 0
            done = False
            info = {'msg': 'next candidate!'}

        elif (action == 1 or self.time == self.N or (action == 2 and self.time == 0)):
            # the new state after this action
            self.state[0] = self.time / (self.N + 0.0)

            # reward and done after this action.
            reward = self.state[2]
            done = True
            info = {'msg': 'the current score is picked or we reached to the last candidate!'}

        elif (action == 2 and self.time >= 1):
            # the new state after this action
            self.state[0] = self.time / (self.N + 0.0)

            # reward and done after this action.
            reward = self.state[3] - self.backtrack_loss_function(self.state[3])
            done = True
            info = {'msg': 'the previous score is picked or we reached to the last candidate!'}


        # fully observable system
        observation = self.state

        return observation, reward, done, info

    def render(self, mode='human'):
        '''
        not implemented
        '''
        if mode == 'text':
            print(self.state)

    def close(self):
        print('Good Bye!')