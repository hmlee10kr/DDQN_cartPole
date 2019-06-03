import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import gym
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

def display_frames_as_gif(frames):
    """
    displays a list of frames as a gif, with controls
    :param frames:
    :return:
    """
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch=plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])


    anim=animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    anim.save('movie_cartpole_DQN.mp4') #save animation
    display(display_animation(anim, default_mode='loop'))

#use namedtuple in this code, able to access with field name, saved with (key, value) format

from collections import namedtuple

Tr=namedtuple('tr', ('name_a', 'value_b'))
Tr_object=Tr('NAME_a', 100)

print(Tr_object) #print name_a='NAME_a', value_b=100
print(Tr_object.value_b) #print 100

#make namedtuples
from collections import namedtuple
Transition=namedtuple('Transition', ('state', 'action', 'next_state','reward'))

#initialize
ENV='CartPole-v0' #Task name
GAMMA=0.99 #time discount rate
MAX_STEPS=200 #maximum levels in one episode
NUM_EPISODES=500 #maximum of episodes


#Replay Memory Class: saves the experienced data from the mini-batch learning, saves the transitions
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity=CAPACITY   #max of able to be saved memory
        self.memory=[]           #variable for saving transitions
        self.index=0             #index for where to save

    def push(self, state, action, state_next, reward):
        '''save transition=(state, action, state_next, reward) in memory'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  #if memory is not full

        #use a namedtuple called Transition to save as (key, value) format
        self.memory[self.index]=Transition(state, action, state_next, reward)
        self.index=(self.index+1)%self.capacity    #push next saving space one back

    def sample(self, batch_size):
        '''call transitions randomly until batchsize full'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''return number ofsaved transitions'''
        return len(self.memory)

#brain of the agent, implements DQN
#defines Q-functions as deep learning NN
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE =32
CAPACITY=10000

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions=num_actions    #gets number of actions(left, right)

        #memroy to keep track of transition
        self.memory=ReplayMemory(CAPACITY)

        #NN
        self.model==nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)  #print structure of NN

        # choose optimizer
        self.optimizer=optim.Adam(self.model.parameters(), lr=0.0001)
    def replay(self):
        '''Experiance Replay: learn NN's combined weight'''
        #check number fo trnasitions, if smaller than mimbatch: does nothing
        if len(self.memory) < BATCH_SIZE:
            return

        #make mini-batch
        transitions=self.memory.sample(BATCH_SIZE)
        batch=Transition(*zip(*transitions))
        state_batch=torch.cat(batch.state)
        action_batch=torch.cat(batch.action)
        reward_batch=torch.cat(batch.reward)
        non_final_next_states=torch.cat([s for s in batch.next_state if s is not None])

        #calculate Answer Q(s_t, a_t) (with NN)

        self.model.eval()

        state_action_values=self.model(state_batch).gather(1, action_batch)

        #calculate max{Q(s_t, a_t)}, careful if next state is valid or not
        non_final_mask=torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        #initialize with 0
        next_state_values=torch.zeros(BATCH_SIZE)

        #find max Q value in index that have a next state
        next_state_values[non_final_mask]=self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values=reward_batch + GAMMA * next_state_values

        #modify weight
        self.model.train()

        #calculate loss with Huber function
        loss=F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        '''choose action by current state'''
        #e-greedy algo
        epsilon=0.5*(1/(episode+1))

        if epsilon <= np.random.uniform(0,1):
            self.model.eval()
            with torch.no_grad():
                action=self.model(state).max(1)[1].view(1,1)

        else:
            #action is randomly returned
            action=torch.LongTensor([[random.randrange(self.num_actions)]])

        return action
#Cart of cartPole
class Agent:
    def __init__(self, num_states, num_actions):
        '''state and number of actions within the tasks'''
        self.brain=Brain(num_states, num_actions)


    def update_q_function(self):
        '''modify Q-function'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''chooses action'''
        action=self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''save in memory'''
        self.brain.memory.push(state, action, state_next, reward)

#class that implements cartPole

class Enviornment:
    def __init__(self):
        self.env=gym.make(ENV)
        num_states=self.env.observation_space.shape[0]
        num_actions=self.env.action_space.n
        self.agent=Agent(num_states, num_actions)

    def run(self):
        episode_10_list=np.zeros(10)
        complete_episodes=0
        episode_final=False
        frames=[]

        for episode in range(NUM_EPISODES):
            observation=self.env.reset()

            state=observation
            state=torch.from_numpy(state).type(torch.FloatTensor)
            state=torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                if episode_final is True:
                    frames.append(self.env.render(mode='rgb_array'))
                action=self.agent.get_action(state, episode)
                observation_next, _, done, _=self.env.step(action.item())
                if done:
                    state_next=None
                    episode_10_list=np.hstack((episode_10_list[1:], step+1))
                    if step<195:
                        reward=torch.FloatTensor([-1.0])
                        complete_episodes=0
                    else:
                        reward=torch.FloatTensor([1.0])
                        complete_episodes=complete_episodes+1
                else:
                    reward=torch.FloatTensor([0,0])
                    state_next=observation_next
                    state_next=torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next=torch.unsqueeze(state_next,0)

                self.agent.memorize(state, action, state_next, reward)

                self.agent.update_q_function()

                state=state_next

                if done:
                    print('%d Episode: Finished after %d steps : Average of steps in Latest 10 episodes = %.1lf' %(episode, step+1, episode_10_list.mean()))
                    break
            if episode_final is True:
                display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print('10 consecutive episodes success')
                episode_final=True


cartpole_env=Enviornment()
cartpole_env.run()
