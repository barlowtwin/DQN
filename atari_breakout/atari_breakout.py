import gym
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F 
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

import gym
env  = gym.make('Breakout-v0')#.unwrapped


# Deep Q Network

h,w,c = 210, 160, 3
num_actions = env.action_space.n

# input is B x 3 x 210 x 160
# output is B x 4   (4 actions)

class DQN(nn.Module):

	def __init__(self, h, w, num_actions):
		super(DQN, self).__init__()

		self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
		self.bn1   = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size = 2, stride = 2)
		self.bn2   = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
		self.bn3   = nn.BatchNorm2d(64)

		# def getting input dim forthe follwing linear layer
		def conv2d_size_out(size, kernel_size, stride, padding):
			return (size - kernel_size - 2 * padding) // stride + 1

		convh = conv2d_size_out(h, 8, 4, 0) # for conv1
		convh = conv2d_size_out(convh, 2, 2, 0) # for conv2
		convh = conv2d_size_out(convh, 3, 1, 0) # for conv3

		convw = conv2d_size_out(w, 8, 4, 0)
		convw = conv2d_size_out(convw, 2, 2, 0)
		convw = conv2d_size_out(convw, 3, 1, 0)

		conv_out_dim = convh * convw * 64

		self.linear1 = nn.Linear(conv_out_dim, 512)
		self.linear2 = nn.Linear(512, num_actions) # 4 actions

	def forward(self, x):

		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = F.relu(self.bn3(self.conv3(out)))
		out = out.view(out.size(0), -1)
		out = F.relu(self.linear1(out)) # Batch x 512
		out = self.linear2(out) # Batch x 4
		return out # Batch x 4

if torch.cuda.is_available():
	device = torch.device("cuda")
else :
	device = torch.device("cpu")


# we have a target model and a Q model. Q model makes
# prediction of Q values which are used to make a action.

# we have target model for prediction of future rewards.
# the weights of target model are updated every 1000 steps
# thus when the loss between Q-values is calculated the target
# Q value is stable. 

policy_net = DQN(h, w, num_actions).to(device) # for predicting q values
target_net = DQN(h, w, num_actions).to(device) # target model
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr = 0.00025)
criterion = nn.HuberLoss()


# RL parameters

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 10000



# experienc replay buffers

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# number of frames to take random actions before choosing actions
# based on policy
epsilon_random_frames = 50000

# number of frames after which exploration stops
epsilon_greedy_frames = 1000000

# maximum replay length
max_memory_length = 100000

# train model after 4 actions
update_after_actions = 4

# how often to update target network
update_target_network = 10000



# Algorithm for Agent in the environment

while True: # run until solved
	
	state = np.array(env.reset())
	state = torch.tensor(state, dtype = torch.float32) # float 32 matches with network tensor
	state = state.permute((2,0,1)).unsqueeze(0).to(device) # 1 x c x h x w
	episode_reward = 0     

	for timestep in range(1, max_steps_per_episode): # 10000

		env.render()
		frame_count += 1

		if frame_count < epsilon_random_frames or epsilon > random.uniform(0,1) :
			# take random action
			action = np.random.choice(num_actions)

		else : # take action based on q-values

			q_values = policy_net(state) # 1 x 4
			q_values = q_values.squeeze(0) # 4

			# index of the max q_value is the action to be taken
			action = torch.argmax(q_values).cpu().numpy()



		# decaying of probability to take random action
		epsilon -= epsilon_interval / epsilon_greedy_frames
		epsilon = max(epsilon, epsilon_min)

		# taking the selected action in the env
		state_next, reward, done, _ = env.step(action)
		if done == False :
			done = 0
		else :
			done = 1

		# h x w x c -> c x h x w
		state_next = torch.tensor(state_next, dtype = torch.float32) # h x w x c
		state_next = state_next.permute((2,0,1)) # c x h x w
		episode_reward += reward

		# saving states, next states, actions and rewards in buffer
		action_history.append(int(action))
		state_history.append(state.squeeze(0).cpu().numpy())
		state_next_history.append(state_next.cpu().numpy()) # tensor is appended of shape # c x h x w
		done_history.append(done)
		rewards_history.append(reward)



		# if next step selects action from policy then current state needs
		# to be in the shape with batch dimension at front. Therefore, making
		# a change here.
		state = state_next.unsqueeze(0).to(device) # 1 x c x h x w

		if frame_count % update_after_actions == 0 and len(done_history) > batch_size :

			# getting indices of samples from buffer
			indices = np.random.choice(range(len(action_history)), size = batch_size)

			# extracting from buffer from the generated indices
			state_sample      = np.array([state_history[i] for i in indices]) # b x c h x w
			state_next_sample = np.array([state_next_history[i] for i in indices]) # b x c x h x w
			rewards_sample    = torch.tensor([rewards_history[i] for i in indices], dtype = torch.float32)
			action_sample     = [action_history[i] for i in indices]
			done_sample       = [done_history[i] for i in indices]

			rewards_sample = rewards_sample.reshape((batch_size, 1)).to(device)

			# converting state_sample to tensor from numpy
			state_sample = torch.tensor(state_sample, dtype = torch.float32).to(device)
			state_next_sample = torch.tensor(state_next_sample, dtype = torch.float32).to(device)

			future_rewards = target_net(state_next_sample).max(1)[0].detach() # batch_size 
			expected_q_values = future_rewards.reshape((batch_size, 1)) +  (gamma * future_rewards).reshape((batch_size,1)) # batch_size x 1

			# now, assigning expected_q_value as -1 to states where done = 1.

			# convert done_sample to tensor and store it on gpu
			done_sample = torch.tensor(done_sample, dtype = torch.float32).to(device)
			expected_q_values = expected_q_values *  (1 - done_sample).reshape((32,1)) + done_sample.reshape((32,1))

			action_sample = torch.tensor(action_sample).reshape((32,1)).to(device)
			q_values = policy_net(state_sample).gather(1, action_sample)
			

			loss = criterion(q_values, expected_q_values)

			# update weights
			optimizer.zero_grad()
			loss.backward()
			for param in policy_net.parameters():
				param.grad.data.clamp(-1,1)
			optimizer.step()

			if frame_count % update_target_network == 0 :
				target_net.load_state_dict(policy_net.state_dict())
				print("running_reward : " + str(reward) + "episode count : " + str(episode_count) + "frame count : " + str(frame_count))


			# clearing some buffer to make space for new buffer

			if len(rewards_history) > max_memory_length :
				del rewards_history[:1]
				del state_history[:1]
				del state_next_history[:1]
				del action_history[:1]
				del done_history[:1]

			if done: # stop the current episode if task terminates
				break

		# update running reward to check condition for solving
		episode_reward_history.append(episode_reward)
		print(episode_reward)
		if len(episode_reward_history) > 100 :
			del episode_reward_history[:1]
		running_reward = np.mean(episode_reward_history)

		episode_count += 1

		if running_reward > 40 :
			print("solved at episode : " + str(episode_count))
			break









			
			
















			





































		







