import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import gym
from utils import ReplayMemory, Transition
import numpy as np
import random
import math
from itertools import count


import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()



import warnings
warnings.filterwarnings('ignore')

##################################################################

# setup env

# Getting input from the environment

env = gym.make('CartPole-v0').unwrapped



transform = transforms.Compose([transforms.ToPILImage(),
	transforms.Resize(40, interpolation = Image.CUBIC),
	transforms.ToTensor()])

def get_cart_location(screen_width):
	
	world_width = env.x_threshold * 2 
	scale = screen_width / world_width
	return int(env.state[0] * scale + screen_width/2.0)

def get_screen():

	screen = env.render(mode = 'rgb_array').transpose((2,0,1))
	_, screen_height, screen_width = screen.shape
	# screen_width is width of the frame
	# screen_height is height of frame
	screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
	view_width = int(screen_width * 0.6)
	cart_location = get_cart_location(screen_width)
	if cart_location < view_width // 2:
		slice_range = slice(view_width)
	elif cart_location > (screen_width - view_width // 2):
		slice_range = slice(-view_width, None)
	else:
		slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
	screen = screen[:, :, slice_range]

	screen = np.ascontiguousarray(screen, dtype = np.float32)/255
	screen = torch.from_numpy(screen)
	screen = transform(screen).unsqueeze(0) # add a dimension for batch to data


	return screen
####################################################################

class DQN(nn.Module):

	def __init__(self, h, w, output_size):
		super(DQN, self).__init__()

		self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2)
		self.bn1   = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
		self.bn2   = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size = 5, stride = 2)
		self.bn3   = nn.BatchNorm2d(32)

		# getting input dim for linear layers
		def conv2d_size_out(size):
			kernel_size = 5
			stride = 2
			padding = 0
			return (size - kernel_size - 2 * padding) // stride + 1

		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		linear_layer_input_dim = convh * convw * 32

		self.linear_out = nn.Linear(linear_layer_input_dim, output_size)

	def forward(self, x):

		x = x.to(device)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.linear_out(x.view(x.size(0), -1))


####################################################################################

# initialization of parameters and model

batch_size = 64
gamma = 0.7
eps_start = 0.9 # epsilon gradaully decreases
eps_end = 0.05
eps_decay = 200
target_update  = 10

if torch.cuda.is_available():
	device = torch.device("cuda")
else : 
	device = torch.device("cpu")

# initializing screen just to get the dimensions to initialize the model
env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# number of actions
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr = 0.001)
memory = ReplayMemory(1000)
episode_duration = []

steps_done = 0

##############################################################################

def select_action(state):
	global steps_done
	eps_sample = random.random()
	eps_threshold = eps_end + (eps_start - eps_end) * math.exp(- 1. * steps_done / eps_decay)
	steps_done += 1
	#print(eps_threshold)
	#print(steps_done)

	if eps_sample > eps_threshold :
		with torch.no_grad():
			return policy_net(state).max(1)[1].view(1,1) # picking action with larger expected reward
	else :
		return torch.tensor([random.randrange(n_actions)], device = device, dtype = torch.float)



def optimize_model():

	if len(memory) < batch_size:
		return

	transitions = memory.sample(batch_size)
	batch = Transition(*zip(*transitions))

	# computing mask for terminating states
	non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device = device, dtype = torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Q(s_t,a), action that would be taken according to the policy
	action_batch = action_batch.reshape((batch_size, 1))
	state_action_values = policy_net(state_batch).gather(1, action_batch)


	# V(s_t+1)
	next_state_values = torch.zeros(batch_size, device = device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

	# expected Q values
	expected_state_action_values = (next_state_values * gamma) + reward_batch

	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	#optimize model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1,1)
	optimizer.step()


#######################################################################

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_duration, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


#########################################################################

num_episodes = 2000
for i_episode in range(num_episodes):

	env.reset()
	last_screen = get_screen()
	current_screen = get_screen()
	state = current_screen - last_screen

	for t in count():

		action = select_action(state)
		action = int(action.item()) # converting tensor to int
		_, reward, done, _ = env.step(action)
		reward = torch.tensor([reward], device = device)

		# observe new state

		last_screen = current_screen
		current_screen = get_screen()

		if not done:
			next_state = current_screen - last_screen
		else :
			next_state = None

		action = torch.tensor(action).reshape((1,1)).to(device) 
		memory.push(state, action, next_state, reward)

		state = next_state

		# optimize policy network
		optimize_model()
		if done :
			episode_duration.append(t+1)
			plot_durations()
			break

	if i_episode % target_update == 0:
		target_net.load_state_dict(policy_net.state_dict())

print('complete')
env.render()
env.close()
plt.ioff()
plt.show()















