from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():

	def __init__(self, capacity):
		self.memory = deque([], maxlen = capacity) # initialize dequeue to store transition tuples

	def push(self, *args): # push transition in memory dequeue
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


###################################################

# scratch code to see how Transition works


# memory = ReplayMemory(1000)
# memory.push(1,2,None,4)
# memory.push(1,2,3,4)
# memory.push(1,2,3,4)
# memory.push(1,2,3,4)
# memory.push(1,2,3,4)
# memory.push(1,2,None,4)
# import random
# import torch
# transitions = memory.sample(len(memory))
# batch = Transition(*zip(*transitions))
# print(batch)

# non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,  batch.next_state)))
# non_final_next_state = [s for s in batch.next_state if s is not None]

# print(non_final_mask)
# print(non_final_next_state)
# print(batch.state)
# print(batch.action)
# print(batch.reward)


###############################################






