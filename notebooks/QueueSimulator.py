import numpy as np
import gym
from gym import spaces
import copy

class QueueSimulator(gym.Env):
    def __init__(self, arrival_rates, mean_delay_requirements, queues_finished_timeslots):
        super(QueueSimulator, self).__init__()
        self.arrival_rates = arrival_rates
        self.mean_delay_requirements = mean_delay_requirements
        self.current_timeslot = 1
        self.queues_finished_timeslots = queues_finished_timeslots # Need some reference to this so that I can use it again after reset
        self.queues = copy.deepcopy(queues_finished_timeslots)
        self.scenario = 1
        
        # Graph the total wait times for each packet, and at certain episode intervals, add the wait times to the map
        self.packets_total_wait_times = [[], [], []]
        self.packets_total_wait_times_map = {}
        
        # Action Space is 3, because we have 3 queues to choose from
        self.action_space = spaces.Discrete(3)
        
        # We know the observation space are the range of possible and observable values. This is wait times,
        # so wait times can be 0 or infinity technically.
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([np.inf, np.inf, np.inf]), dtype=np.dtype(int))
   
    def calc_state(self):
        # Use -1 as indicator for packet not arrived. Makes more sense to use something like np.inf, but easier to create q_table with -1 than np.inf
        calc_state = [-1, -1, -1]
        for i, queue in enumerate(self.queues):
            # queue = [4, 7, 10, ...]
            packet_arrived = False
            packet_wait = -1
            if queue:    # Check the packet exists first
                if queue[0] <= self.current_timeslot:
                    packet_arrived = True
                    packet_wait = self.current_timeslot - queue[0]
                    
            # Only update state wait time if there exists a packet indicated by the packet_arrived boolean. Else the state will stay as -1
            if packet_arrived:
                calc_state[i] = packet_wait
        return calc_state

    def step(self, action):
        # First, check how long each queue has been waiting for (this is the initial state)
        current_state = self.calc_state()
        
        # Now calc reward
        # If the current_state is -1 then packet has no arrived, and we DO NOT WANT TO reward the model. We shouldnt reward it for no choice
        if current_state[action] == -1:
            reward = -100
        else:
            # The reward is some arbitrary number divided by the queue's mean_delay, and then multiplied by the queue wait time. 
            # If the queue has waited its mean delay, then it will cancel out the mean_delay denominator and thus reward will be complete 100.
            reward = (100 / self.mean_delay_requirements[action]) * (current_state[action])
            if reward > 100:
                reward = 100
            
        # If best effort queue is chosen, rewrite the reward.
        # Reward when the best effort queue is chosen should be really high, but not as high as when mean_delay_requirement is met (100 reward)
        # 70 is chosen because for PQ1, if it's waited at least 5/6 of its mean delay requirement, that is (100/6)*5 = 83, which would make it
        # prioritised over the best effort reward, which is what we want.
        # Similarly with PQ2, (100/4)*3 = 75, which would also make it more prioritised which is what we want.
        if action == 2:
            if current_state[action] == -1:
                reward = -100
            else:
                reward = 99
        
        # Everytime you transmit a packet, keep track of how long that packet had to wait in packets_total_wait_times to graph later
        # also, only delete if that packet actually exists in the queue (i.e. its arrived, which is only at a certain timeslot onwards)
        if (len(self.queues[action]) > 0 and self.queues[action][0] <= self.current_timeslot):
            self.packets_total_wait_times[action].append(self.current_timeslot - self.queues[action][0])
            del self.queues[action][0]
        
        # Now get new state to send back
        new_state = self.calc_state()
        
        done = False
        if all(len(queue) == 0 for queue in self.queues):
            done = True
        
        self.current_timeslot += 1
        return new_state, reward, done, {}
        
    # Since q_learning expects state reset too, return the calc_state method
    # Also, take episode number here cuz i want to graph the wait_times after X episode intervals
    def reset(self, episode):
        interval = 5000//10
        if episode % interval == 0:
            self.packets_total_wait_times_map.update( {episode: self.packets_total_wait_times} )
        self.current_timeslot = 0
        self.queues = copy.deepcopy(self.queues_finished_timeslots)
        self.packets_total_wait_times = [[], [], []]
        return self.calc_state()
        
    def render(self):
        return self

    # Gets the total wait times from the last episode of the model.
    def calc_average_wait_time(self):
        average_queue_wait = []
        last_episode = list(self.packets_total_wait_times_map.keys())[-1]

        for i, queue in enumerate(self.packets_total_wait_times_map.get(last_episode)):
            average_queue_wait.append(round(sum(queue)/len(queue),2))
        
        return average_queue_wait

    # Manually update the map to put the episode in as an entry
    def update_last_episode_stats(self, episode):
        self.packets_total_wait_times_map.update( {episode: self.packets_total_wait_times} )

 
