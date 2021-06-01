import numpy as np
import gym
from gym import spaces
import copy

# Same as the QueueSimulator class, except the state now has to consider what its current queue is as well (so the calc_state has changed)
class QueueSimulatorScenario2(gym.Env):
    def __init__(self, arrival_rates, mean_delay_requirements, queues_finished_timeslots):
        super(QueueSimulatorScenario2, self).__init__()
        self.arrival_rates = arrival_rates
        self.mean_delay_requirements = mean_delay_requirements
        self.current_timeslot = 1
        self.queues_finished_timeslots = queues_finished_timeslots # Need some reference to this so that I can use it again after reset
        self.queues = copy.deepcopy(queues_finished_timeslots)
        self.scenario = 2
        
        # Graph the total wait times for each queue, and at certain episode intervals, add the wait times to the map
        self.queues_total_wait_times = [[], [], []]
        self.queues_total_wait_times_map = {}

        # Used when we look at state and if we need to add an additional +1 to the timeslots.
        self.previous_queue = -1
        self.total_queue_switches = 0
        
        # Action Space is 3, because we have 3 queues to choose from
        self.action_space = spaces.Discrete(3)
        
        # We know the observation space are the range of possible and observable values. This is wait times,
        # so wait times can be 0 or infinity technically.
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([np.inf, np.inf, np.inf]), dtype=np.dtype(int))
   
    # Get total wait times for each queue (so how long a queue has been waiting for on a packet to send).
    # Remember, gets how long EACH packet has been waiting, so can be quite large since its every packet and not just the first.
    def calc_state(self, current_queue):
        # Use -1 as indicator for packet not arrived. Makes more sense to use something like np.inf, but easier to create q_table with -1 than np.inf
        calc_state = [-1, -1, -1, current_queue]
        for i, queue in enumerate(self.queues):
            queue_total_wait = 0
            packet_arrived = False
            for packet in queue:
                if packet <= self.current_timeslot:
                    queue_total_wait += (self.current_timeslot - packet)
                    packet_arrived = True
                    
            # Only update state wait time if there exists a packet indicated by the packet_arrived boolean. Else the state will stay as -1
            if packet_arrived:
                calc_state[i] = queue_total_wait
        return calc_state

    def step(self, action):
        # First, check how long each queue has been waiting for (this is the initial state)
        current_state = self.calc_state(self.previous_queue)
        
        # Now calc reward
        # If the current_state is -1 then packet has no arrived, and we DO NOT WANT TO reward the model. We shouldnt reward it for no choice
        if current_state[action] == -1:
            reward = 0
        else:
            # The reward is some arbitrary number divided by the queue's mean_delay, and then multiplied by the packet wait time. 
            # If the packet has waited its mean delay, then it will cancel out the mean_delay denominator and thus reward will be complete 100.
            reward = (100 / self.mean_delay_requirements[action]) * (self.current_timeslot - self.queues[action][0])
            
        # Reward when the best effort queue is chosen should be really high, but not as high as when mean_delay_requirement is met (100 reward)
        if action == 2:
            reward = 90
        
        # Everytime you transmit a packet, keep track of how long that packet had to wait in queues_total_wait_times to graph later
        # also, only delete if that packet actually exists in the queue (i.e. its arrived, which is only at a certain timeslot onwards)
        if (len(self.queues[action]) > 0 and self.queues[action][0] <= self.current_timeslot):
            self.queues_total_wait_times[action].append(self.current_timeslot - self.queues[action][0])
            del self.queues[action][0]
        
        # Now get new state to send back
        new_state = self.calc_state(action)
        
        done = False
        if all(len(queue) == 0 for queue in self.queues):
            done = True
        
        if self.previous_queue != action and self.previous_queue != -1:
            self.current_timeslot += 1
            self.total_queue_switches += 1
        self.current_timeslot += 1
        self.previous_queue = action
        return new_state, reward, done, {}
        
    # Since q_learning expects state reset too, return the calc_state method
    # Also, take episode number here cuz i want to graph the wait_times after X episode intervals
    def reset(self, episode):
        interval = 5000//10
        if episode % interval == 0:
            self.queues_total_wait_times_map.update( {episode: self.queues_total_wait_times} )
        self.current_timeslot = 0
        self.queues = copy.deepcopy(self.queues_finished_timeslots)
        self.queues_total_wait_times = [[], [], []]
        self.previous_queue = -1
        self.total_queue_switches = 0
        return self.calc_state(-1)
        
    def render(self):
        return self
