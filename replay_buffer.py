import threading
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.lock = threading.Lock()

    @property
    # Determine whether the buffer is full
    def full(self):
        with self.lock:
            return self.current_size == self.size

    # Randomly sampling a batch of data from a buffer
    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        # current state o, current auxiliary goal g
        # next state o_2, next auxiliary goal g_2 --> calculate the target Q value for more stable training.
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # Sample a batch of training samples from buffers.
        # Sampling results are saved in transitions.
        transitions = self.sample_transitions(buffers, batch_size)

        # r: reward
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key
        return transitions

    # Storing a complete episode into the replay buffer
    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
            T: time steps
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            # Keeps track of the number of transitions that have been stored, 
            # making control of the state in the replay buffer
            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0
    
    # Gets an index that can be used to store a new episode or transition.
    # When storing episodes into the replay buffer, we need to determine the location to store them.
    def _get_storage_idx(self, inc=None):
        # Setting inc to 1, i.e. storing a separate episode
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        
        # go consecutively until you hit the end, and then go randomly.
        
        # If there is enough space in the current replay buffer to store the new episodes, 
        # then allocate inc indexes consecutively starting from the current one, which indicate where the new episodes will be stored. 
        # This ensures that the existing space is filled before storing the new episode from the beginning of the replay buffer.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
            
        # If the current replay buffer does not have enough space for inc new episodes, 
        # the remaining inc indexes are randomly assigned to episodes already in the replay buffer to overwrite the old experience data. 
        # This ensures that the experience data in the replay buffer is kept fresh.
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        
        # If the replay buffer is full, 
        # the inc index is randomly assigned to the entire replay buffer, 
        # overwriting the old experience data.    
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
