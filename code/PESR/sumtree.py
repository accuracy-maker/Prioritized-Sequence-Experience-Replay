class SumTree:
    def __init__(self,data_size):
        """
        : param data_size: the number of transitions
        """
        
        self.nodes = [0] * (2 * data_size - 1) # the whole tree
        self.data = [None] * data_size # only include the child data
        
        self.size = data_size
        self.count = 0
        self.real_size = 0
        
        
    @property
    def total_priority(self):
        return self.nodes[0] # the total priority stored in root node
    
    def add(self,priority,data):
        """
        Add an experience with its priority (value) to the tree.
        If the tree is full, it will start overwriting the oldest data.
        
        :param value: The priority value of the experience.
        :param data: The experience data to store.
        
        """
        self.data[self.count] = data # add a transition to child buffer
        self.update(self.count,priority)
        
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)
        
    def update(self,data_idx,priority):
        """
        when a node is added in the tree
        the value of it and its' parent will be updated
        
        :param data_idx: index of buffer or self.data
        :param priority: the priority of the data_idx
        """
        # convert data_idx to tree_idx
        tree_idx = data_idx + self.size - 1
        delta_value = priority - self.nodes[tree_idx]
        print(f'delta: {delta_value}')
        self.nodes[tree_idx] = priority
        
        self._propagate(tree_idx,delta_value)
        
    def _propagate(self,tree_idx,delta_value):
        """
        when the child node changed, the value of others will be updated
        
        :param tree_idx: the node changed
        :param delta_value: the increase value
        
        """
        parent_idx = (tree_idx - 1) // 2
        while parent_idx >= 0:
            self.nodes[parent_idx] += delta_value
            parent_idx = (parent_idx - 1) // 2
            
    def get(self,cumsum):
        """
        Get the data at a given cumulative sum of priorities.

        :param cumsum: The cumulative sum used to find a leaf.
        :return: A tuple containing the data index in the buffer, the priority, and the data itself.
        """
        assert cumsum <= self.total_priority

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]
    
    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"