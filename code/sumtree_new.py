class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]


    def child_idx(self, data_idx):
      child_idx = data_idx + self.size - 1

      return child_idx

    def parent_idx(self, child_idx):
      parent_idx = (child_idx - 1) // 2

      return parent_idx

    def propagate(self, child_idx, delta_value):
      parent = self.parent_idx(child_idx)

      while parent >= 0:
        self.nodes[parent] += delta_value
        parent = self.parent_idx(parent)

    def update(self, data_idx, value):
        child = self.child_idx(data_idx)  # child index in tree array
        delta_value = value - self.nodes[child]

        self.nodes[child] = value

        self.propagate(child, delta_value)

    def increase_priority(self, data_idx, value, max_value, decay_scheme):
        if value < 0.0:
          raise ValueError(f'Sum tree values should be nonnegative. Got {value}.')
        child = self.child_idx(data_idx)

        old_priority = self.nodes[child]
        if decay_scheme == 'max':
          value = max(value, old_priority)
        elif decay_scheme == 'add':
          value = min(value + old_priority, max_value)
        delta_value = value - old_priority
        print(f'increasing ... delta_value={delta_value}')
        if delta_value == 0 or child < 0:
          return
        self.nodes[child] = value

        self.propagate(child, delta_value)

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

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