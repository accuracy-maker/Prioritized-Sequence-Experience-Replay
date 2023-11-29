# Prioritized-Sequence-Experience-Replay

My implementation of Prioritized Sequence Experience Replay. The paper link is: https://arxiv.org/pdf/1905.12726.pdf

## Data Structure: SumTree

I learnt this data structure from: http://www.sefidian.com/2022/11/09/sumtree-data-structure-for-prioritized-experience-replay-per-explained-with-python-code/

```
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

    def propagate(self, idx, delta_value):
      parent = (idx - 1) // 2

      while parent >= 0:
        self.nodes[parent] += delta_value
        parent = (parent - 1) // 2

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        delta_value = value - self.nodes[idx]

        self.nodes[idx] = value

        self.propagate(idx, delta_value)

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
```

The provided code defines a Python class called `SumTree`, which is a specialized data structure for efficiently handling a specific set of operations, mainly related to managing a cumulative sum of priorities.

I generated some test codes to test the class:

```

# Test the sum tree 
if __name__ == '__main__':
    # Assuming the SumTree class definition is available

    # Function to print the state of the tree for easier debugging
    def print_tree(tree):
        print("Tree Total:", tree.total)
        print("Tree Nodes:", tree.nodes)
        print("Tree Data:", tree.data)
        print()

    # Create a SumTree instance
    tree_size = 5
    tree = SumTree(tree_size)

    # Add some data with initial priorities
    print("Adding data to the tree...")
    for i in range(tree_size):
        data = f"Data-{i}"
        priority = i + 1  # Priority is just a simple increasing number for this test
        tree.add(priority, data)
        print_tree(tree)

    # Update priority of a data item
    print("Updating priority...")
    update_index = 2  # For example, update the priority of the third item
    new_priority = 10
    tree.update(update_index, new_priority)
    print_tree(tree)

    # Retrieve data based on cumulative sum
    print("Retrieving data based on cumulative sum...")
    cumulative_sums = [5, 15, 20]  # Test with different cumulative sums
    for cumsum in cumulative_sums:
        idx, node_value, data = tree.get(cumsum)
        print(f"Cumulative Sum: {cumsum} -> Retrieved: {data} with Priority: {node_value}")
        print()
```

Here is the outputs:

```

Adding data to the tree...
Tree Total: 1
Tree Nodes: [1, 1, 0, 0, 1, 0, 0, 0, 0]
Tree Data: ['Data-0', None, None, None, None]

Tree Total: 3
Tree Nodes: [3, 1, 2, 0, 1, 2, 0, 0, 0]
Tree Data: ['Data-0', 'Data-1', None, None, None]

Tree Total: 6
Tree Nodes: [6, 1, 5, 0, 1, 2, 3, 0, 0]
Tree Data: ['Data-0', 'Data-1', 'Data-2', None, None]

Tree Total: 10
Tree Nodes: [10, 5, 5, 4, 1, 2, 3, 4, 0]
Tree Data: ['Data-0', 'Data-1', 'Data-2', 'Data-3', None]

Tree Total: 15
Tree Nodes: [15, 10, 5, 9, 1, 2, 3, 4, 5]
Tree Data: ['Data-0', 'Data-1', 'Data-2', 'Data-3', 'Data-4']

Updating priority...
Tree Total: 22
Tree Nodes: [22, 10, 12, 9, 1, 2, 10, 4, 5]
Tree Data: ['Data-0', 'Data-1', 'Data-2', 'Data-3', 'Data-4']

Retrieving data based on cumulative sum...
Cumulative Sum: 5 -> Retrieved: Data-4 with Priority: 5

Cumulative Sum: 15 -> Retrieved: Data-2 with Priority: 10

Cumulative Sum: 20 -> Retrieved: Data-2 with Priority: 10
```
