import numpy as np
from typing import List

from stretch.motion.base import Node

class TreeNode(Node):
    """Stores an individual spot in the tree"""

    def __init__(self, state: np.ndarray, parent=None):
        """A treenode is just a pointer back to its parent and an associated state."""
        super(TreeNode, self).__init__(state)
        self.state = state
        self.parent = parent

    def backup(self) -> List["TreeNode"]:
        """Get the full plan by looking back from this point. Returns a list of TreeNodes which contain state."""
        sequence = []
        node = self
        # Look backwards to get a tree
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]


