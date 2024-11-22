# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

import numpy as np

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
