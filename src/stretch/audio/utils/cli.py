# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Standard imports
import readline  # Improve interactive input, e.g., up to access history, tab auto-completion.
from typing import Optional


class HistoryCompleter:
    """
    This class enables readline tab auto-completion from the history.

    Adapted from https://pymotw.com/3/readline/
    """

    def __init__(self):
        """
        Initialize the HistoryCompleter.
        """
        self.matches = []

    @staticmethod
    def get_history_items() -> list[str]:
        """
        Get the history items.

        Returns
        -------
        list[str]
            The history items.
        """
        num_items = readline.get_current_history_length() + 1
        return [readline.get_history_item(i) for i in range(1, num_items)]

    def complete(self, text: str, state: int) -> Optional[str]:
        """
        Return the next possible completion for 'text'.

        This is called successively with state == 0, 1, 2, ... until it returns None.

        Parameters
        ----------
        text : str
            The string to complete.
        state : int
            The state of the completion.

        Returns
        -------
        Optional[str]
            The next possible completion for 'text'.
        """
        response = None
        if state == 0:
            history_values = HistoryCompleter.get_history_items()
            if text:
                self.matches = sorted(h for h in history_values if h and h.startswith(text))
            else:
                self.matches = []
        try:
            response = self.matches[state]
        except IndexError:
            response = None
        return response
