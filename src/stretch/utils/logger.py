# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from termcolor import colored


class Logger:
    def __init__(self, name: str, hide_info: bool = False) -> None:
        self.name = name
        self._hide_info = hide_info

    def hide_info(self) -> None:
        self._hide_info = True

    def _flatten(self, args: tuple) -> str:
        """Flatten a tuple of arguments into a string joined by spaces.

        Args:
            args (tuple): Tuple of arguments to flatten.

        Returns:
            str: Flattened string.
        """
        text = " ".join([str(arg) for arg in args])
        if self.name is not None:
            text = f"[{self.name}] {text}"
        return text

    def error(self, *args) -> None:
        text = self._flatten(args)
        print(colored(text, "red"))

    def info(self, *args) -> None:
        if not self._hide_info:
            text = self._flatten(args)
            print(colored(text, "white"))

    def warning(self, *args) -> None:
        text = self._flatten(args)
        print(colored(text, "yellow"))

    def alert(self, *args) -> None:
        text = self._flatten(args)
        print(colored(text, "green"))


_default_logger = Logger(None)


def error(*args) -> None:
    _default_logger.error(*args)


def info(*args) -> None:
    _default_logger.info(*args)


def warning(*args) -> None:
    _default_logger.warning(*args)


def alert(*args) -> None:
    _default_logger.alert(*args)
