#!/usr/bin/env python

import os
import sys

# For Windows
if os.name == "nt":
    import msvcrt

# For Unix (Linux, macOS)
else:
    import termios
    import tty


def key_pressed(key):
    print(f"Key '{key}' was pressed")


def getch():
    if os.name == "nt":  # Windows
        return msvcrt.getch().decode("utf-8").lower()
    else:  # Unix-like
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.lower()


def main():
    print("Press W, A, S, or D. Press 'q' to quit.")
    while True:
        char = getch()
        if char == "q":
            break
        elif char in ["w", "a", "s", "d"]:
            key_pressed(char)
        else:
            print("Invalid input. Please press W, A, S, or D.")


if __name__ == "__main__":
    main()
