from termcolor import colored


def _flatten(args: tuple) -> str:
    """Flatten a tuple of arguments into a string joined by spaces.

    Args:
        args (tuple): Tuple of arguments to flatten.

    Returns:
        str: Flattened string.
    """
    return " ".join([str(arg) for arg in args])


def error(*args) -> None:
    print(colored(_flatten(args), "red"))


def info(*args) -> None:
    print(*args)


def warning(*args) -> None:
    print(colored(_flatten(args), "yellow"))


def alert(*args) -> None:
    print(colored(_flatten(args), "green"))
