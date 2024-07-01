from termcolor import colored


def error(*args):
    print(colored(*args, "red"))


def info(*args):
    print(*args)


def warning(*args):
    print(colored(*args, "yellow"))
