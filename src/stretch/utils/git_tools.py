from pathlib import Path
from typing import Optional

import git


def get_git_repo() -> Optional[git.Repo]:
    path = str(Path(__file__).resolve().parent)

    repo_names = ["stretch_ai", "stretchpy"]

    # Find the root of the repo given __file__
    for name in repo_names:
        idx = path.find(name)
        if idx != -1:
            idx += len(name)
            break

    repo_root = path[:idx]
    try:
        return git.Repo(repo_root)
    except git.exc.InvalidGitRepositoryError:
        return None


def get_git_branch() -> Optional[str]:
    repo = get_git_repo()
    if repo is None:
        return None
    return repo.active_branch


def get_git_commit() -> Optional[str]:
    repo = get_git_repo()
    if repo is None:
        return None
    return repo.head.commit.hexsha


def get_git_commit_message() -> Optional[str]:
    repo = get_git_repo()
    if repo is None:
        return None
    return repo.head.commit.message


if __name__ == "___main__":
    print(get_git_repo())
    print(get_git_branch())
    print(get_git_commit())
    print(get_git_commit_message())
