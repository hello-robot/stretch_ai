# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

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
    return str(repo.active_branch)


def get_git_commit() -> Optional[str]:
    repo = get_git_repo()
    if repo is None:
        return None
    return str(repo.head.commit.hexsha)


def get_git_commit_message() -> Optional[str]:
    repo = get_git_repo()
    if repo is None:
        return None
    return str(repo.head.commit.message)


if __name__ == "__main__":
    print("Repo:", get_git_repo())
    print("Branch:", get_git_branch())
    print("Commit:", get_git_commit())
    print("Message:", get_git_commit_message())
