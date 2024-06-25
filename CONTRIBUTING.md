# Contributing to Stretchpy

We welcome contributions to Stretchpy! Please read the following guidelines before submitting a pull request.

This repository is in an early state of development. Guidelines are subject to change and tests will be by necessity incomplete.

### Setup

Install the code and set up the pre-commit hooks:
```
cd stretchpy/src
pip install -e .[dev]
pre-commit install
```

### Style

We use [black](https://black.readthedocs.io/en/stable/) and [flake8](https://flake8.pycqa.org/en/latest/) to format our code. 
In addition, we use `isort` for sorting imports. 


You can run them with:
```
# Make sure code is formatted correctly
black .
flake8

# Make sure imports are organized and sorted for easy reading
isort .
```

However, these should all be run automatically by the pre-commit hooks. You can force a run with:
```
pre-commit run --all-files
```

### Pull Requests

We follow a squash-and-merge strategy for pull requests, which means that all commits in a PR are squashed into a single commit before merging. This keeps the git history clean and easy to read.

Please make sure your PR is up-to-date with the latest changes in the main branch before submitting. You can do this by rebasing your branch on the main branch:
```
git checkout main
git pull
git checkout <your-branch>
git rebase main
```

#### Draft PRs

If a PR is still a work-in-progress and not ready for review, please open it with "WIP: (final PR title)" in the title to indicate that it is still a work in progress. This will indicate to reviewers that the PR is not ready for review yet. In addition, use the "Draft" PR status on github to indicate that the PR is not ready yet.

### Documentation

Please make sure to update the documentation if you are adding new features or changing existing ones. This includes docstrings, README, and any other relevant documentation. Use [type hints](https://docs.python.org/3/library/typing.html) to make the code more readable and maintainable.

For example:
```python
def add(a: int, b: int) -> int:
    return a + b
```

This shows what `a` and `b` are expected to be and what the function returns -- in this case, all are `int` variables.


### Testing

We use [pytest](https://docs.pytest.org/en/7.0.1/) for testing. Please make sure to add tests for new features or changes to existing code. You can run the tests with:
```
cd src
pytest
```

Run mypy to check for type errors:
```bash
python -m mypy --exclude src/stretch/perception/detection/detic/Detic/ --explicit-package-bases --exclude third_party --namespace-packages --disable-error-code=import-untyped .
```


### File Structure

TODO: Add file structure here
