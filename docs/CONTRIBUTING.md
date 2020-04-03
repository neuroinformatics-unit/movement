### Contributing
#### Setup
To set up a development install, please:
* Fork this repository
* Clone your fork

    `git clone https://github.com/YOUR_USERNAME/movement`
    
    `cd neuro`
* Add this repository as the upstream

    `git remote add upstream https://github.com/adamltyson/movement`
    
* Install an editable, development version of `imlib` 

    `pip install -e .[dev]`

* To keep your fork up to date:

    `git fetch upstream`
    
    `git merge upstream/master`
    
#### Pull requests
In all cases, please submit code to the main repository via a pull request. 
Upon approval, please merge via "Squash and Merge" on Github to maintain a 
clean commit history.


#### Formatting
`movement` uses [Black](https://github.com/python/black) to ensure a consistent 
code style. Please run `black ./ -l 79 --target-version py37` before making 
any commits. To prevent any errors, it is easier to add a formatting check 
as a [pre-commit hook](https://www.atlassian.com/git/tutorials/git-hooks). 
E.g. on linux by adding this to your `.git/hooks/pre-commit`:

    black ./ -l 79 --target-version py37 --check || exit 1

#### Testing
`movement` uses [pytest](https://docs.pytest.org/en/latest/) for testing. Please 
try to ensure that all functions are tested in `tests/tests/unit_tests` and 
all workflows/command-line tools are tested in `tests/tests/unit_tests`.

#### Travis
All commits & pull requests will be build by [Travis](https://travis-ci.com). 
To ensure there are no issues, ensure that all tests run (`pytest`) and there 
are no formatting issues (`black ./ -l 79 --target-version py37 --check`) 
before pushing changes.

#### Releases
Travis will automatically release any tagged commit on the master branch. 
Hence to release a new version of `movement`, use either GitHub, or the git 
CLI to tag the relevant commit and push to master.
