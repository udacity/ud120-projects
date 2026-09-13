ud120-projects
==============

Starter project code for students taking Udacity ud120

### Setup
- Create a virtual environment for Python 3.9 or higher.
- $ pip install -r requirements.txt

### Contribution Note
The code in this repo has been upgraded from Python 2 to Python 3 by [Siddharth Kekre](https://github.com/iSiddharth20) in this [PR](https://github.com/udacity/ud120-projects/pull/302). Refer to the [Change Log](https://github.com/iSiddharth20/ud120-projects/blob/master/CHANGELOG.md) for more details. 

### Path Note
According to your `ud120-projects` location you may have to change the path for every `sys.path.append(os.path.abspath(("../tools/")))` to `sys.path.append(os.path.abspath(("./tools/")))` (Basically '../' to './' for all of the files path) or vice-versa in your `.py` files.
