ud120-projects
==============

Starter project code for students taking Udacity ud120


### python2 setup

1) Create, if necessary, a virtual environment, using Python2 as Python binary.

    Detailed info on the [official docs](https://virtualenv.pypa.io/en/stable/installation/)

    ```virtualenv -p `which python` venv```

2) Activate the newly created virtual environment

    `source venv/bin/activate`

3) Install required packages

    `pip install -r requirements.txt`

### running startup script

This will download the Enron dataset (may take a while).

```
cd tools
python startup.py
```
