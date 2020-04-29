# [WIP] ud120 Projects 2.0

The aim of this repo is to improve original starter project code for students taking
[Intro to Machine Learning](https://classroom.udacity.com/courses/ud120) on Udacity with
python 3.8, conda managing and jupyter notebooks

## Mini-Projects

- [Lesson 2: Naive Bayes Mini-Project](./lesson-2-naive-bayes/nb_author_id.ipynb)
- [Lesson 3: SVM Mini-Project](./lesson-3-svm/svm_author_id.ipynb)
- [Lesson 4: Decision Trees Mini-Project](./lesson-4-decision-tree/dt_author_id.ipynb)
- [Lesson 5: Choose Your own Algorithm Mini Project](./lesson-5-choose-your-own/your_algorithm.ipynb)
- [Lesson 6: Datasets and Questions Mini-Project](./lesson-6-datasets-questions/explore_enron_data.ipynb)
- [Lesson 7: Regressions Mini-Project](./lesson-7-regression/finance_regression.ipynb)

## Cloning the repo

```bash
$ git clone https://github.com/trsvchn/ud120-projects-v2.git
$ cd ud120-projects-v2
```

## Setting up conda environment

Download and install anaconda from [here](https://www.anaconda.com/distribution/)

Create environment

```bash
$ conda env create -f environment.yml
```

Activate `ud120` environment via

```bash
$ conda activate ud120
```

## Run starter script to check env and download required data

```bash
$ python ./utils/starter.py
```
