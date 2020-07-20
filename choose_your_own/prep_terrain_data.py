#!/usr/bin/python
import random


def make_terrain_data(n_points=1000):
    """
    make the toy dataset
    """
    random.seed(42)
    grade = [random.random() for ii in range(0, n_points)]
    bumpy = [random.random() for ii in range(0, n_points)]
    error = [random.random() for ii in range(0, n_points)]
    y = [
        round(grade[ii] * bumpy[ii] + 0.3 + 0.1 * error[ii])
        for ii in range(0, n_points)
    ]
    for ii in range(0, len(y)):
        if grade[ii] > 0.8 or bumpy[ii] > 0.8:
            y[ii] = 1.0

    # split into train/test sets
    x = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75 * n_points)
    x_train = x[0:split]
    x_test = x[split:]
    y_train = y[0:split]
    y_test = y[split:]

    # grade_sig = [x_train[ii][0] for ii in range(0, len(x_train)) if y_train[ii] == 0]
    # bumpy_sig = [x_train[ii][1] for ii in range(0, len(x_train)) if y_train[ii] == 0]
    # grade_bkg = [x_train[ii][0] for ii in range(0, len(x_train)) if y_train[ii] == 1]
    # bumpy_bkg = [x_train[ii][1] for ii in range(0, len(x_train)) if y_train[ii] == 1]

    # training_data = {
    #     "fast": {"grade": grade_sig, "bumpiness": bumpy_sig},
    #     "slow": {"grade": grade_bkg, "bumpiness": bumpy_bkg},
    # }

    # grade_sig = [x_test[ii][0] for ii in range(0, len(x_test)) if y_test[ii] == 0]
    # bumpy_sig = [x_test[ii][1] for ii in range(0, len(x_test)) if y_test[ii] == 0]
    # grade_bkg = [x_test[ii][0] for ii in range(0, len(x_test)) if y_test[ii] == 1]
    # bumpy_bkg = [x_test[ii][1] for ii in range(0, len(x_test)) if y_test[ii] == 1]

    # test_data = {
    #     "fast": {"grade": grade_sig, "bumpiness": bumpy_sig},
    #     "slow": {"grade": grade_bkg, "bumpiness": bumpy_bkg},
    # }

    return x_train, y_train, x_test, y_test
