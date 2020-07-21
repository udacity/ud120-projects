#!/usr/bin/python

import base64
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def visualize(y_values, x_values, color, label, y_label="y", x_label="x", filename="{}.png".format(datetime.now())):
    plt.scatter(y_values, x_values, color=color, label=label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)


def pretty_picture(clf, x_test, y_test):
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = 0.01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, z, cmap=plt.get_cmap('PiYG'))

    # Plot also the test points
    grade_sig = [x_test[ii][0] for ii in range(0, len(x_test)) if y_test[ii] == 0]
    bumpy_sig = [x_test[ii][1] for ii in range(0, len(x_test)) if y_test[ii] == 0]
    grade_bkg = [x_test[ii][0] for ii in range(0, len(x_test)) if y_test[ii] == 1]
    bumpy_bkg = [x_test[ii][1] for ii in range(0, len(x_test)) if y_test[ii] == 1]

    visualize(grade_sig, bumpy_sig, color="b", label="fast", y_label="grade", x_label="bumpiness")
    visualize(grade_bkg, bumpy_bkg, color="r", label="slow", y_label="grade", x_label="bumpiness")


def output_image(name, image_format, input_bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {"name": name, "format": image_format, "bytes": base64.encodestring(input_bytes)}
    print(image_start + json.dumps(data) + image_end)
