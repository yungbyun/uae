import numpy as np
import matplotlib.pyplot as plot

'''
Usage:
import myplot
guy = myplot.MyPlot()
guy.show_list(list)

import numpy as np
import myplot.py
gildong = myplot.MyPlot()
gildong.set_attribute('o-')
gildong.show(np.arange(1, 5, 0.1))

'''


class MyPlot: # display using a graph
    attr = 'o-'  # line attribute
    x_label = ''
    y_label = ''

    def set_attribute(self, a):
        self.attr = a

    def set_labels(self, xl, yl):
        self.x_label = xl
        self.y_label = yl

    # arange function returns ndarray which is array of evenly spaced values.
    def show_arange(self, arr=np.arange(1, 10, 0.1)):
        plot.plot(arr, self.attr)
        plot.xlabel(self.x_label)
        plot.ylabel(self.y_label)
        plot.show()

    def show_list(self, list):
        plot.plot(list, self.attr)
        plot.xlabel(self.x_label)
        plot.ylabel(self.y_label)
        plot.show()

