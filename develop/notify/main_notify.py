from calculation import *
from sql_read import *
from datetime import datetime, timedelta
import sys
import random
from time import time, sleep

import matplotlib.pyplot as plt  # $ pip install matplotlib
from matplotlib import use as mpl_use
import matplotlib.animation as animation
mpl_use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# def data_gen():
#     while True:
#         time_history = 6000  # seconds
#         time_last = datetime.now()
#         time_first = time_last - timedelta(0, time_history, 0)
#         trade = read_settings_sql(time_first, time_last)
#         trade_graf = calculation(trade)
#         yield trade_graf
#
#
# class MplCanvas(FigureCanvas):
#
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = fig.add_subplot(111)
#         super(MplCanvas, self).__init__(fig)
#
#
# class MainWindow(QtWidgets.QMainWindow):
#
#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)
#
#         self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
#         self.setCentralWidget(self.canvas)
#
#         self.data = next(data_gen())
#         self.xdata = list(range(len(self.data)))
#         self.ydata = self.data['profit']
#         self.update_plot()
#
#         self.show()
#
#         # Setup a timer to trigger the redraw by calling update_plot.
#         self.timer = QtCore.QTimer()
#         self.timer.setInterval(10)
#         self.timer.timeout.connect(self.update_plot)
#         self.timer.start()
#
#     def update_plot(self):
#         # Drop off the first y element, append a new one.
#         self.ydata = self.ydata
#         self.canvas.axes.cla()  # Clear the canvas.
#         self.canvas.axes.plot(self.xdata, self.ydata, 'r')
#         # Trigger the canvas to update and redraw.
#         self.canvas.draw()


fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax1.set_title('Trades on trend')
ax2.set_title('Profit')
ax1.grid()
ax2.grid()

def data_gen():
    while True:
        time_history = 600  # seconds
        time_last = datetime.now()
        time_first = time_last - timedelta(0, time_history, 0)
        trade = read_settings_sql(time_first, time_last)
        trade_graf = calculation(trade)
        yield trade_graf

def animate_ax1(trade_graf):

    x = range(len(trade_graf))
    ax1.set_xlim(0, len(trade_graf))
    ax1.set_ylim(trade_graf['btc_usdt'].min()-trade_graf['btc_usdt'].std()/10, trade_graf['btc_usdt'].max()+trade_graf['btc_usdt'].std()/10)
    graph = ax1.plot(x, trade_graf['btc_usdt'], color='lightsteelblue')
    ax1.plot(x, trade_graf['predict_max_on_btc_usdt'], '.', color='deepskyblue')
    ax1.plot(x, trade_graf['predict_min_on_btc_usdt'], '.', color='burlywood')
    ax1.plot(x, trade_graf['trade_max_on_btc_usdt'], '.', color='green')
    ax1.plot(x, trade_graf['trade_min_on_btc_usdt'], '.', color='red')
    return graph

def animate_ax2(trade_graf):

    x = range(len(trade_graf))
    ax2.set_xlim(0, len(trade_graf))
    ax2.set_ylim(trade_graf['profit'].min()-trade_graf['profit'].std()/10, trade_graf['profit'].max()+trade_graf['profit'].std()/10)
    graph = ax2.plot(x, trade_graf['profit'], color='lightsteelblue')
    ax2.plot(x, trade_graf['trade_on_profit'], '.', color='red')
    return graph

def main():
    print('Trend on window')

    ani_ax1 = animation.FuncAnimation(fig, animate_ax1, data_gen, blit=True)
    ani_ax2 = animation.FuncAnimation(fig, animate_ax2, data_gen, blit=True)
    plt.show()

    # app = QtWidgets.QApplication(sys.argv)
    # w = MainWindow()
    # app.exec_()


if __name__ == '__main__':
    main()