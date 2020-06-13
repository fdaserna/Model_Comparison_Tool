# -*- coding: utf-8 -*-
import sys,os,copy
from PyQt5.QtWidgets import *
# from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
# from PyQt5.QtWidgets import QWidget, QApplication, QLabel,  QTableWidget,QHBoxLayout, QTableWidgetItem, QComboBox,QFrame
# from PyQt5.QtGui import QFont,QColor,QBrush,QPixmap
from PyQt5 import QtCore, QtWidgets,QtGui

# from numpy import arange
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RangeSpinBox(QWidget):

	def __init__(self,name):
		super(RangeSpinBox, self).__init__()
		self.name=name
		self.set_minimum=0
		self.set_maximum=1
		# confidence阈值
		self.range_box_min = QDoubleSpinBox()
		self.range_box_min.setRange(0.00, 1.00)
		self.range_box_min.setSingleStep(0.05)
		self.range_box_min.setDecimals(3)
		self.range_box_min.setValue(self.set_minimum)
		self.range_box_min.valueChanged.connect(self.min_slot)

		self.range_box_max = QDoubleSpinBox()
		self.range_box_max.setRange(0.00, 1.00)
		self.range_box_max.setSingleStep(0.05)
		self.range_box_max.setDecimals(3)
		self.range_box_max.setValue(self.set_maximum)
		self.range_box_max.valueChanged.connect(self.max_slot)

		self.range_show = QtWidgets.QLabel(self)
		self.range_show.setText(str(self.set_minimum) + '≤' + self.name + '≤' + str(self.set_maximum))

		# 第一层阈值
		self.range_layout1 = QHBoxLayout()
		self.range_layout1.addWidget(self.range_box_min)
		self.range_layout1.addWidget(self.range_show)
		self.range_layout1.addWidget(self.range_box_max)

		# self.groupbox = QGroupBox("范围设置")
		# self.groupbox.setLayout(self.range_layout1)
		#
		# self.mainLayout = QVBoxLayout()
		# self.mainLayout.addWidget(self.groupbox)
		# self.setLayout(self.mainLayout)

		self.setLayout(self.range_layout1)
		self.setWindowTitle("范围设置")

	def min_slot(self, value):
		if float(value) < 0.00000001:
			value = 0
		if float(value)>self.set_maximum:
			value = self.set_maximum
		self.set_minimum = value
		self.range_show.setText(str(self.set_minimum)[:5] + '≤' + self.name + '≤' + str(self.set_maximum)[:5])

	def max_slot(self, value):
		if float(value) < 0.00000001:
			value = 0
		if float(value)<self.set_minimum:
			value = self.set_minimum
		self.set_maximum = value
		self.range_show.setText(str(self.set_minimum)[:5] + '≤' + self.name + '≤' + str(self.set_maximum)[:5])


class CheckBoxDemo(QWidget):

	def __init__(self, parent=None):
		super(CheckBoxDemo, self).__init__(parent)

		self.groupBox = QGroupBox("PR曲线显示")
		self.groupBox.setFlat(False)

		self.layout = QHBoxLayout()
		self.checkBox1 = QCheckBox("显示实线")
		self.checkBox1.setToolTip("用于计算mAP的P-R曲线")
		self.checkBox1.setChecked(True)
		self.layout.addWidget(self.checkBox1)

		self.checkBox2 = QCheckBox("显示虚线")
		self.checkBox2.setToolTip("实际P-R曲线")
		self.checkBox2.setChecked(True)
		self.layout.addWidget(self.checkBox2)

		self.checkBox3 = QCheckBox("坐标轴自适应")
		self.checkBox3.setToolTip("坐标轴的范围随曲线范围变化")
		self.checkBox3.setChecked(False)
		self.layout.addWidget(self.checkBox3)

		self.checkBox4 = QCheckBox("显示图例")
		self.checkBox4.setChecked(True)
		self.layout.addWidget(self.checkBox4)

		self.groupBox.setLayout(self.layout)
		self.mainLayout = QVBoxLayout()
		self.mainLayout.addWidget(self.groupBox)

		self.setLayout(self.mainLayout)
		self.setWindowTitle("checkbox demo")


class NumericItem(QtWidgets.QTableWidgetItem):

	def __lt__(self, other):
		return (self.data(QtCore.Qt.UserRole) <
				other.data(QtCore.Qt.UserRole))


class ResTable(QtWidgets.QTableWidget):

	def __init__(self):
		horizontal_header = ["文件名", "预测框标签", "tp_num", "fp_num", "AP", "mAP", "AR", "recall", "precision","iteration"]
		super(ResTable, self).__init__(0, len(horizontal_header))
		self.setHorizontalHeaderLabels(horizontal_header)
		for i in range(len(horizontal_header)):
			self.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)  

	def insert_data(self,
					res,
					horizontal_header=["文件名", "预测框标签", "tp_num", "fp_num", "AP", "mAP", "AR", "recall", "precision","iteration"]
					):
		self.setColumnCount(len(horizontal_header))
		self.setHorizontalHeaderLabels(horizontal_header)
		self.setRowCount(len(res[0]))

		for column, values in enumerate((res)):
			for row, value in enumerate(values):
				text = str(value)
				if column >= 4 and column <= 8:
					value = float(value) * 100
					text = ('{:.2f}%'.format(value))
				item = NumericItem(text)
				item.setData(QtCore.Qt.UserRole, value)

				self.setItem(row, column, item)

		self.setSortingEnabled(True)
		self.sortItems(0, QtCore.Qt.AscendingOrder)


class QTypeSignal(QObject):
	sendmsg = pyqtSignal(list)

	def __init__(self):
		super(QTypeSignal, self).__init__()

	def run(self,res):
		self.sendmsg.emit(res)


class QTypeSlot(QObject):

	def __init__(self):
		super(QTypeSlot, self).__init__()

	def get(self, msg):
		print("QSlot get msg => " + msg)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	# checkboxDemo = CheckBoxDemo()
	# checkboxDemo.show()
	range_box=RangeSpinBox()
	range_box.show()
	sys.exit(app.exec_())
