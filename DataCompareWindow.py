import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QWidget, QApplication, QLabel,  QTableWidget,QHBoxLayout, QTableWidgetItem, QComboBox,QFrame
from PyQt5 import QtCore, QtWidgets,QtGui

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axisartist.axislines import SubplotZero

from MyWidget import *
from cal import *


class CompareForm(QWidget):

    def __init__(self, name = '模型比较'):
        super(CompareForm,self).__init__()
        self.dir = './'
        self.txt_list = []
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        # print(type(self.cwd))
        self.iou_threshold = 0.5  # iou阈值
        self.confidence_threshold = 0  # 置信度阈值
        self.model_list=[]
        self.rec_prec_list=[]

        self.txt_list_backup=[]
        self.txt_list_backup_use=False

        self.initUi(name)

    def my_create_widget(self):
        self.res_table = ResTable()  # 表格

        self.btn_choose_dir = QtWidgets.QPushButton(self)
        self.btn_choose_dir.setObjectName("btn_choose_dir")
        self.btn_choose_dir.setText("选择ground truth文件夹")

        self.btn_choose_muti_file = QtWidgets.QPushButton(self)
        self.btn_choose_muti_file.setObjectName("btn_choose_muti_file")
        self.btn_choose_muti_file.setText("选择多个模型结果")

        self.btn_add_muti_file = QtWidgets.QPushButton(self)
        self.btn_add_muti_file.setObjectName("btn_choose_muti_file")
        self.btn_add_muti_file.setText("添加多个模型结果")

        """在窗体内创建button对象"""
        self.cal_button = QtWidgets.QPushButton("计算模型评价指标并绘制P-R曲线", self)
        self.cal_button.setToolTip("计算结果，可点击表头排序")

        # 设置信号"""按钮与鼠标点击事件相关联"""
        self.btn_choose_dir.clicked.connect(self.slot_btn_choose_dir)
        self.btn_choose_muti_file.clicked.connect(self.slot_btn_choose_muti_file)
        self.btn_add_muti_file.clicked.connect(self.slot_btn_add_muti_file)
        self.cal_button.clicked.connect(self.on_click)

        # iou阈值
        self.iouThreshSpinBox = QDoubleSpinBox()
        self.iouThreshSpinBox.setRange(0.000, 1.000)
        self.iouThreshSpinBox.setSingleStep(0.05)
        self.iouThreshSpinBox.setDecimals(3)
        self.iouThreshSpinBox.setValue(self.iou_threshold)
        self.iouThreshSpinBox.valueChanged.connect(self.onValueChangedIouThresh)

        self.iou_threshold_show = QtWidgets.QLabel(self)
        self.iou_threshold_show.setText('IoU阈值:>' + str(self.iou_threshold))

        #confidence阈值
        self.confidence_threshold_box = QDoubleSpinBox()
        self.confidence_threshold_box.setRange(0.00, 1.00)
        self.confidence_threshold_box.setSingleStep(0.05)
        self.confidence_threshold_box.setDecimals(3)
        self.confidence_threshold_box.setValue(self.confidence_threshold)
        self.confidence_threshold_box.valueChanged.connect(self.confidence_threshold_slot)

        self.confidence_threshold_show = QtWidgets.QLabel(self)
        self.confidence_threshold_show.setText('confidence阈值:≥' + str(self.confidence_threshold))

        self.threshold_layout1=QHBoxLayout()
        self.threshold_layout1.addWidget(self.iou_threshold_show)
        self.threshold_layout1.addWidget(self.iouThreshSpinBox)
        self.threshold_layout1.addWidget(self.confidence_threshold_show)
        self.threshold_layout1.addWidget(self.confidence_threshold_box)

        self.threshold_groupbox = QGroupBox("阈值设置")
        self.threshold_groupbox.setLayout(self.threshold_layout1)

        self.dir_show = QtWidgets.QLineEdit(self)
        # self.txt_dir_show = QtWidgets.QLineEdit(self)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.my_checkbox = CheckBoxDemo()

        # 复选框连接槽函数
        self.my_checkbox.checkBox1.stateChanged.connect(lambda: self.btnstate(self.my_checkbox.checkBox1))
        self.my_checkbox.checkBox2.toggled.connect(lambda: self.btnstate(self.my_checkbox.checkBox2))
        self.my_checkbox.checkBox3.stateChanged.connect(lambda: self.btnstate(self.my_checkbox.checkBox3))
        self.my_checkbox.checkBox4.toggled.connect(lambda: self.btnstate(self.my_checkbox.checkBox4))

        self.ap_range_box = RangeSpinBox("AP")
        self.map_range_box = RangeSpinBox("mAP")
        self.ar_range_box = RangeSpinBox("AR")

        self.range_layout1=QHBoxLayout()
        self.range_layout1.addWidget(self.ap_range_box)
        self.range_layout1.addWidget(self.map_range_box)
        self.range_layout1.addWidget(self.ar_range_box)
        self.range_groupbox = QGroupBox("范围设置")
        self.range_groupbox.setLayout(self.range_layout1)

    def initUi(self,name):
        self.my_create_widget()
        self.resize(1600, 900)
        self.setWindowTitle(name)

        # 全局布局（注意参数 self）
        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(self.res_table,0,0,-1,1)
        self.main_layout.addWidget(self.dir_show,0,1)
        self.main_layout.addWidget(self.btn_choose_dir,0,2)
        self.main_layout.addWidget(self.btn_choose_muti_file,0,3)
        self.main_layout.addWidget(self.btn_add_muti_file,0,4)

        self.main_layout.addWidget(self.range_groupbox,1,1,1,3)
        self.main_layout.addWidget(self.cal_button,1,4,1,-1)

        self.main_layout.addWidget(self.threshold_groupbox,2,1,1,1)
        self.main_layout.addWidget(self.my_checkbox.groupBox,2,2,1,-1)
        self.main_layout.addWidget(self.canvas, 3, 1, -1, -1)
        self.setLayout(self.main_layout)
        self.main_layout.setColumnStretch(0,4)
        self.main_layout.setColumnStretch(1,1)
        self.main_layout.setColumnStretch(2,1)
        self.main_layout.setColumnStretch(3,1)
        self.main_layout.setRowStretch(0, 1)
        self.main_layout.setRowStretch(1, 0.5)
        self.main_layout.setRowStretch(2, 0.5)
        self.main_layout.setRowStretch(3, 3)

        # font = QtGui.QFont()
        # font.setPointSize(12)
        # self.dir_show.setFont(font)
        # self.btn_choose_dir.setFont(font)
        # self.range_groupbox.setFont(font)
        # self.btn_choose_muti_file.setFont(font)
        # self.btn_add_muti_file.setFont(font)
        # self.threshold_groupbox.setFont(font)
        # self.cal_button.setFont(font)
        # self.my_checkbox.groupBox.setFont(font)

    def onValueChangedIouThresh(self, value):
        if float(value)<0.00000001:
            value=0
        self.iou_threshold = value
        self.iou_threshold_show.setText('IoU阈值:>' + str(self.iou_threshold)[:5])

    def confidence_threshold_slot(self,value):
        if float(value)<0.00000001:
            value=0
        self.confidence_threshold = value
        self.confidence_threshold_show.setText('confidence阈值:≥' + str(self.confidence_threshold)[:5])

    def btnstate(self, btn):
        # print("aa",self.my_checkbox.checkBox1.isChecked(),
        #            self.my_checkbox.checkBox2.isChecked())
        # print("modelaa",self.model_list)
        self.plot_(self.model_list,
                   self.rec_prec_list,
                   self.my_checkbox.checkBox1.isChecked(),
                   self.my_checkbox.checkBox2.isChecked(),
                   self.my_checkbox.checkBox3.isChecked(),
                   self.my_checkbox.checkBox4.isChecked()
                   )


    def on_click(self):
        ground_truth_path=self.dir
        try:
            self.res,self.model_list,self.rec_prec_list=get_res(ground_truth_path,
                        self.txt_list,
                        self.iou_threshold,
                        self.confidence_threshold)

            self.res, self.model_list, self.rec_prec_list=range_screen(self.res,self.model_list,self.rec_prec_list,
                    self.ap_range_box.set_minimum,
                    self.ap_range_box.set_maximum,
                    self.map_range_box.set_minimum,
                    self.map_range_box.set_maximum,
                    self.ar_range_box.set_minimum,
                    self.ar_range_box.set_maximum,)

            self.plot_(self.model_list,
                       self.rec_prec_list,
                       self.my_checkbox.checkBox1.isChecked(),
                       self.my_checkbox.checkBox2.isChecked(),
                       self.my_checkbox.checkBox3.isChecked(),
                       self.my_checkbox.checkBox4.isChecked()
                       )

            self.res_table.insert_data(self.res)
        except:
            if self.txt_list_backup_use:
                self.txt_list = self.txt_list_backup
            pass


    def plot_(self, model_list, rec_prec_list,show_full_line=True,show_dotted_line=True,self_adaption=False,show_legend=True):
        # 清图
        plt.clf()
        # ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8])
        #https://blog.csdn.net/wuzlun/article/details/80053277


        if not self_adaption:
            fig = self.figure
            ax = SubplotZero(fig, 1, 1, 1)
            fig.add_subplot(ax)
            ax.set_ylim(-0.01,1.01)
            # ax.set_yticks([-1,-0.5,0,0.5,1])
            ax.set_xlim([-0.01,1.01])

            # fig = plt.figure(1, (10, 10))
            # ax = SubplotZero(fig, 1, 1, 1)
            # fig.add_subplot(ax)
            # ax.set_ylim(-0.01,1.01)
            # # ax.set_yticks([-1,-0.5,0,0.5,1])
            # ax.set_xlim([-0.01,1.01])
        else:
            ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8])
        # ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8])

        if len(model_list)<1 or len(rec_prec_list)<1:
            self.canvas.draw()
            return

        if show_full_line and show_dotted_line:
            for i in range(len(model_list)):
                color = np.random.rand(3, )
                ax.plot(rec_prec_list[4 * i], rec_prec_list[4 * i + 1], c=color, ls=":")  # rec, prec
                ax.plot(rec_prec_list[4 * i + 2], rec_prec_list[4 * i + 3], label=model_list[i], c=color)  # mrec, mprec
        elif show_full_line:
            for i in range(len(model_list)):
                color = np.random.rand(3, )
                ax.plot(rec_prec_list[4 * i + 2], rec_prec_list[4 * i + 3], label=model_list[i], c=color)  # mrec, mprec
        elif show_dotted_line:
            for i in range(len(model_list)):
                color = np.random.rand(3, )
                ax.plot(rec_prec_list[4 * i], rec_prec_list[4 * i + 1], c=color, label=model_list[i],ls=":")  # rec, prec
        else:
            self.canvas.draw()
            return

        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title('P-R curve')
        if show_legend:
            ax.legend()
        # ax.legend(loc='lower left')
        self.canvas.draw()


    def slot_btn_choose_dir(self):
        # global dir_choose
        dir_choose= QFileDialog.getExistingDirectory(self,
                                    "选取文件夹",
                                    self.cwd) # 起始路径

        if dir_choose == "":
            print("\n取消选择")
            return

        self.dir=dir_choose+'/'
        self.cwd=self.dir
        self.dir_show.setText(str(self.dir))
        self.on_click()

    def slot_btn_choose_muti_file(self):
        files, filetype = QFileDialog.getOpenFileNames(self,
                                    "多文件选择",
                                    self.cwd, # 起始路径
                                    "Text Files(*.txt)")
                                    # "All Files (*);;PDF Files (*.pdf);;Text Files (*.txt)")

        if len(files) == 0:
            print("\n取消选择")
            return

        self.txt_list_backup_use=True
        self.txt_list_backup=[]
        self.txt_list_backup = copy.deepcopy(self.txt_list)

        self.txt_list=files
        # self.txt_dir_show.setText(str(self.txt_list))
        self.on_click()
        self.txt_list_backup_use = False

    def slot_btn_add_muti_file(self):
        files, filetype = QFileDialog.getOpenFileNames(self,"多文件选择",self.cwd,"Text Files(*.txt)")
        # "All Files (*);;PDF Files (*.pdf);;Text Files (*.txt)")

        if len(files) == 0:
            print("\n取消选择")
            return
        # self.txt_list = files
        self.txt_list_backup_use=True
        self.txt_list_backup=[]
        # for txt in self.txt_list:
        #     self.txt_list_backup.append(txt)
        self.txt_list_backup=copy.deepcopy(self.txt_list)

        for file in files:
            if file not in self.txt_list:
                self.txt_list.append(file)
        # self.txt_dir_show.setText(str(self.txt_list))
        self.on_click()
        self.txt_list_backup_use =False

if __name__=="__main__":
    app = QApplication(sys.argv)
    CompareForm = CompareForm('模型比较')
    CompareForm.show()
    sys.exit(app.exec_())
