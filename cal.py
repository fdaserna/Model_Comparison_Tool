import sys
import matplotlib.pyplot as plt
import numpy as np

def range_screen(res,model_list, rec_prec_list,
				ap_set_min=0,
				ap_set_max=1,
				map_set_min=0,
				map_set_max=1,
				ar_set_min=0,
				ar_set_max=1):
	if (ap_set_min == 0 and ap_set_max == 1 and map_set_min == 0 and map_set_max == 1 and ar_set_min == 0 and ar_set_max == 1):
		return res,model_list,rec_prec_list
	else:
		res_temp = []
		model_list_temp=[]
		rec_prec_list_temp=[]

		for j in range(len(res)):
			res_temp.append([])

		for i in range(len(res[0])):
			if res[4][i] >= ap_set_min and res[4][i] <= ap_set_max and res[5][i] >= map_set_min and res[5][i] <= map_set_max and res[6][i] >= ap_set_min and res[6][i] <= ap_set_max:
				for j in range(len(res)):
					res_temp[j].append(res[j][i])

				model_list_temp.append(model_list[i])

				for k in range(4):
					rec_prec_list_temp.append(rec_prec_list[k+4*i])

		return res_temp, model_list_temp, rec_prec_list_temp

def get_res(dir,
			txt_list,
			iou_threshold=0.5,
			confidence_threshold=0):
	ground_truth_path = dir
	candidate_bound_path = []
	cb_txt = []
	for txt in txt_list:
		index = len(txt) - 1
		while (index >= 0 and txt[index] != '/'):
			index -= 1
		candidate_bound_path.append(txt[:index + 1])  #模型文件路径
		cb_txt.append(txt[index + 1:])
	# horizontalHeader = ["文件名", "预测框标签", "tp_num", "fp_num", "ap", "map"]

	model_list = []
	label_list = []
	tp_list = []
	fp_list = []
	ap_list = []
	rec_list=[]
	prec_list=[]
	iteration_num_list=[]

	map_list = []
	ar_list=[]
	res = []
	rec_prec_list=[]

	# plt.axis([0, 1.05, 0, 1.05])  # 坐标范围
	for j in range(len(cb_txt)):
		tp_num, fp_num, mrec, mpre, ap_num ,rec,prec= calculate_ap(ground_truth_path, candidate_bound_path[j], cb_txt[j], iou_threshold,confidence_threshold)
		rec = rec.tolist()
		prec = prec.tolist()
		mrec = mrec.tolist()
		mpre = mpre.tolist()

		model_name = cb_txt[j]
		label_name = str(get_label(cb_txt[j])[0])
		iteration_num = str(get_label(cb_txt[j])[1])
		# print("迭代",iteration_num)
		# print(model_name,"label_name:",label_name,iteration_num)
		#
		# plt.plot(rec, prec, label=model_name, c=np.random.rand(3, ))

		rec_prec_list.append(rec)
		rec_prec_list.append(prec)
		rec_prec_list.append(mrec)
		rec_prec_list.append(mpre)

		# ["文件名", "预测框标签", "tp_num", "fp_num", "AP", "mAP", "AR", "置信度大于0的recall", "置信度大于0的precision"]
		model_list.append(model_name)
		label_list.append(label_name)
		tp_list.append(tp_num)
		fp_list.append(fp_num)
		ap_list.append(ap_num)
		rec_list.append(rec[-1])  #置信度大于0的召回率
		prec_list.append(prec[-1])  #置信度大于0的精准率
		iteration_num_list.append(int(iteration_num))

	ap_dict = {}
	ar_dict={}
	for i in range(len(model_list)):
		end_index=-1*(4+len(label_list[i]))
		# print(model_list[i],"model_list[i][:-4]是",model_list[i][:-(4+len(label_list[i]))],"ui",model_list[i][:64])
		if (model_list[i][:end_index] in ap_dict):
			ap_dict[model_list[i][:end_index]] = (ap_dict[model_list[i][:end_index]] + ap_list[i]) / 2  #ap
			ar_dict[model_list[i][:end_index]] = (ar_dict[model_list[i][:end_index]] + rec_list[i]) / 2  #ar
		else:
			ap_dict[model_list[i][:end_index]] = ap_list[i]
			ar_dict[model_list[i][:end_index]] = rec_list[i]

	for i in range(len(model_list)):
		end_index = -1 * (4 + len(label_list[i]))
		map_list.append(ap_dict[model_list[i][:end_index]])
		ar_list.append(ar_dict[model_list[i][:end_index]])

	res.append(model_list)
	res.append(label_list)
	res.append(tp_list)
	res.append(fp_list)
	res.append(ap_list)
	res.append(map_list)
	res.append(ar_list)
	res.append(rec_list)
	res.append(prec_list)
	res.append(iteration_num_list)
	return res,model_list,rec_prec_list

def voc_ap(rec, prec, use_07_metric=False):
	""" ap = voc_ap(rec, prec, [use_07_metric])
	Compute VOC AP given precision and recall.
	If use_07_metric is true, uses the
	VOC 07 11 point method (default:False).
	"""
	if use_07_metric:
		# 11 point metric
		ap = 0.
		for t in np.arange(0., 1.1, 0.1):
			if np.sum(rec >= t) == 0:
				p = 0
			else:
				p = np.max(prec[rec >= t])
			ap = ap + p / 11.
	else:
		# correct AP calculation
		# first append sentinel values at the end
		mrec = np.concatenate(([0.], rec, [1.]))
		mpre = np.concatenate(([0.], prec, [0.]))

		# compute the precision envelope
		for i in range(mpre.size - 1, 0, -1):
			mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

		# to calculate area under PR curve, look for points
		# where X axis (recall) changes value
		i = np.where(mrec[1:] != mrec[:-1])[0]
		# and sum (\Delta recall) * prec
		ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

		smrec=mrec[i]
		smpre=mpre[i]
		# print("mrec",len(smrec),len(mrec),len(rec))
	return ap,smrec,smpre

def calculate_ap(ground_truth_path,
				candidate_bound_path,
				cb_txt,
				iou_threshold=0.5,
				confidence_threshold=0
				 ):
	gt_num = 0
	empty_num = 0
	tp_num = 0
	fp_num = 0
	candidate_bound_num = 0
	fp=[]
	tp=[]
	candidate_bound = (candidate_bound_path + cb_txt)

	cb_list_temp = txt_to_list(candidate_bound, ' ')
	gt_num=len(cb_list_temp)
	for i in cb_list_temp:
		# confidence = float(i[1])  #预测框置信度
		cb_coordinate = [float(aa) for aa in i[2:]]
		# 预测框坐标
		gt_coordinates = txt_to_list(ground_truth_path + str(i[0]) + '.txt', ',')
		if (len(gt_coordinates) == 0):
			empty_num += 1
		else:
			pass

	cb_list=[]
	#置信度阈值
	for cb in cb_list_temp:
		if (float(cb[1]) >= float(confidence_threshold)):
			cb_list.append(cb)


	# 对第二列逆序排序
	cb_list.sort(key=lambda x:x[1],reverse=True)

	cb_label=get_label(cb_txt)[0]
	# print(cb_txt,"预测框标签是", cb_label)

	car_type = {
		'f': 'car',
		'w': 'phone',
		'o': 'similar',
		's': 'smoke'
	}

	for i in cb_list:
		confidence = float(i[1])  #预测框置信度
		cb_coordinate = [float(aa) for aa in i[2:]]   # 预测框坐标
		gt_coordinates = txt_to_list(ground_truth_path + str(i[0]) + '.txt', ',')
		# print("gt文件",gt_coordinates)
		# 预测框数
		# if (len(cb_coordinate)!= 0):
		#	 candidate_bound_num += 1

		if (gt_coordinates != []):
			candidate_bound_num += 1   # 预测框数
		# if (len(gt_coordinates) == 0):
		#	 empty_num += 1
		# else:
		#	 pass
		# loop_sign=True

		for j in gt_coordinates:
			# print("j",j)
			iou_value = 0
			gt_coordinate = [float(bb) for bb in j[2:]]
			if (len(gt_coordinate) > 3):
				# gt_num += 1  # 标注框数量
				for k in range(2, len(gt_coordinate)):
					gt_coordinate[k] = (gt_coordinate[k] + gt_coordinate[k - 2])
				iou_value = calculateIoU(gt_coordinate, cb_coordinate)  # IOU结果
			else:
				pass

			gt_label = j[0]
			if (gt_label in car_type):
				# if car_type[gt_label] == cb_label:
				#		 # and loop_sign:
				#	 gt_num += 1  # 标注框数量
				if (iou_value > iou_threshold and car_type[gt_label] == cb_label):
					tp_num += 1  #预测框的正确数
				else:
					pass
			else:
				pass
		# loop_sign = False
		fp_num = candidate_bound_num - tp_num
		tp.append(tp_num)
		fp.append(fp_num)

	tp = np.asarray(tp)
	fp = np.asarray(fp)
	gt_num = gt_num - empty_num
	rec = tp / float(max(gt_num,np.finfo(np.float64).eps))
	prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
	ap,mrec,mpre= voc_ap(rec, prec)

	# print("gtnum:",gt_num,"empty_num",empty_num,"candidate_bound_num",candidate_bound_num,"cb_list长度",len(cb_list))
	return tp_num,fp_num,mrec,mpre,ap,rec,prec

def get_label(cb_txt):  #获取预测框标签
	candidate_bound_label = ""
	iteration_num = ""
	cb_index = len(cb_txt) - 5

	while (cb_index >= 0 and cb_txt[cb_index] != '-' and cb_txt[cb_index].isalnum()):
		candidate_bound_label = cb_txt[cb_index] + candidate_bound_label
		cb_index -= 1

	while cb_index >= 0 and not (cb_txt[cb_index].isdigit()):
		cb_index -= 1

	while cb_index >= 0 and (cb_txt[cb_index].isdigit()):
		iteration_num = cb_txt[cb_index] + iteration_num
		cb_index -= 1
	# print("iter",iteration_num)
	# print(cb_txt, "预测框标签是", candidate_bound_label)

	return candidate_bound_label,iteration_num

def txt_to_list(filename,split_sign):
	data = []
	try:
		with open(filename, 'r') as f:  #with语句自动调用close()方法
			line = f.readline()
			while line:
				read_data=[x for x in line.strip().split(split_sign)]
				data.append(read_data)
				line = f.readline()
	except IOError:
		# print(filename,"is not accessible.")
		pass
	return data   #返回数据为二维列表形式

def calculateIoU(candidateBound, groundTruthBound):  #候选框（candidate bound）原标记框（ground truth bound）
	cx1 = candidateBound[0]
	cy1 = candidateBound[1]
	cx2 = candidateBound[2]
	cy2 = candidateBound[3]

	gx1 = groundTruthBound[0]
	gy1 = groundTruthBound[1]
	gx2 = groundTruthBound[2]
	gy2 = groundTruthBound[3]

	carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
	garea = (gx2 - gx1) * (gy2 - gy1)  # G的面积

	x1 = max(cx1, gx1)
	y1 = max(cy1, gy1)
	x2 = min(cx2, gx2)
	y2 = min(cy2, gy2)
	w = max(0, x2 - x1)
	h = max(0, y2 - y1)
	area = w * h  # C∩G的面积

	iou = area / (carea + garea - area)

	return iou
