# -*- coding: utf-8 -*-     
import numpy as np

#用法
#/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/3dpes/cam_1/00072_00001.bmp 72
#/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/3dpes/cam_0/00154_00000.bmp 154
#对如上格式txt，按照id从小到大排序


dataset = "viper"
exp = "val"

def rank_by_id(dataset,exp):
	with open('/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/'+dataset+'/'+exp+'.txt', 'r') as f:  
	    data = f.readlines()
	    x = [None for i in range(len(data))]
	    y = [None for i in range(len(data))]
	    #去掉尾部“/n”
	    for n,line in enumerate(data):
		line=line.strip('\n')
		x[n] = line.split(' ',1)[0]
		y[n] = int(line.split(' ',1)[1]) 
	    #排序，返回索引
	    index = np.argsort(y)        


	with open('/home/nikoong/Algorithm_test/dgd_person_reid/external/exp/datasets/'+dataset+'/'+exp+'_triplet.txt', 'w') as f:
	    for line in index:
		f.write(data[line].strip('\n') + '\n') 
	return 


for dataset in ['cuhk03','cuhk01','prid','viper','3dpes','ilids']:
    for exp in ['train','val']:
        rank_by_id(dataset,exp)
