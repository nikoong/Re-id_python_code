import os
import sys

dir = "/home/nikoong/Algorithm_test/New_dataset/cam_0"
try:
    file1 = open("/home/nikoong/Algorithm_test/New_dataset/test.txt","w")
    namelist = os.listdir(dir)
    
    
    for line in namelist:
        id_ =  line.split("_",2)[0]
        print dir+"/"+line+" "+str(int(id_))
        file1.write(dir+"/"+line+" "+str(int(id_))+"\n")
    
finally:
    if file1:	
    	file1.close()

