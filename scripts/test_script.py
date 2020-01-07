#coding:utf-8
import os
import numpy as np
import numpy.random as npr
import shutil
import cv2

"""
main_path = "E:\\cut_pic\\20190821\\20190909_num_every_angle"
f= open("E:\\alrogirths\\slim\\scripts\\pos.txt",'w')
count=0

for dirpath in os.listdir(main_path):
    if dirpath in [1,2,3,4,5,6,7,8,9,10,11,12,13,"1","2","3","4","5","6","7","8","9","10","11","12","13"]:
        dir_path = os.path.join(main_path,dirpath)
        picname_list = os.listdir(dir_path)
        random_list = npr.choice(len(os.listdir(dir_path)),2000,replace=False)
        for i in random_list:
            pic_path = os.path.join(dir_path,picname_list[i])
            # print(pic_path)
            src=cv2.imread(pic_path,0)
            dst = cv2.resize(src,(24,24),0,0)
            dst_path = os.path.join("E:\\train4\\20190917_pos_2424_nums",picname_list[i])
            cv2.imwrite(dst_path,dst)
"""
xmlpath="E:\\train4\\20190917_xml_data_num\\cascade.xml"
cascade=cv2.CascadeClassifier(xmlpath)

main_path ="E:\\cut_pic\\20190924_cutpokers\\cut_pokers\\13"
restore_path ="E:\\cut_pic\\20190924_cutpokers\\restore_pokers\\13_nums"
count=1
for dirname in os.listdir(main_path):
    dir_path = os.path.join(main_path,dirname)
    for picname in os.listdir(dir_path):
        pic_path = os.path.join(dir_path,picname)
        print(pic_path)
        src=cv2.imread(pic_path,0)
        gray = cv2.equalizeHist(src)
        pokers=cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=1,minSize=(24,24),maxSize=(38,38))
        for (x,y,w,h) in pokers:
            cut_pic = src[y:y+h,x:x+w]
            res_picname = os.path.join(restore_path,dirname+"_"+str(count)+".jpg")
            print(res_picname)
            cv2.imwrite(res_picname,cut_pic)
            count+=1
print("完成")