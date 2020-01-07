#coding:utf-8
import os
import shutil

main_path = r'E:\train4\neg_n16w'

a = ["2019 _ 副本.jpg"]
print(a[0][-6:-4])
print(a[0][:-9]+a[0][-4:])
if u"副本" in a[0]:
    print("1111111111111111")


res_path="E:\\cut_pic\\quick_detect_pic\\quick_pos"

for picname in os.listdir(main_path):
    # if
    if u"副本"==picname[-7:-5]:
    # if picname[:3] == 'neg':
        pic_path = os.path.join(main_path,picname)
        # dst_path =os.path.join(res_path,picname)
        # shutil.move(pic_path,dst_path)
        os.remove(pic_path)
        # add_pic_path = os.path.join(main_path,picname[:-9]+"tsts"+picname[-4:])
        # add_pic_path = os.path.join(main_path,picname[:-9]+"tsts"+picname[-4:])
        # shutil.move(add_pic_path, res_path+"\\"+picname[:-9]+picname[-4:])
        # os.remove(add_pic_path)
        # os.rename(pic_path,add_pic_path)
