# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 07:47:18 2019

@author: acer
"""

'''<<5行代碼讀取csv檔>>'''
#degroup_csv_file = 'C:/Users/acer/Desktop/python_teaching/degroup.csv'
degroup_csv_file = 'degroup.csv'
f_degroupcsv = open(degroup_csv_file,'r')     #打開csv_file, 'r'是指唯讀read
data_degroupcsv = f_degroupcsv.readlines()    #readlines是把整個csv逐行讀取, 第一行是標籤名稱, 第二行開始是數據內容, 數據類型是list
print("data_degroupcsv:", data_degroupcsv[0:3])   #打印出第0,1,2筆資料
f_degroupcsv.close()    #必須close_file, 否則當作沒有關閉

'''<<利用csv讀取文件的圖片名稱和label名稱>>'''
dict_to_save = {}    #把csv的資料以dict的形式儲存
label_set = set()    #把label的資料以set的形式儲存
for data in data_degroupcsv[1:5]:    #[1:5]表示讀取第1,2,3,4筆資料, (第0筆資料是標籤), for i in data_degroupcsv: 表示逐筆資料處理
    split_line = data.split(',')    #將list中一整串字串以','分裂為小字串, 例如'xxx.jpeg,350,720,83,240,134,284,11\n' --> 'xxx.jpeg', '350', '720', '83', '240', '134', '284', '11\n'
    print("split_line:", split_line)    
    img_name = split_line[0]    #取出第0個元素, 即圖片名稱
    print("img_name:", img_name)
    img_label = split_line[-1].strip()    #.strip()方法刪除首除的空格或換行符號, 即\n
    label_set.add(int(img_label))    #把img_label由str類型轉換為int類型, 然後順序加入label_set中
    dict_to_save[img_name] = img_label    #把img_name和img_label制作成為字典(dict), dict的特點是
print('label_set:',label_set)
print('number of label:',len(label_set))    #len()計算有多少個元素
print('dict_to_save:',dict_to_save)

