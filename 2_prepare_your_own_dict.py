'''this is a demo about how the data is stored in dictionary'''

list_img = ['apple.jpg,11\n','orange.jpg,12\n', ,'banana.jpg,13\n']
dict_to_save = {}    #把csv的資料以dict的形式儲存
label_set = set()    #把label的資料以set的形式儲存
for i in list_img:    
    split_line = i.split(',')    #將list中一整串字串以','分裂為小字串, 例如'apple.jpg,11\n' --> 'apple.jpeg', '11\n'   
    img_name = split_line[0]    #只要list中第0個元素, 即apple.jpg
    img_label = split_line[-1].strip()    #只要list中最後1個元素, 即11\n, 但去所有空格及換行符號, 即11
    label_set.add(int(img_label))    #把'11'轉換為11, 後續處理用int類型
    dict_to_save[img_name] = img_label    #儲存為dict類型, 方便查找
    #print("split_line:", split_line) 
    #print("img_name:", img_name)    
    #print('img_label:',img_label)

print('label:',label_set)
print('number of label:',len(label_set))
