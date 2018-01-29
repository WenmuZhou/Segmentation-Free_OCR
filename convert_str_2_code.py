# -*- coding: utf-8 -*-
# @Time    : 2017/11/21 19:52
# @Author  : zhoujun
from src.config import Alphabet

label_file = open('zhengque.txt',encoding='utf8',mode='r')
label_file1 = open('zhengque1.txt',encoding='utf8',mode='w')

label_list = [x.strip() for x in label_file.readlines()]
label_file.close()

for s_i in label_list:
    s_i1 = ''.join([str(Alphabet.CHINESECHAR_LETTERS_DIGITS_EXTENDED.index(x)) + Alphabet.BLANK_SYMBOL for x in s_i])[:-1]
    label_file1.write(s_i1 + '\n')
label_file1.close()