#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2016/10/16 20:06
# @Author  : zhoujun
# @Site    :
# @File    : temp.py
# @Software: PyCharm Community Edition
# 加载必要的库
'''
opencv处理图像，近期研究的所有代码
'''
import cv2
import os
import shutil
import math
import numpy as np

folder = 'data/images/'
show_resize = 2


class Rect:
    '''
    矩形类
    '''
    startx = 0
    endx = 0
    starty = 0
    endy = 0

    def __init__(self):
        pass

    def __str__(self):
        return '%d %d %d %d' % (self.startx, self.starty, self.endx, self.endy)


def find_contours(filename):
    '''
    对图像进行连通域检测，找到图像中的文字区域
    :param filename:图像名
    :return:包含连通域矩形框的列表
    '''
    # 获取包含矩阵信息的list
    result = thre(filename=filename, is_adapteive=True)
    img_show(result, window_name='thre', resize=show_resize)

    cv2.imwrite(folder + 'temp/thre.jpg', result)

    # cv2.bitwise_not(result, dst)
    # 膨胀
    dst = cv2.Canny(result, 100, 100, 3)
    img_show(dst, window_name='dst', resize=show_resize)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (40, 2))
    dilated = cv2.dilate(dst, element)
    img_show(dilated, window_name='dilated', resize=show_resize)

    # 轮廓检测
    image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_list = []
    for i in contours:

        first = i[0]
        rect = Rect()
        rect.startx = first[0][0]
        rect.endx = first[0][0]
        rect.starty = first[0][1]
        rect.endy = first[0][1]
        for x in i:
            rect.startx = min(x[0][0], rect.startx)
            rect.endx = max(x[0][0], rect.endx)
            rect.starty = min(x[0][1], rect.starty)
            rect.endy = max(x[0][1], rect.endy)
        rect_list.append(rect)

    avg_height = get_height(rect_list)
    temp_list = [rect for rect in rect_list if
                 rect.endy - rect.starty <= avg_height * 2 and rect.endy - rect.starty > avg_height * 0.7]

    return temp_list


def get_height(rect_list):
    '''
    计算高度的分布，并且得出分布最大区域的平均值
    :param rect_list:连通域矩形列表
    :return:所有连通域中最大的高度
    '''
    # 计算最大值
    max_height = 0
    for rect in rect_list:
        if rect.endy - rect.starty > max_height:
            max_height = rect.endy - rect.starty
    # 向上取整
    max_height = math.ceil(max_height)

    # 计算分布区间
    if max_height % 10 >= 0:
        max_height = int(max_height // 10 + 1)
    else:
        max_height = int(max_height // 10)
    # 存放每个区间的宽度
    list2 = [[] for i in range(max_height)]
    # 将宽度放入每一个区间
    for rect in rect_list:
        height = rect.endy - rect.starty
        if height < 10:
            continue
        temp = height // 10
        list2[int(math.floor(temp))].append(height)

    # 先取出数量最多的宽度区间
    index_top1 = get_max_height_index(list2)
    sum_height = sum(list2[index_top1])
    length = len(list2[index_top1])
    del (list2[index_top1])

    # 取出数量最第二多的宽度区间
    index_top1 = get_max_height_index(list2)

    sum_height += sum(list2[index_top1])
    length += len(list2[index_top1])
    return math.ceil(sum_height / length)


def get_max_height_index(list2):
    '''
    得到数量最多的高度区间索引
    :param list2: 存储了所有矩形高度信息的列表
    :return:数量最多的高度区间索引
    '''
    max_height = 0
    index_top1 = 0
    for i in range(len(list2)):
        if len(list2[i]) > max_height:
            max_height = len(list2[i])
            index_top1 = i
    return index_top1


def combine_connected_domain(rect_list):
    '''
    将相邻的连通域融合在一起
    :param rect_list: 连通域矩阵列表
    :return:
    '''
    # 计算所有框中的高度最大值
    max_height = 0
    for rect in rect_list:
        temp_width = rect.endy - rect.starty
        if temp_width > max_height:
            max_height = temp_width
    print('max height', max_height)
    # y方向的允许误差设定为最大高度的0.3
    wucha_y = max_height * 0.3

    for i in range(0, len(rect_list) - 1):
        previous = i
        last = i + 1
        while rect_list[previous].startx == -10:
            # 当前矩形已经被合并到前一个,取出前一个进行比较
            previous = previous - 1

        # 两个举行的starty和endy之差在一定范围内
        mid_p = (rect_list[previous].starty + rect_list[previous].endy) / 2
        mid_l = (rect_list[last].starty + rect_list[last].endy) / 2

        if abs(mid_p - mid_l) < wucha_y and rect_list[last].startx - rect_list[previous].endx < max_height:
            rect_list[previous].startx = min(rect_list[previous].startx, rect_list[last].startx)
            rect_list[previous].starty = min(rect_list[previous].starty, rect_list[last].starty)
            rect_list[previous].endx = max(rect_list[previous].endx, rect_list[last].endx)
            rect_list[previous].endy = max(rect_list[previous].endy, rect_list[last].endy)
            rect_list[last].startx = -10
            rect_list[last].starty = -10
            rect_list[last].endx = -10
            rect_list[last].endy = -10
    rect_list = [rect for rect in rect_list if rect.startx != -10]
    return rect_list


def bubble_sort(rect_list):
    '''
    冒泡排序，稳定的排序，对列表进行排序，从上到下，从左至右
    :param rect_list:
    :return:
    '''
    max_height = 0
    for rect in rect_list:
        temp_width = rect.endy - rect.starty
        if temp_width > max_height:
            max_height = temp_width

    for i in range(0, len(rect_list) - 1):
        for j in range(0, len(rect_list) - 1 - i):
            mid_iy = rect_list[j].starty + (rect_list[j].endy - rect_list[j].starty) / 2
            if mid_iy > rect_list[j + 1].endy:
                # rect_i在rect_j下面，交换
                rect_list[j], rect_list[j + 1] = rect_list[j + 1], rect_list[j]
            elif mid_iy <= rect_list[j + 1].endy and mid_iy >= rect_list[j + 1].starty:
                # rect_i和rect_j在同一行
                if rect_list[j].startx >= rect_list[j + 1].startx:
                    rect_list[j], rect_list[j + 1] = rect_list[j + 1], rect_list[j]
            else:
                # rect_i在rect_j上面
                pass


def img_show(src, window_name='defoult', resize=show_resize):
    '''
    显示图片
    :param src:带显示的图像
    :param window_name:图像窗口名字
    :param resize:resize比例
    :return:None
    '''
    res = cv2.resize(src, (int(src.shape[1] // resize), int(src.shape[0] // resize)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow(window_name, res)


def draw_rect_line(filename, rect_list, windowsname="矩形", save=False, save_filename='temp'):
    '''
    在图片上绘制连通域并保存会之后的图片
    :param filename:带绘制图像的的名字
    :param rect_list:连通域矩形列表
    :param windowsname:显示窗口的名字
    :param save:是否保存绘制后的图像
    :param save_filename:保存的图像名字
    :return:None
    '''
    # src = cv2.imread(filename, 0)
    result2 = thre(filename=filename, is_adapteive=True)
    result2 = cv2.cvtColor(result2, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite(folder + 'temp/thre.jpg', result2)
    i = 0
    for rect in rect_list:
        i = i + 1
        # cv2.putText(result2, str(i), (rect.startx - 20, rect.endy + 20),
        #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),
        #             thickness=1,
        #             lineType=1)
        cv2.rectangle(result2, (rect.startx, rect.starty), (rect.endx, rect.endy), (255, 0, 0), 5)

    img_show(result2, windowsname)
    if save:
        if not os.path.exists(folder + 'temp/'):
            os.makedirs(folder + 'temp/')
        cv2.imwrite(folder + 'temp/' + save_filename + '.jpg', result2)


def save_image(filename, rect_list):
    '''
    保存列表中的图像到文件夹中
    :param filename:图像文件名
    :param rect_list:框框列表
    :return:
    '''
    result = thre(filename=filename, is_adapteive=True)
    save_folder = folder + 'single7/'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    i = 0
    for char_line in rect_list:
        # 获取ROI区域
        dst = result[max(char_line.starty - 2, 0):min(char_line.endy + 2, result.shape[0]),
              max(char_line.startx + 2, 0):min(char_line.endx + 2, result.shape[1])]

        tempfile = save_folder + str(i) + '.jpg'
        print(tempfile)
        cv2.imwrite(tempfile, dst)
        i += 1


def thre(filename, is_adapteive=False, thre_value=95):
    '''
    对图像进行二值化
    :param filename:图像文件名
    :param is_adapteive: 是否自适应
    :param thre_value: 非自适应模式下的阈值
    :return: 二值化后的图像
    '''
    src = cv2.imread(filename, 0)
    if is_adapteive:
        # result = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 195, 50)
        ret, result = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        retval, result = cv2.threshold(src, 95, 255, cv2.THRESH_BINARY)
        kernel = np.uint8(np.zeros((5, 5)))
        for x in range(5):
            kernel[x, 2] = 1
            kernel[2, x] = 1
            # 腐蚀图像
        dst = result
        cv2.bitwise_not(result, dst)
        eroded = cv2.erode(dst, kernel)
        dilite = cv2.dilate(eroded, kernel)
        cv2.bitwise_not(dilite, result)

    return result


if __name__ == '__main__':
    filename = folder + '6.jpg'
    # 获取连通域
    rect_list = find_contours(filename)
    draw_rect_line(filename=filename, rect_list=rect_list, windowsname='find_contours', save=True,
                   save_filename='1_find_contours')
    # 连通域排序，从左到右，从上到下
    bubble_sort(rect_list)
    draw_rect_line(filename, rect_list, 'sort', True, 'ocr_yi_sort')

    # 连通域融合
    rect_list = combine_connected_domain(rect_list)
    draw_rect_line(filename, rect_list, 'combine', True, 'ocr_yi_combine')

    save_image(filename, rect_list)
    print('finish')
    cv2.waitKey()
