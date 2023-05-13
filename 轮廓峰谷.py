import numpy
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import *

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def delete_repit(list):
    temp_set = []
    mid_num_x = 0
    mid_num_y = 0
    temp_set.append(list[0])
    for i in range(1, len(list)):
        if list[i][0] - list[0][0] < 40:
            temp_set.append(list[i])
    for i in range(0, len(temp_set)):
        mid_num_x += temp_set[i][0]
        mid_num_y += temp_set[i][1]
    mid_num_x = mid_num_x / len(temp_set)
    mid_num_y = mid_num_y / len(temp_set)
    mid_num_x = int(mid_num_x)
    mid_num_y = int(mid_num_y)
    return len(temp_set), [mid_num_x, mid_num_y]


image = cv.imread('li0.jpg', 1)
img1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转灰度图
_, otsu = cv.threshold(img1, None, 255, cv.THRESH_OTSU)  # OTSU阈值分割
otsu = cv.erode(otsu, kernel=numpy.ones((3, 3), numpy.uint8))  # 腐蚀去毛边
canny = cv.Canny(otsu, 100, 500, None, 3)  # 边缘检测
set1 = []  # 原始图像边缘结合
for j in range(canny.shape[1]):
    for i in range(canny.shape[0]):
        if canny[i][j] == 0:
            continue
        else:
            set1.append([j, i])
# print(set1)
# print(len(set1))

add_begin = set1[0][1]
add_final = set1[len(set1) - 1][1]
# print(add_begin, add_final)
img2 = numpy.pad(canny, ((0, 0), (25, 25)), 'constant', constant_values=((0, 0), (0, 0)))
un8 = numpy.zeros((img2.shape[0], img2.shape[1], 3), numpy.uint8) + 200
un9 = un8.copy()

for i in range(25):
    img2[add_begin][i] = 255
    img2[add_final][img2.shape[1] - i - 1] = 255

set2 = []  # 扩张图像边缘集合

for j in range(img2.shape[1]):
    for i in range(img2.shape[0]):
        if img2[i][j] == 0:
            continue
        else:
            un8[i][j] = [0, 0, 0]
            un9[i][j] = [0, 0, 0]
            set2.append([j, i])
# print(set2)
# print(len(set2))

lst1 = []  # 波谷集合
lst2 = []  # 波峰集合
for i in range(25, len(set2) - 25):
    point_set = (set2[i - 25][1], set2[i - 24][1], set2[i - 23][1], set2[i - 22][1], set2[i - 21][1],
                 set2[i - 20][1], set2[i - 19][1], set2[i - 18][1], set2[i - 17][1], set2[i - 16][1],
                 set2[i - 15][1], set2[i - 14][1], set2[i - 13][1], set2[i - 12][1], set2[i - 11][1],
                 set2[i - 10][1], set2[i - 9][1], set2[i - 8][1], set2[i - 7][1], set2[i - 6][1],
                 set2[i - 5][1], set2[i - 4][1], set2[i - 3][1], set2[i - 2][1], set2[i - 1][1],
                 set2[i + 1][1], set2[i + 2][1], set2[i + 3][1], set2[i + 4][1], set2[i + 5][1],
                 set2[i + 6][1], set2[i + 7][1], set2[i + 8][1], set2[i + 9][1], set2[i + 10][1],
                 set2[i + 11][1], set2[i + 12][1], set2[i + 13][1], set2[i + 14][1], set2[i + 15][1],
                 set2[i + 16][1], set2[i + 17][1], set2[i + 18][1], set2[i + 19][1], set2[i + 20][1],
                 set2[i + 21][1], set2[i + 22][1], set2[i + 23][1], set2[i + 24][1], set2[i + 25][1])

    max_y_label = max(point_set)
    min_y_label = min(point_set)
    if set2[i][1] >= max_y_label:
        # print(set2[i][0], set2[i][1])
        lst1.append([set2[i][0], set2[i][1]])
        # un8[set2[i][1], set2[i][0]] = [0, 255, 0]
    if set2[i][1] <= min_y_label:
        # print(set2[i][0], set2[i][1])
        lst2.append([set2[i][0], set2[i][1]])
        # un9[set2[i][1], set2[i][0]] = [0, 0, 255]
    else:
        continue

print(lst1)
print(lst2)
lst3 = lst1.copy()  # 谷
lst4 = lst2.copy()  # 峰
bofeng_mid_point = []
bogu_mid_point = []

for i in lst1:
    un8[i[1], i[0]] = (0, 255, 0)
    if len(lst3) != 0:
        return_num, return_point = delete_repit(lst3)
        bogu_mid_point.append(return_point)
        for i in range(return_num):
            lst3.pop(0)
    else:
        pass

for i in lst2:
    un8[i[1], i[0]] = (255, 0, 0)
    if len(lst4) != 0:
        return_num, return_point = delete_repit(lst4)
        bofeng_mid_point.append(return_point)
        for i in range(return_num):
            lst4.pop(0)
    else:
        pass

print(bogu_mid_point)
print(bofeng_mid_point)

for i in range(len(bogu_mid_point)):
    un8[bogu_mid_point[i][1], bogu_mid_point[i][0]] = [0, 255, 0]
for i in range(len(bofeng_mid_point)):
    un8[bofeng_mid_point[i][1], bofeng_mid_point[i][0]] = [255, 0, 0]

# _,th2 = cv.threshold(img2,100,255,cv.THRESH_BINARY_INV)  # 颜色反转
# cv.imshow('th2',th2)
plt.figure()
plt.title('img2')
plt.imshow(img2, cmap=plt.cm.gray)
plt.figure()
plt.title('波谷')
plt.imshow(un8, cmap=plt.cm.gray)
# plt.figure()
# plt.title('波峰')
# plt.imshow(un9, cmap=plt.cm.gray)
plt.show()
