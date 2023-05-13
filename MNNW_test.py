import numpy
import math
import cv2 as cv
from numpy import *
from scipy import optimize, odr
import functools
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def type_M(bo_feng, bo_gu):
    flag1 = 0
    flag2 = 0
    gu_to_feng_point = []
    feng_to_gu_point = []
    FLAG1 = 0
    FLAG2 = 0
    last_bo_gu = []
    last_bo_feng = []
    bo_feng.pop(0)
    bo_feng.pop(len(bo_feng) - 1)
    print(bo_feng)
    print(bo_gu)
    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            cv.line(un8, (bo_gu[flag2][0], bo_gu[flag2][1]), (bo_feng[flag2][0], bo_feng[flag2][1]), (255, 255, 255), 1)
            xielv_k_gu = (bo_feng[flag2][1] - bo_gu[flag2][1]) / (bo_feng[flag2][0] - bo_gu[flag2][0])
            mid_point_x_gu = (bo_gu[flag2][0] + bo_feng[flag2][0]) / 2
            mid_point_y_gu = (bo_gu[flag2][1] + bo_feng[flag2][1]) / 2
            zhongxian_k_gu = -1 / xielv_k_gu
            zhongxian_k_gu = round(zhongxian_k_gu, 3)
            chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
            chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
            cv.line(un8, (int(chuixian_x1_gu), 190), (int(chuixian_x2_gu), 310), (255, 255, 255), 1)
            flag2 += 1
            temp1 = []
            temp1.append(zhongxian_k_gu)
            temp1.append([mid_point_x_gu, mid_point_y_gu])
            gu_to_feng_point.append(temp1)

        else:
            cv.line(un8, (bo_feng[flag1][0], bo_feng[flag1][1]), (bo_gu[flag1 + 1][0], bo_gu[flag1 + 1][1]), (255, 255, 255), 1)
            xielv_k_feng = (bo_feng[flag1][1] - bo_gu[flag1 + 1][1]) / (bo_feng[flag1][0] - bo_gu[flag1 + 1][0])
            mid_point_x_feng = (bo_feng[flag1][0] + bo_gu[flag1 + 1][0]) / 2
            mid_point_y_feng = (bo_feng[flag1][1] + bo_gu[flag1 + 1][1]) / 2
            zhongxian_k_feng = -1 / xielv_k_feng
            zhongxian_k_feng = round(zhongxian_k_feng, 3)
            chuixian_x1_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (190 - mid_point_y_feng)
            chuixian_x2_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (310 - mid_point_y_feng)
            cv.line(un8, (round(chuixian_x1_feng), 190), (round(chuixian_x2_feng), 310), (255, 255, 255), 1)
            flag1 += 1
            temp2 = []
            temp2.append(zhongxian_k_feng)
            temp2.append([mid_point_x_feng, mid_point_y_feng])
            feng_to_gu_point.append(temp2)

    for i in range(len(gu_to_feng_point)):
        x1 = gu_to_feng_point[FLAG1][1][0]
        y1 = gu_to_feng_point[FLAG1][1][1]
        k1 = gu_to_feng_point[FLAG1][0]
        x2 = feng_to_gu_point[FLAG1][1][0]
        y2 = feng_to_gu_point[FLAG1][1][1]
        k2 = feng_to_gu_point[FLAG1][0]
        x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
        y = k1 * (x - x1) + y1
        last_bo_gu.append([round(x, 3), round(y, 3)])
        FLAG1 += 1
    last_bo_gu = numpy.array(last_bo_gu)

    for i in range(len(feng_to_gu_point) - 1):
        x1 = feng_to_gu_point[FLAG2][1][0]
        y1 = feng_to_gu_point[FLAG2][1][1]
        k1 = feng_to_gu_point[FLAG2][0]
        x2 = gu_to_feng_point[FLAG2 + 1][1][0]
        y2 = gu_to_feng_point[FLAG2 + 1][1][1]
        k2 = gu_to_feng_point[FLAG2 + 1][0]
        x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
        y = k1 * (x - x1) + y1
        last_bo_feng.append([round(x, 3), round(y, 3)])
        FLAG2 += 1
    last_bo_feng = numpy.array(last_bo_feng)

    return last_bo_gu, last_bo_feng


def type_W(bo_feng, bo_gu):
    flag1 = 0
    flag2 = 0
    gu_to_feng_point = []
    feng_to_gu_point = []
    FLAG1 = 0
    FLAG2 = 0
    last_bo_gu = []
    last_bo_feng = []
    bo_gu.pop(0)
    bo_gu.pop(len(bo_gu) - 1)
    print(bo_feng)
    print(bo_gu)

    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            cv.line(canny, (bo_feng[flag1][0], bo_feng[flag1][1]), (bo_gu[flag1][0], bo_gu[flag1][1]), (255, 255, 255), 1)
            xielv_k_feng = (bo_gu[flag1][1] - bo_feng[flag1][1]) / (bo_gu[flag1][0] - bo_feng[flag1][0])
            mid_point_x_feng = (bo_feng[flag1][0] + bo_gu[flag1][0]) / 2
            mid_point_y_feng = (bo_feng[flag1][1] + bo_gu[flag1][1]) / 2
            zhongxian_k_feng = -1 / xielv_k_feng
            zhongxian_k_feng = round(zhongxian_k_feng, 3)
            chuixian_x1_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (190 - mid_point_y_feng)
            chuixian_x2_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (310 - mid_point_y_feng)
            cv.line(canny, (round(chuixian_x1_feng), 190), (round(chuixian_x2_feng), 310), (255, 255, 255), 1)
            flag1 += 1
            temp1 = []
            temp1.append(zhongxian_k_feng)
            temp1.append([mid_point_x_feng, mid_point_y_feng])
            feng_to_gu_point.append(temp1)

        else:
            cv.line(canny, (bo_gu[flag2][0], bo_gu[flag2][1]), (bo_feng[flag2 + 1][0], bo_feng[flag2 + 1][1]), (255, 255, 255), 1)
            xielv_k_gu = (bo_feng[flag2 + 1][1] - bo_gu[flag2][1]) / (bo_feng[flag2 + 1][0] - bo_gu[flag2][0])
            mid_point_x_gu = (bo_gu[flag2][0] + bo_feng[flag2 + 1][0]) / 2
            mid_point_y_gu = (bo_gu[flag2][1] + bo_feng[flag2 + 1][1]) / 2
            zhongxian_k_gu = -1 / xielv_k_gu
            zhongxian_k_gu = round(zhongxian_k_gu, 3)
            chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
            chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
            cv.line(canny, (int(chuixian_x1_gu), 190), (int(chuixian_x2_gu), 310), (255, 255, 255), 1)
            flag2 += 1
            temp2 = []
            temp2.append(zhongxian_k_gu)
            temp2.append([mid_point_x_gu, mid_point_y_gu])
            gu_to_feng_point.append(temp2)

    for i in range(len(feng_to_gu_point)):
        x1 = feng_to_gu_point[FLAG1][1][0]
        y1 = feng_to_gu_point[FLAG1][1][1]
        k1 = feng_to_gu_point[FLAG1][0]
        x2 = gu_to_feng_point[FLAG1][1][0]
        y2 = gu_to_feng_point[FLAG1][1][1]
        k2 = gu_to_feng_point[FLAG1][0]
        x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
        y = k1 * (x - x1) + y1
        last_bo_feng.append([round(x, 3), round(y, 3)])
        FLAG1 += 1
    last_bo_feng = numpy.array(last_bo_feng)
    for i in range(len(gu_to_feng_point) - 1):
        x1 = gu_to_feng_point[FLAG2][1][0]
        y1 = gu_to_feng_point[FLAG2][1][1]
        k1 = gu_to_feng_point[FLAG2][0]
        x2 = feng_to_gu_point[FLAG2 + 1][1][0]
        y2 = feng_to_gu_point[FLAG2 + 1][1][1]
        k2 = feng_to_gu_point[FLAG2 + 1][0]
        x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
        y = k1 * (x - x1) + y1
        last_bo_gu.append([round(x, 3), round(y, 3)])
        FLAG2 += 1
    last_bo_gu = numpy.array(last_bo_gu)
    return last_bo_gu, last_bo_feng


def type_INV_N(bo_feng, bo_gu):
    flag1 = 0
    flag2 = 0
    gu_to_feng_point = []
    feng_to_gu_point = []
    FLAG1 = 0
    FLAG2 = 0
    last_bo_gu = []
    last_bo_feng = []
    bo_gu.pop(0)
    bo_feng.pop(len(bo_feng)-1)
    print(bo_feng)
    print(bo_gu)
    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            cv.line(canny, (bo_feng[flag1][0], bo_feng[flag1][1]), (bo_gu[flag1][0], bo_gu[flag1][1]), (255, 255, 255), 1)
            xielv_k_feng = (bo_gu[flag1][1] - bo_feng[flag1][1]) / (bo_gu[flag1][0] - bo_feng[flag1][0])
            mid_point_x_feng = (bo_feng[flag1][0] + bo_gu[flag1][0]) / 2
            mid_point_y_feng = (bo_feng[flag1][1] + bo_gu[flag1][1]) / 2
            zhongxian_k_feng = -1 / xielv_k_feng
            zhongxian_k_feng = round(zhongxian_k_feng, 3)
            chuixian_x1_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (190 - mid_point_y_feng)
            chuixian_x2_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (310 - mid_point_y_feng)
            cv.line(canny, (round(chuixian_x1_feng), 190), (round(chuixian_x2_feng), 310), (255, 255, 255), 1)
            flag1 += 1
            temp1 = []
            temp1.append(zhongxian_k_feng)
            temp1.append([mid_point_x_feng, mid_point_y_feng])
            feng_to_gu_point.append(temp1)
        else:
            if flag2 < (len(bo_gu) - 1):
                cv.line(canny, (bo_gu[flag2][0], bo_gu[flag2][1]), (bo_feng[flag2 + 1][0], bo_feng[flag2 + 1][1]), (255, 255, 255), 1)
                xielv_k_gu = (bo_feng[flag2 + 1][1] - bo_gu[flag2][1]) / (bo_feng[flag2 + 1][0] - bo_gu[flag2][0])
                mid_point_x_gu = (bo_gu[flag2][0] + bo_feng[flag2 + 1][0]) / 2
                mid_point_y_gu = (bo_gu[flag2][1] + bo_feng[flag2 + 1][1]) / 2
                zhongxian_k_gu = -1 / xielv_k_gu
                zhongxian_k_gu = round(zhongxian_k_gu, 3)
                chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
                chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
                cv.line(canny, (int(chuixian_x1_gu), 190), (int(chuixian_x2_gu), 310), (255, 255, 255), 1)
                flag2 += 1
                temp2 = []
                temp2.append(zhongxian_k_gu)
                temp2.append([mid_point_x_gu, mid_point_y_gu])
                gu_to_feng_point.append(temp2)
    print(len(feng_to_gu_point))
    print(len(gu_to_feng_point))
    for i in range(len(feng_to_gu_point) - 1):
        x1 = feng_to_gu_point[FLAG1][1][0]
        y1 = feng_to_gu_point[FLAG1][1][1]
        k1 = feng_to_gu_point[FLAG1][0]
        x2 = gu_to_feng_point[FLAG1][1][0]
        y2 = gu_to_feng_point[FLAG1][1][1]
        k2 = gu_to_feng_point[FLAG1][0]
        x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
        y = k1 * (x - x1) + y1
        last_bo_feng.append([round(x, 3), round(y, 3)])
        FLAG1 += 1
    last_bo_feng = numpy.array(last_bo_feng)
    for i in range(len(gu_to_feng_point)):
        x1 = gu_to_feng_point[FLAG2][1][0]
        y1 = gu_to_feng_point[FLAG2][1][1]
        k1 = gu_to_feng_point[FLAG2][0]
        x2 = feng_to_gu_point[FLAG2 + 1][1][0]
        y2 = feng_to_gu_point[FLAG2 + 1][1][1]
        k2 = feng_to_gu_point[FLAG2 + 1][0]
        x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
        y = k1 * (x - x1) + y1
        last_bo_gu.append([round(x, 3), round(y, 3)])
        FLAG2 += 1
    last_bo_gu = numpy.array(last_bo_gu)
    return last_bo_gu, last_bo_feng


def type_N(bo_feng, bo_gu):
    flag1 = 0
    flag2 = 0
    gu_to_feng_point = []
    feng_to_gu_point = []
    FLAG1 = 0
    FLAG2 = 0
    last_bo_gu = []
    last_bo_feng = []
    bo_feng.pop(0)
    bo_gu.pop(len(bo_gu)-1)
    print(bo_feng)
    print(bo_gu)
    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            cv.line(un8, (bo_gu[flag1][0], bo_gu[flag1][1]), (bo_feng[flag1][0], bo_feng[flag1][1]), (255, 255, 255), 1)
            xielv_k_gu = (bo_feng[flag1][1]-bo_gu[flag1][1])/(bo_feng[flag1][0]-bo_gu[flag1][0])
            mid_point_x_gu = (bo_feng[flag1][0] + bo_gu[flag1][0]) / 2
            mid_point_y_gu = (bo_feng[flag1][1] + bo_gu[flag1][1]) / 2
            zhongxian_k_gu = -1 / xielv_k_gu
            zhongxian_k_gu = round(zhongxian_k_gu, 3)
            chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
            chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
            cv.line(un8, (int(chuixian_x1_gu), 190), (int(chuixian_x2_gu), 310), (255, 255, 255), 1)
            flag1 += 1
            temp1 = []
            temp1.append(zhongxian_k_gu)
            temp1.append([mid_point_x_gu, mid_point_y_gu])
            gu_to_feng_point.append(temp1)

        else:
            if flag2 < (len(bo_feng) - 1):
                cv.line(un8, (bo_feng[flag2][0], bo_feng[flag2][1]), (bo_gu[flag2 + 1][0], bo_gu[flag2 + 1][1]), (255, 255, 255), 1)
                xielv_k_gu = (bo_gu[flag2 + 1][1] - bo_feng[flag2][1]) / (bo_gu[flag2 + 1][0] - bo_feng[flag2][0])
                mid_point_x_gu = (bo_feng[flag2][0] + bo_gu[flag2 + 1][0]) / 2
                mid_point_y_gu = (bo_feng[flag2][1] + bo_gu[flag2 + 1][1]) / 2
                zhongxian_k_gu = -1 / xielv_k_gu
                zhongxian_k_gu = round(zhongxian_k_gu, 3)
                chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
                chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
                cv.line(un8, (int(chuixian_x1_gu), 190), (int(chuixian_x2_gu), 310), (255, 255, 255), 1)
                flag2 += 1
                temp2 = []
                temp2.append(zhongxian_k_gu)
                temp2.append([mid_point_x_gu, mid_point_y_gu])
                feng_to_gu_point.append(temp2)
    print(len(gu_to_feng_point))
    print(len(feng_to_gu_point))
    for i in range(len(gu_to_feng_point)-1):
        x1 = gu_to_feng_point[FLAG1][1][0]
        y1 = gu_to_feng_point[FLAG1][1][1]
        k1 = gu_to_feng_point[FLAG1][0]
        x2 = feng_to_gu_point[FLAG1][1][0]
        y2 = feng_to_gu_point[FLAG1][1][1]
        k2 = feng_to_gu_point[FLAG1][0]
        x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
        y = k1 * (x - x1) + y1
        last_bo_gu.append([round(x, 3), round(y, 3)])
        FLAG1 += 1
    last_bo_gu = numpy.array(last_bo_gu)

    for i in range(len(feng_to_gu_point)):
        x1 = feng_to_gu_point[FLAG2][1][0]
        y1 = feng_to_gu_point[FLAG2][1][1]
        k1 = feng_to_gu_point[FLAG2][0]
        x2 = gu_to_feng_point[FLAG2 + 1][1][0]
        y2 = gu_to_feng_point[FLAG2 + 1][1][1]
        k2 = gu_to_feng_point[FLAG2 + 1][0]
        x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
        y = k1 * (x - x1) + y1
        last_bo_feng.append([round(x, 3), round(y, 3)])
        FLAG2 += 1
    last_bo_feng = numpy.array(last_bo_feng)
    return last_bo_gu, last_bo_feng


image = cv.imread('li0.jpg', 1)
img1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转灰度图
_, otsu = cv.threshold(img1, None, 255, cv.THRESH_OTSU)  # OTSU阈值分割
otsu = cv.erode(otsu, kernel=numpy.ones((3, 3), numpy.uint8))  # 腐蚀去毛边
canny = cv.Canny(otsu, 100, 500, None, 3)  # 边缘检测
# _, th2 = cv.threshold(canny, 100, 255, cv.THRESH_BINARY_INV)  # 颜色反转
set = []
for j in range(canny.shape[1]):
    for i in range(canny.shape[0]):
        if canny[i][j] == 0:
            continue
        else:
            set.append([j, i])
print(set)

un8 = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint8) + 200
for i in set:
    un8[i[1]][i[0]] = [255, 0, 0]

# type-m
# bo_feng = [[0, 256],  [58, 247], [132, 244], [217, 246], [290, 248], [357, 245], [426, 244], [505, 246], [580, 244], [664, 246],    [699, 264]]
# bo_gu = [         [10, 262], [75, 304], [168, 263], [242, 267], [315, 264], [390, 264], [464, 266], [537, 273], [620, 303], [696, 265]]
# tempi, tempj = type_M(bo_feng, bo_gu)

### type-w
# bo_feng = [     [58, 247], [132, 244], [217, 246], [290, 248], [357, 245], [426, 244], [505, 246], [580, 244], [664, 246]]
# bo_gu = [[21, 256],   [75, 304], [168, 263], [242, 267], [315, 264], [390, 264], [464, 266], [537, 273], [620, 303],       [687, 257]]
# tempi, tempj = type_W(bo_feng, bo_gu)

## type-INV-n
# bo_feng = [       [58, 247], [132, 244], [217, 246], [290, 248], [357, 245], [426, 244], [505, 246], [580, 244], [664, 246],     [699, 264]]
# bo_gu = [[21, 256],    [75, 304], [168, 263], [242, 267], [315, 264], [390, 264], [464, 266], [537, 273], [620, 303],[696, 265]]
# tempi, tempj = type_INV_N(bo_feng, bo_gu)

## type-n
bo_feng = [[0, 256],    [58, 247], [132, 244], [217, 246], [290, 248], [357, 245], [426, 244], [505, 246], [580, 244], [664, 246]]
bo_gu = [         [10, 262], [75, 304], [168, 263], [242, 267], [315, 264], [390, 264], [464, 266], [537, 273], [620, 303],      [696, 265]]
tempi, tempj = type_N(bo_feng, bo_gu)

for i in tempi:
    un8[int(i[1]), int(i[0])] = [0, 0, 255]
    un8[int(i[1] - 1), int(i[0])] = [0, 0, 255]
    un8[int(i[1] + 1), int(i[0])] = [0, 0, 255]
    un8[int(i[1]), int(i[0] - 1)] = [0, 0, 255]
    un8[int(i[1]), int(i[0] + 1)] = [0, 0, 255]
    un8[int(i[1] - 1), int(i[0] - 1)] = [0, 0, 255]
    un8[int(i[1] - 1), int(i[0] + 1)] = [0, 0, 255]
    un8[int(i[1] + 1), int(i[0] - 1)] = [0, 0, 255]
    un8[int(i[1] + 1), int(i[0] + 1)] = [0, 0, 255]
# for i in tempj:
#     un8[int(i[1]), int(i[0])] = [0, 255, 0]           # 波峰
#     un8[int(i[1] - 1), int(i[0])] = [0, 255, 0]
#     un8[int(i[1] + 1), int(i[0])] = [0, 255, 0]
#     un8[int(i[1]), int(i[0] - 1)] = [0, 255, 0]
#     un8[int(i[1]), int(i[0] + 1)] = [0, 255, 0]
#     un8[int(i[1] - 1), int(i[0] - 1)] = [0, 255, 0]
#     un8[int(i[1] - 1), int(i[0] + 1)] = [0, 255, 0]
#     un8[int(i[1] + 1), int(i[0] - 1)] = [0, 255, 0]
#     un8[int(i[1] + 1), int(i[0] + 1)] = [0, 255, 0]

plt.figure()
plt.grid(False)
plt.imshow(canny, cmap=plt.cm.gray)
plt.figure()
plt.grid(False)
plt.imshow(un8, cmap=plt.cm.gray)
plt.show()
