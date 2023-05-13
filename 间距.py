import numpy
import cv2
import matplotlib.pyplot as plt
from numpy import *
import math

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

def delete_repit(list):
    temp_set = []
    mid_num_x = 0
    mid_num_y = 0
    temp_set.append(list[0])  # 存入列表第一个点的坐标
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

def type_M(bo_feng, bo_gu):
    flag1 = 0
    flag2 = 0
    gu_to_feng_point = []
    feng_to_gu_point = []
    FLAG1 = 0
    FLAG2 = 0
    last_bo_gu = []
    last_bo_feng = []
    bo_feng.pop(0)  # 删除两个端点
    bo_feng.pop(len(bo_feng) - 1)  # 删除两个端点
    # print(bo_feng)
    # print(bo_gu)
    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            xielv_k_gu = (bo_feng[flag2][1] - bo_gu[flag2][1]) / (bo_feng[flag2][0] - bo_gu[flag2][0])  # 连线斜率
            mid_point_x_gu = (bo_gu[flag2][0] + bo_feng[flag2][0]) / 2  # 连线中点
            mid_point_y_gu = (bo_gu[flag2][1] + bo_feng[flag2][1]) / 2  # 连线中点
            zhongxian_k_gu = -1 / xielv_k_gu  # 中垂线斜率
            zhongxian_k_gu = round(zhongxian_k_gu, 3)  # 斜率保留三位小数
            chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)  # 中垂线从190画到310
            chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)  # 中垂线从190画到310
            flag2 += 1
            temp1 = []
            temp1.append(zhongxian_k_gu)  # 中垂线斜率
            temp1.append([mid_point_x_gu, mid_point_y_gu])  # 中垂线和连线的交点
            gu_to_feng_point.append(temp1)  # 保留中垂线的斜率，一个坐标点，点斜式

        else:
            xielv_k_feng = (bo_feng[flag1][1] - bo_gu[flag1 + 1][1]) / (bo_feng[flag1][0] - bo_gu[flag1 + 1][0])
            mid_point_x_feng = (bo_feng[flag1][0] + bo_gu[flag1 + 1][0]) / 2
            mid_point_y_feng = (bo_feng[flag1][1] + bo_gu[flag1 + 1][1]) / 2
            zhongxian_k_feng = -1 / xielv_k_feng
            zhongxian_k_feng = round(zhongxian_k_feng, 3)
            chuixian_x1_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (190 - mid_point_y_feng)
            chuixian_x2_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (310 - mid_point_y_feng)
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
    # last_bo_gu = numpy.array(last_bo_gu)

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
    # last_bo_feng = numpy.array(last_bo_feng)

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
    # print(bo_feng)
    # print(bo_gu)
    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            xielv_k_feng = (bo_gu[flag1][1] - bo_feng[flag1][1]) / (bo_gu[flag1][0] - bo_feng[flag1][0])
            mid_point_x_feng = (bo_feng[flag1][0] + bo_gu[flag1][0]) / 2
            mid_point_y_feng = (bo_feng[flag1][1] + bo_gu[flag1][1]) / 2
            zhongxian_k_feng = -1 / xielv_k_feng
            zhongxian_k_feng = round(zhongxian_k_feng, 3)
            chuixian_x1_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (190 - mid_point_y_feng)
            chuixian_x2_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (310 - mid_point_y_feng)
            flag1 += 1
            temp1 = []
            temp1.append(zhongxian_k_feng)
            temp1.append([mid_point_x_feng, mid_point_y_feng])
            feng_to_gu_point.append(temp1)

        else:
            xielv_k_gu = (bo_feng[flag2 + 1][1] - bo_gu[flag2][1]) / (bo_feng[flag2 + 1][0] - bo_gu[flag2][0])
            mid_point_x_gu = (bo_gu[flag2][0] + bo_feng[flag2 + 1][0]) / 2
            mid_point_y_gu = (bo_gu[flag2][1] + bo_feng[flag2 + 1][1]) / 2
            zhongxian_k_gu = -1 / xielv_k_gu
            zhongxian_k_gu = round(zhongxian_k_gu, 3)
            chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
            chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
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
    # last_bo_feng = numpy.array(last_bo_feng)
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
    # last_bo_gu = numpy.array(last_bo_gu)
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
    bo_feng.pop(len(bo_feng) - 1)
    # print(bo_feng)
    # print(bo_gu)
    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            xielv_k_feng = (bo_gu[flag1][1] - bo_feng[flag1][1]) / (bo_gu[flag1][0] - bo_feng[flag1][0])
            mid_point_x_feng = (bo_feng[flag1][0] + bo_gu[flag1][0]) / 2
            mid_point_y_feng = (bo_feng[flag1][1] + bo_gu[flag1][1]) / 2
            zhongxian_k_feng = -1 / xielv_k_feng
            zhongxian_k_feng = round(zhongxian_k_feng, 3)
            chuixian_x1_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (190 - mid_point_y_feng)
            chuixian_x2_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (310 - mid_point_y_feng)
            flag1 += 1
            temp1 = []
            temp1.append(zhongxian_k_feng)
            temp1.append([mid_point_x_feng, mid_point_y_feng])
            feng_to_gu_point.append(temp1)
        else:
            if flag2 < (len(bo_gu) - 1):
                xielv_k_gu = (bo_feng[flag2 + 1][1] - bo_gu[flag2][1]) / (bo_feng[flag2 + 1][0] - bo_gu[flag2][0])
                mid_point_x_gu = (bo_gu[flag2][0] + bo_feng[flag2 + 1][0]) / 2
                mid_point_y_gu = (bo_gu[flag2][1] + bo_feng[flag2 + 1][1]) / 2
                zhongxian_k_gu = -1 / xielv_k_gu
                zhongxian_k_gu = round(zhongxian_k_gu, 3)
                chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
                chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
                flag2 += 1
                temp2 = []
                temp2.append(zhongxian_k_gu)
                temp2.append([mid_point_x_gu, mid_point_y_gu])
                gu_to_feng_point.append(temp2)

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
    # last_bo_feng = numpy.array(last_bo_feng)
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
    # last_bo_gu = numpy.array(last_bo_gu)
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
    bo_gu.pop(len(bo_gu) - 1)
    # print(bo_feng)
    # print(bo_gu)
    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            xielv_k_gu = (bo_feng[flag1][1] - bo_gu[flag1][1]) / (bo_feng[flag1][0] - bo_gu[flag1][0])
            mid_point_x_gu = (bo_feng[flag1][0] + bo_gu[flag1][0]) / 2
            mid_point_y_gu = (bo_feng[flag1][1] + bo_gu[flag1][1]) / 2
            zhongxian_k_gu = -1 / xielv_k_gu
            zhongxian_k_gu = round(zhongxian_k_gu, 3)
            chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
            chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
            flag1 += 1
            temp1 = []
            temp1.append(zhongxian_k_gu)
            temp1.append([mid_point_x_gu, mid_point_y_gu])
            gu_to_feng_point.append(temp1)
        else:
            if flag2 < (len(bo_feng) - 1):
                xielv_k_gu = (bo_gu[flag2 + 1][1] - bo_feng[flag2][1]) / (bo_gu[flag2 + 1][0] - bo_feng[flag2][0])
                mid_point_x_gu = (bo_feng[flag2][0] + bo_gu[flag2 + 1][0]) / 2
                mid_point_y_gu = (bo_feng[flag2][1] + bo_gu[flag2 + 1][1]) / 2
                zhongxian_k_gu = -1 / xielv_k_gu
                zhongxian_k_gu = round(zhongxian_k_gu, 3)
                chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
                chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
                flag2 += 1
                temp2 = []
                temp2.append(zhongxian_k_gu)
                temp2.append([mid_point_x_gu, mid_point_y_gu])
                feng_to_gu_point.append(temp2)

    for i in range(len(gu_to_feng_point) - 1):
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
    # last_bo_gu = numpy.array(last_bo_gu)
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
    # last_bo_feng = numpy.array(last_bo_feng)
    return last_bo_gu, last_bo_feng

def BiLinear_interpolation(img, dstH, dstW):  # 双线性插值
    scrH, scrW, _ = img.shape
    img = numpy.pad(img, ((0, 1), (0, 1), (0, 0)), 'constant')  # 边缘填充
    retimg = numpy.zeros((dstH, dstW, 3), dtype=numpy.uint8)  # 创建背景板
    for i in range(dstH):
        for j in range(dstW):
            scrx = (i + 1) * (scrH / dstH) - 1
            scry = (j + 1) * (scrW / dstW) - 1
            x = math.floor(scrx)  # math.floor 取整
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            retimg[i, j] = (1 - u) * (1 - v) * img[x, y] + u * (1 - v) * img[x + 1, y] + (1 - u) * v * img[x, y + 1] + u * v * img[x + 1, y + 1]
    return retimg


set1 = []  # 原始图像边缘结合
set2 = []  # 扩张图像边缘集合
lst1 = []  # 波谷集合
lst2 = []  # 波峰集合
bofeng_mid_point = []  # 提取后的波峰
bogu_mid_point = []  # 提取后的波谷
Final_value = []
Real_weigh = 2  # mm
K_pixel = 0.033  # mm/pix
Pixel_weigh = Real_weigh / K_pixel  # pix
iii = 1220
image = cv2.imread('./test/'+ str(iii) +'.jpg', 1)
# image = cv2.imread('./test/1.jpg',1)
image2 = image.copy()
# originalimg = cv2.imread('big.jpg',1)
# lunkuo_img = originalimg[648:768,2020:2272] # 轮廓图像
# lunkuo_img = BiLinear_interpolation(lunkuo_img,lunkuo_img.shape[0]*3,lunkuo_img.shape[1]*3)  # 轮廓图像放大
gray_img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度图
_, otsu = cv2.threshold(gray_img1, None, 255, cv2.THRESH_OTSU)  # OTSU阈值分割
otsu = cv2.erode(otsu, kernel=numpy.ones((7, 5), numpy.uint8))  # 腐蚀去毛边
# otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, numpy.ones((3, 3), numpy.uint8))
canny = cv2.Canny(otsu, 100, 500, None, 3)  # 边缘检测
for j in range(canny.shape[1]):
    for i in range(canny.shape[0]):
        if canny[i][j] == 0:
            continue
        else:
            set1.append([j, i])  # 原始边缘轮廓
cv2.imshow('canny',canny)
img2 = numpy.pad(canny, ((0, 0), (25, 25)), 'constant', constant_values=((0, 0), (0, 0)))  # 边缘扩充图像
un8 = numpy.zeros((img2.shape[0], img2.shape[1], 3), numpy.uint8) + 255  # 边缘扩充图像取反
for i in range(25):
    img2[set1[0][1]][i] = 255
    img2[set1[len(set1) - 1][1]][img2.shape[1] - i - 1] = 255
for j in range(img2.shape[1]):
    for i in range(img2.shape[0]):
        if img2[i][j] == 0:
            continue
        else:
            un8[i][j] = [0, 0, 0]
            set2.append([j, i])  # 扩充后边缘轮廓
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
        lst1.append([set2[i][0], set2[i][1]])  # 包含波谷的点
    if set2[i][1] <= min_y_label:
        lst2.append([set2[i][0], set2[i][1]])  # 包含波峰的点
    else:
        continue
lst3 = lst1.copy()  # 谷
lst4 = lst2.copy()  # 峰
print("lst1:", lst1)
print("lst2:", lst2)
for i in lst1:
    if len(lst3) != 0:
        return_num, return_point = delete_repit(lst3)
        bogu_mid_point.append(return_point)
        for i in range(return_num):
            lst3.pop(0)
    else:
        pass

for i in lst2:
    if len(lst4) != 0:
        return_num, return_point = delete_repit(lst4)
        bofeng_mid_point.append(return_point)
        for i in range(return_num):
            lst4.pop(0)
    else:
        pass

print('bogu_mid_point:', bogu_mid_point)
print('bofeng_mid_point:', bofeng_mid_point)

if len(bofeng_mid_point) > len(bogu_mid_point):
    tempi, tempj = type_M(bofeng_mid_point, bogu_mid_point)
    print('bogudata:', tempi)
    # print('bofengdata:',tempj)
elif len(bofeng_mid_point) < len(bogu_mid_point):
    tempi, tempj = type_W(bofeng_mid_point, bogu_mid_point)
    print('bogudata:', tempi)
    # print('bofengdata:', tempj)
else:
    if bofeng_mid_point[0][0] > bogu_mid_point[0][0] and bofeng_mid_point[len(bofeng_mid_point) - 1][0] > bogu_mid_point[len(bogu_mid_point) - 1][0]:
        tempi, tempj = type_INV_N(bofeng_mid_point, bogu_mid_point)
        print('bogudata:', tempi)
        # print('bofengdata:', tempj)
    elif bofeng_mid_point[0][0] < bogu_mid_point[0][0] and bofeng_mid_point[len(bofeng_mid_point) - 1][0] < bogu_mid_point[len(bogu_mid_point) - 1][0]:
        tempi, tempj = type_N(bofeng_mid_point, bogu_mid_point)
        print('bogudata:', tempi)
        # print('bofengdata:', tempj)
    else:
        print("工艺设计问题，请重新编写代码段！！！")

for i in range(len(bogu_mid_point)):
    un8[bogu_mid_point[i][1], bogu_mid_point[i][0]] = [0, 255, 0]
for i in range(len(bofeng_mid_point)):
    un8[bofeng_mid_point[i][1], bofeng_mid_point[i][0]] = [255, 0, 0]
# for i in tempi:
#     un8[int(i[1]), int(i[0])] = [0, 0, 255]         # 周围 9个像素点
#     un8[int(i[1] - 1), int(i[0])] = [0, 0, 255]
#     un8[int(i[1] + 1), int(i[0])] = [0, 0, 255]
#     un8[int(i[1]), int(i[0] - 1)] = [0, 0, 255]
#     un8[int(i[1]), int(i[0] + 1)] = [0, 0, 255]
#     un8[int(i[1] - 1), int(i[0] - 1)] = [0, 0, 255]
#     un8[int(i[1] - 1), int(i[0] + 1)] = [0, 0, 255]
#     un8[int(i[1] + 1), int(i[0] - 1)] = [0, 0, 255]
#     un8[int(i[1] + 1), int(i[0] + 1)] = [0, 0, 255]

# for i in tempj:
#     un8[int(i[1]), int(i[0])] = [0, 255, 0]
#     un8[int(i[1] - 1), int(i[0])] = [0, 255, 0]
#     un8[int(i[1] + 1), int(i[0])] = [0, 255, 0]
#     un8[int(i[1]), int(i[0] - 1)] = [0, 255, 0]
#     un8[int(i[1]), int(i[0] + 1)] = [0, 255, 0]
#     un8[int(i[1] - 1), int(i[0] - 1)] = [0, 255, 0]
#     un8[int(i[1] - 1), int(i[0] + 1)] = [0, 255, 0]
#     un8[int(i[1] + 1), int(i[0] - 1)] = [0, 255, 0]
#     un8[int(i[1] + 1), int(i[0] + 1)] = [0, 255, 0]


print(tempi)
for i in range(len(tempi) - 1):
    # print(tempi[i][0])
    # Final_dist = math.sqrt(tempi[i+1][0] - tempi[i][0])**2  - Pixel_weigh
    Final_dist = tempi[i + 1][0] - tempi[i][0] - Pixel_weigh
    Final_value.append(round(Final_dist * K_pixel, 3))
    # print(Final_value)
print(Final_value)
ADC = round((tempi[Final_value.index(max(Final_value)) + 1][0] - tempi[Final_value.index(max(Final_value))][0]) / 2)
# print(Final_value.index(max(Final_value)))
# print(tempi[Final_value.index(max(Final_value))], tempi[Final_value.index(max(Final_value)) + 1])
# print(round(tempi[Final_value.index(max(Final_value)) + 1][0] - tempi[Final_value.index(max(Final_value))][0]))
# print(round(tempi[Final_value.index(max(Final_value))][1] + ADC), round(tempi[Final_value.index(max(Final_value))][1] - ADC))

# y1 - [(x2-x1)/2], y1 + [(x2-x1)/2]

LT_point = (round(tempi[Final_value.index(max(Final_value))][0]) - 25, round(tempi[Final_value.index(max(Final_value))][1] - ADC*2))
# RT_point = (round(tempi[Final_value.index(max(Final_value)) + 1][0]) - 25, round(tempi[Final_value.index(max(Final_value))][1] - ADC))
# LB_point = (round(tempi[Final_value.index(max(Final_value))][0]) - 25, round(tempi[Final_value.index(max(Final_value))][1] + ADC))
RB_point = (round(tempi[Final_value.index(max(Final_value)) + 1][0]) - 25, round(tempi[Final_value.index(max(Final_value))][1] + ADC))
# print(LT_point,RT_point,LB_point,RB_point)
# for i in range(len(tempi)-1):
#     print((tempi[i+1][0]-tempi[i][0])-Pixel_weigh)

cv2.rectangle(image2, LT_point, RB_point, (0, 255, 0), 2)
cv2.putText(image2, str(max(Final_value))+'mm', (round(tempi[Final_value.index(max(Final_value))][0]) - 25,
                                            round(tempi[Final_value.index(max(Final_value))][1] - ADC*2-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, 0)
for i in tempi:
    image2[int(i[1]), int(i[0])-25] = [255, 255, 255]
    image2[int(i[1]-1), int(i[0]) - 25] = [255, 255, 255]
    image2[int(i[1]+1), int(i[0]) - 25] = [255, 255, 255]
    image2[int(i[1]), int(i[0]) - 26] = [255, 255, 255]
    image2[int(i[1]), int(i[0]) - 24] = [255, 255, 255]
    image2[int(i[1]-1), int(i[0]) - 24] = [255, 255, 255]
    image2[int(i[1]-1), int(i[0]) - 26] = [255, 255, 255]
    image2[int(i[1]+1), int(i[0]) - 24] = [255, 255, 255]
    image2[int(i[1]+1), int(i[0]) - 26] = [255, 255, 255]

cv2.imshow('image2', image2)
# cv2.imwrite('./test/'+'result_%d.jpg'%(iii), image2)
# cv2.imwrite('image2.jpg',image2)
# plt.figure()
# plt.imshow(un8, cmap=plt.cm.gray)
# plt.show()


cv2.waitKey(0)
