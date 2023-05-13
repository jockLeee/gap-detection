import cv2
from skimage import morphology
import matplotlib.pyplot as plt
import math
import numpy
from numpy import *

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
set1 = []  # 原始图像边缘结合
set2 = []  # 扩张图像边缘集合
lst1 = []  # 波谷集合
lst2 = []  # 波峰集合
bofeng_mid_point = []  #
bogu_mid_point = []

# 骨架提取
def get_skeleton(binary):
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)  # 骨架提取
    # skel, distance = morphology.medial_axis(binary, return_distance=True)
    # skeleton0 = distance * skel
    skeleton = skeleton0.astype(numpy.uint8) * 255
    return skeleton

def medial_ax(binary):
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(numpy.uint8) * 255
    return dist_on_skel

# 冒泡排序
def bubble_sort(sequence, sequence2, sequence3):
    # 遍历的趟数，n个元素，遍历n-1趟
    for i in range(1, len(sequence)):
        # 从头遍历列表
        for j in range(0, len(sequence) - 1):
            # 遍历过程中，若前者数值大于后者，则交换
            if sequence[j] > sequence[j + 1]:
                # 注意，python中列表交换，不需要中间变量，可直接交换
                sequence[j], sequence[j + 1] = sequence[j + 1], sequence[j]
                sequence2[j], sequence2[j + 1] = sequence2[j + 1], sequence2[j]
                sequence3[j], sequence3[j + 1] = sequence3[j + 1], sequence3[j]
    # 返回处理完成后的列表
    return sequence

# 排序去除重复元素
def delete_num(temp, temp1, temp2):
    k = 1
    for i in range(0, len(temp) - 1):
        for j in range(k, len(temp)):
            if temp[j] - temp[i] < 5:
                temp.pop(i)
                temp1.pop(i)
                temp2.pop(i)
                k += 1
                break
            else:
                k += 1
                break
    return temp

# 平均灰度值
def average_Grayval(img):
    sum = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum += img[i, j]
    average = sum / img.shape[0] / img.shape[1]
    return average

# 改进灰度均衡化
def get_strengthen_gray(image, precision, low_value):
    save_dir = ".\\save"
    path = "./save/"
    x = int(image.shape[0] / precision)  # 分块长度
    for i in range(0, precision + 1):
        un8 = numpy.zeros((x, image.shape[1], 3), numpy.uint8)
        un9 = image[i * x:(i + 1) * x, 0:image.shape[1]]
        sum = 0
        for k in range(0, un9.shape[0]):
            for j in range(0, un9.shape[1]):
                sum += un9[k][j]
        average_gray = sum / (un9.shape[0] * un9.shape[1])
        if average_gray < low_value:
            un7 = cv2.equalizeHist(un9)
        else:
            un7 = un9
        cv2.imwrite(save_dir + '/' + '%d.jpg' % (i), un7)
    save_path = path + str(0) + ".jpg"
    img_out = cv2.imread(save_path)
    num = precision + 1
    for i in range(1, num):
        save_path = path + str(i) + ".jpg"
        img_tmp = cv2.imread(save_path)
        img_out = numpy.concatenate((img_out, img_tmp), axis=0)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("%d.jpg" % (num), img_out)
    return img_out

def BiLinear_interpolation(img,dstH,dstW):  # 双线性插值
    scrH,scrW,_=img.shape
    img=numpy.pad(img,((0,1),(0,1),(0,0)),'constant')  # 边缘填充
    retimg=numpy.zeros((dstH,dstW,3),dtype=numpy.uint8)   # 创建背景板
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH/dstH)-1
            scry=(j+1)*(scrW/dstW)-1
            x=math.floor(scrx)      # math.floor 取整
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            retimg[i,j]=(1-u)*(1-v)*img[x,y] + u*(1-v)*img[x+1,y] + (1-u)*v*img[x,y+1] + u*v*img[x+1,y+1]
    return retimg

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
    # print(bo_feng)
    # print(bo_gu)
    for i in range(len(bo_feng) + len(bo_gu) - 1):
        if i % 2 == 0:
            xielv_k_gu = (bo_feng[flag2][1] - bo_gu[flag2][1]) / (bo_feng[flag2][0] - bo_gu[flag2][0])
            mid_point_x_gu = (bo_gu[flag2][0] + bo_feng[flag2][0]) / 2
            mid_point_y_gu = (bo_gu[flag2][1] + bo_feng[flag2][1]) / 2
            zhongxian_k_gu = -1 / xielv_k_gu
            zhongxian_k_gu = round(zhongxian_k_gu, 3)
            chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
            chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)

            flag2 += 1
            temp1 = []
            temp1.append(zhongxian_k_gu)
            temp1.append([mid_point_x_gu, mid_point_y_gu])
            gu_to_feng_point.append(temp1)

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

def return_angle(flag = False):
    # 读取图像
    # originalimg = cv2.imread('big.jpg',1)
    # image = originalimg[1580:2060,1898:2538]  # 中心彩色图像
    # img = cv2.imread('big.jpg',0)
    # img = img[1580:2060,1898:2538]     # 中心灰度图像
    image = cv2.imread('640+480.jpg',1)
    img = cv2.imread('640+480.jpg',0)
    image3 = image.copy()       # 中心彩色图像复制
    temp1 = get_strengthen_gray(img, precision=79, low_value=200)
    average = average_Grayval(img) / 2
    _, temp = cv2.threshold(temp1, average, 255, cv2.THRESH_BINARY)
    erosion = cv2.dilate(temp, kernel = numpy.ones((1, 3), numpy.uint8))   # 腐蚀 膨胀
    erosion = cv2.erode(erosion, kernel = numpy.ones((3, 3), numpy.uint8))
    skel = get_skeleton(erosion)
    edges = cv2.Canny(skel, 150, 550, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=numpy.pi / 180, threshold=100, minLineLength=200, maxLineGap=150)  # 概率霍夫变换
    inf_angle = []  # 角度
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2 - y1) > (img.shape[1] / 2):
                angle = math.degrees(math.atan(((x2 - x1) / (y2 - y1))))
                up_int = round(angle, 3)
                slip_angle = float(str(angle)[0:5])
                print(up_int, '\t', slip_angle)
                inf_angle.append(slip_angle)
    # print('Angle:',inf_angle)
    max_angle = max(inf_angle)
    min_angle = min(inf_angle)
    mid_angle = numpy.median(inf_angle)
    print(max_angle,min_angle,mid_angle)
    return max_angle,min_angle,mid_angle


# ********************************  计算间距  **********************************************#
def return_distance(flag = False):
    # lunkuo_img = originalimg[648:768,2020:2270] # 轮廓图像
    # lunkuo_img = BiLinear_interpolation(lunkuo_img,lunkuo_img.shape[0]*3,lunkuo_img.shape[1]*3)  # 轮廓图像放大
    lunkuo_img = cv2.imread('0831.jpg',1)
    outline_img1 = cv2.cvtColor(lunkuo_img, cv2.COLOR_BGR2GRAY)  # 转灰度图
    _, otsu = cv2.threshold(outline_img1, None, 255, cv2.THRESH_OTSU)  # OTSU阈值分割
    otsu = cv2.erode(otsu, kernel=numpy.ones((3, 3), numpy.uint8))  # 腐蚀去毛边
    canny = cv2.Canny(otsu, 100, 500, None, 3)  # 边缘检测

    for j in range(canny.shape[1]):
        for i in range(canny.shape[0]):
            if canny[i][j] == 0:
                continue
            else:
                set1.append([j, i])
    img2 = numpy.pad(canny, ((0, 0), (25, 25)), 'constant', constant_values=((0, 0), (0, 0)))
    un8 = numpy.zeros((img2.shape[0], img2.shape[1], 3), numpy.uint8) + 255
    for i in range(25):
        img2[set1[0][1]][i] = 255
        img2[set1[len(set1) - 1][1]][img2.shape[1] - i - 1] = 255
    for j in range(img2.shape[1]):
        for i in range(img2.shape[0]):
            if img2[i][j] == 0:
                continue
            else:
                un8[i][j] = [0, 0, 0]
                set2.append([j, i])
    for i in range(25, len(set2) - 25):
        max_y_label = max(set2[i - 25][1], set2[i - 24][1], set2[i - 23][1], set2[i - 22][1], set2[i - 21][1],
                          set2[i - 20][1], set2[i - 19][1], set2[i - 18][1], set2[i - 17][1], set2[i - 16][1],
                          set2[i - 15][1], set2[i - 14][1], set2[i - 13][1], set2[i - 12][1], set2[i - 11][1],
                          set2[i - 10][1], set2[i - 9][1], set2[i - 8][1], set2[i - 7][1], set2[i - 6][1],
                          set2[i - 5][1], set2[i - 4][1], set2[i - 3][1], set2[i - 2][1], set2[i - 1][1],
                          set2[i + 1][1], set2[i + 2][1], set2[i + 3][1], set2[i + 4][1], set2[i + 5][1],
                          set2[i + 6][1], set2[i + 7][1], set2[i + 8][1], set2[i + 9][1], set2[i + 10][1],
                          set2[i + 11][1], set2[i + 12][1], set2[i + 13][1], set2[i + 14][1], set2[i + 15][1],
                          set2[i + 16][1], set2[i + 17][1], set2[i + 18][1], set2[i + 19][1], set2[i + 20][1],
                          set2[i + 21][1], set2[i + 22][1], set2[i + 23][1], set2[i + 24][1], set2[i + 25][1])
        min_y_label = min(set2[i - 25][1], set2[i - 24][1], set2[i - 23][1], set2[i - 22][1], set2[i - 21][1],
                          set2[i - 20][1], set2[i - 19][1], set2[i - 18][1], set2[i - 17][1], set2[i - 16][1],
                          set2[i - 15][1], set2[i - 14][1], set2[i - 13][1], set2[i - 12][1], set2[i - 11][1],
                          set2[i - 10][1], set2[i - 9][1], set2[i - 8][1], set2[i - 7][1], set2[i - 6][1],
                          set2[i - 5][1], set2[i - 4][1], set2[i - 3][1], set2[i - 2][1], set2[i - 1][1],
                          set2[i + 1][1], set2[i + 2][1], set2[i + 3][1], set2[i + 4][1], set2[i + 5][1],
                          set2[i + 6][1], set2[i + 7][1], set2[i + 8][1], set2[i + 9][1], set2[i + 10][1],
                          set2[i + 11][1], set2[i + 12][1], set2[i + 13][1], set2[i + 14][1], set2[i + 15][1],
                          set2[i + 16][1], set2[i + 17][1], set2[i + 18][1], set2[i + 19][1], set2[i + 20][1],
                          set2[i + 21][1], set2[i + 22][1], set2[i + 23][1], set2[i + 24][1], set2[i + 25][1])
        if set2[i][1] >= max_y_label:
            lst1.append([set2[i][0], set2[i][1]])
        if set2[i][1] <= min_y_label:
            lst2.append([set2[i][0], set2[i][1]])
        else:
            continue
    lst3 = lst1.copy()  # 谷
    lst4 = lst2.copy()  # 峰
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

    if len(bofeng_mid_point) > len(bogu_mid_point):
        tempi, tempj = type_M(bofeng_mid_point,bogu_mid_point)
    elif len(bofeng_mid_point) < len(bogu_mid_point):
        tempi, tempj = type_W(bofeng_mid_point, bogu_mid_point)
    else:
        if bofeng_mid_point[0][0] > bogu_mid_point[0][0] and bofeng_mid_point[len(bofeng_mid_point)-1][0] > bogu_mid_point[len(bogu_mid_point)-1][0]:
            tempi, tempj = type_INV_N(bofeng_mid_point, bogu_mid_point)
        elif bofeng_mid_point[0][0] < bogu_mid_point[0][0] and bofeng_mid_point[len(bofeng_mid_point)-1][0] < bogu_mid_point[len(bogu_mid_point)-1][0]:
            tempi, tempj = type_N(bofeng_mid_point, bogu_mid_point)
        else:
            print("工艺设计问题，请重新编写代码段！！！")
    return tempi,tempj

if __name__ == '__main__':
    # return_angle(flag = False)
    tempi,tempj = return_distance(flag=True)
    print(tempj)