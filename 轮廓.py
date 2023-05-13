import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

def average_Grayval(img):  # 平均灰度值
    sum = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum += img[i, j]
    average = sum / img.shape[0] / img.shape[1]
    return average

def partial_derivative(new_gray_img):  # 自定义canny
    new_gray_img = np.pad(new_gray_img, ((0, 1), (0, 1)), constant_values=0)  # 填充
    h, w = new_gray_img.shape
    dx_gray = np.zeros([h - 1, w - 1])  # 用来存储x方向偏导
    dy_gray = np.zeros([h - 1, w - 1])  # 用来存储y方向偏导
    df_gray = np.zeros([h - 1, w - 1])  # 用来存储梯度强度
    for i in range(h - 1):
        for j in range(w - 1):
            dx_gray[i, j] = new_gray_img[i, j + 1] - new_gray_img[i, j]
            dy_gray[i, j] = new_gray_img[i + 1, j] - new_gray_img[i, j]
            df_gray[i, j] = np.sqrt(np.square(dx_gray[i, j]) + np.square(dy_gray[i, j]))

    df_gray = np.pad(df_gray, ((1, 1), (1, 1)), constant_values=0)  # 填充
    h, w = df_gray.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if df_gray[i, j] != 0:
                gx = math.fabs(dx_gray[i - 1, j - 1])
                gy = math.fabs(dy_gray[i - 1, j - 1])
                if gx > gy:
                    weight = gy / gx
                    grad1 = df_gray[i + 1, j]
                    grad2 = df_gray[i - 1, j]
                    if gx * gy > 0:
                        grad3 = df_gray[i + 1, j + 1]
                        grad4 = df_gray[i - 1, j - 1]
                    else:
                        grad3 = df_gray[i + 1, j - 1]
                        grad4 = df_gray[i - 1, j + 1]
                else:
                    weight = gx / gy
                    grad1 = df_gray[i, j + 1]
                    grad2 = df_gray[i, j - 1]
                    if gx * gy > 0:
                        grad3 = df_gray[i + 1, j + 1]
                        grad4 = df_gray[i - 1, j - 1]
                    else:
                        grad3 = df_gray[i + 1, j - 1]
                        grad4 = df_gray[i - 1, j + 1]
                t1 = weight * grad1 + (1 - weight) * grad3
                t2 = weight * grad2 + (1 - weight) * grad4
                if df_gray[i, j] > t1 and df_gray[i, j] > t2:
                    df_gray[i, j] = df_gray[i, j]
                else:
                    df_gray[i, j] = 0
    return df_gray

image = cv.imread('li0.jpg', 1)
img1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)    # 转灰度图
# cv.imshow('img',img1)
# for i in range(img1.shape[0]):
#     for j in range(img1.shape[1]):
#         img1[i,j] = img1[i,j] - 10
#         pass
#     pass
# cv.imshow('img',img1)

# _, th1 = cv.threshold(img1, 100, 255, cv.THRESH_BINARY)
# cv.imshow('th1', th1)

_, otsu = cv.threshold(img1, None,255, cv.THRESH_OTSU)      # OTSU阈值分割
cv.imshow('otsu', otsu)

otsu = cv.erode(otsu,kernel=np.ones((3, 3), np.uint8))      # 腐蚀去毛边
cv.imshow('otsu1', otsu)

# dst = cv.addWeighted(img1,1,otsu,0.1,1)  # 权重相加看处理效果
# cv.imshow('dst', dst)

# temp1 = partial_derivative(otsu)      # 自写边缘检测
# cv.imshow('temp1',temp1)

canny = cv.Canny(otsu,100,500,None,3)   # 边缘检测
cv.imshow('canny',canny)
un8 = np.zeros((canny.shape[0], canny.shape[1], 3), np.uint8) + 200
# _,th2 = cv.threshold(canny,100,255,cv.THRESH_BINARY_INV)  # 颜色反转
# cv.imshow('th2',th2)

# txt = open("th2.txt",mode='w',encoding = "utf-8")
# for j in range(canny.shape[1]):
#     for i in range(canny.shape[0]):
#         if canny[i][j] <= 122:
#             canny[i][j] = 0
#         else:
#             canny[i][j] = 255
#         txt.write(str(canny[i][j]))
#         txt.write('  ')
# txt.close()

set = []
for j in range(canny.shape[1]):
    for i in range(canny.shape[0]):
        if canny[i][j] == 0:
            continue
        else:
            set.append([j,i])

print(set)
for i in set:
    un8[i[1],i[0]] = 0
    # un8[i[1]+1, i[0]] = 0
    # un8[i[1]-1, i[0]] = 0
# plt.imshow(canny, cmap=plt.cm.gray)
# plt.show()
cv.imshow('kuot',un8)
cv.waitKey(0)






# import cv2
# import numpy
# import math
# def BiLinear_interpolation(img,dstH,dstW):  # 双线性插值
#     scrH,scrW,_=img.shape
#     img=numpy.pad(img,((0,1),(0,1),(0,0)),'constant')  # 边缘填充
#     retimg=numpy.zeros((dstH,dstW,3),dtype=numpy.uint8)   # 创建背景板
#     for i in range(dstH):
#         for j in range(dstW):
#             scrx=(i+1)*(scrH/dstH)-1
#             scry=(j+1)*(scrW/dstW)-1
#             x=math.floor(scrx)      # math.floor 取整
#             y=math.floor(scry)
#             u=scrx-x
#             v=scry-y
#             retimg[i,j]=(1-u)*(1-v)*img[x,y] + u*(1-v)*img[x+1,y] + (1-u)*v*img[x,y+1] + u*v*img[x+1,y+1]
#     return retimg
#
# img = cv2.imread('big.jpg',1)
# lunkuo = img[648:768,2020:2272]
# lunkuo = BiLinear_interpolation(lunkuo,lunkuo.shape[0]*3,lunkuo.shape[1]*3)
# zhongxin = img[1580:2060,1898:2538]
# cv2.imshow('lun',lunkuo)
# cv2.imshow('xin',zhongxin)
# print(lunkuo.shape)
# cv2.waitKey(0)
# cv2.destroyAllWindows()