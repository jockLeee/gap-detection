import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Times New Roman"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
plt.figure(figsize=(30, 15), dpi=50)
testnum = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
machine = [0.433, 0.158, 0.16, 0.076, 0.091, 0.092, 0.224, 0.192, 0.223, 0.145, 0.188]
worker = [0.5, 0.15, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.23, 0.15, 0.2]
assists = []
tempnum = [0.12,0.05,0.13,0.22,0.21,0.31,0.11,0.14,0.12,0.16,0.09]
for i in range(len(testnum)):
    assists.append(abs(machine[i]-worker[i]))
# assists = [16, 7, 8, 10, 10, 7, 9, 5, 9, 7, 12, 4, 11, 8, 10, 9, 9, 8, 8, 7, 10]
plt.plot(testnum, assists, color='red', marker='o', linestyle='-.', linewidth=3)
# plt.plot(testnum, tempnum, color='red', marker='o', linestyle='-.', linewidth=3) # 测试
# plt.plot(testnum, worker, c='green', linestyle='--')
# plt.plot(testnum, assists, c='blue', linestyle='-.', label="助攻")
plt.scatter(testnum, assists, s =300 ,c='b')
# plt.scatter(testnum, machine, c='red') # 散点图点大小
# plt.scatter(testnum, worker, c='green')   # 散点图点大小

plt.text(10, 0.06, '哈哈哈', fontsize=28)
# plt.legend(loc='best')

plt.xticks(fontsize=32)  # x轴刻度字体大小
plt.yticks(fontsize=32)  # y轴刻度字体大小
plt.xticks(range(0, 12, 1))
plt.yticks(np.arange(-0.1, 0.2, 0.05))
# plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Winding layer number", fontdict={'size': 32})
plt.ylabel("Absolute value of difference", fontdict={'size': 32})
plt.title("Absolute value of 1.5mm conductor phase difference", fontdict={'size': 32})

plt.xlim(0, 13)  # X轴范围
plt.ylim(-0.1, 0.15)  # 显示y轴范围

plt.show()


# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('1.jpg',1)
# edges = cv.Canny(img,100,200)
# plt.subplot(121)
# plt.imshow(img,cmap = 'gray')
# plt.title('Original Image')
# plt.xticks([])
# plt.yticks([])
# plt.subplot(122)
# plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image')
# plt.xticks([])
# plt.yticks([])
# plt.show()