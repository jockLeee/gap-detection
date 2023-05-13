from numpy import *
from scipy import optimize, odr
import functools
from matplotlib import pyplot as plt, cm, colors

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# 方法一 代数逼近法
method_1 = '代数逼近法    '
# 坐标
x = r_[14,   15,  15,  16,  16,  17,  17,  18,  19,  20,  20,  21,  22,  23,  24,  25,  26,  26,  27,  28,  29,  30,  31,  32]
y = r_[-261, -260, -261, -259, -260, -258, -259, -258, -257, -256, -257, -255, -254, -254, -253, -253, -252, -253, -252, -251, -251, -250, -250, -250]
# basename = 'arc'
print(type(x))
# 质心坐标
x_m = mean(x)
y_m = mean(y)

print(x_m, y_m)

# 相对坐标
u = x - x_m
v = y - y_m

Suv = sum(u * v)
Suu = sum(u ** 2)
Svv = sum(v ** 2)
Suuv = sum(u ** 2 * v)
Suvv = sum(u * v ** 2)
Suuu = sum(u ** 3)
Svvv = sum(v ** 3)

# 求线性系统
A = array([[Suu, Suv], [Suv, Svv]])
B = array([Suuu + Suvv, Svvv + Suuv]) / 2.0
uc, vc = linalg.solve(A, B)

xc_1 = x_m + uc
yc_1 = y_m + vc

# 半径残余函数
Ri_1 = sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
R_1 = mean(Ri_1)

# residu_1 = sum((Ri_1 - R_1) ** 2)
# ncalls_1 = 0
# residu2_2 = 0

fmt = '%-22s  %10.5f  %10.5f  %10.5f  '

print('-' * (22 + 4 * (10 + 1)))
print(fmt % (method_1, xc_1, yc_1, R_1))

def plot_all(residu=False):
    plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
    plt.axis('equal')
    theta_fit = linspace(-pi, pi, 180)

    x_fit1 = xc_1 + R_1 * cos(theta_fit)
    y_fit1 = yc_1 + R_1 * sin(theta_fit)
    plt.plot(x_fit1, y_fit1, 'k--', label=method_1, lw=2)
    plt.plot([xc_1], [yc_1], 'gD', mec='r', mew=1)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
    plt.legend(loc='best', labelspacing=0.1)

plot_all(residu=True)
plt.show()







