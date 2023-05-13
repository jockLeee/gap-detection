from numpy import *
from scipy import optimize, odr
import functools
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# 方法一 代数逼近法
method_1 = '代数逼近法    '
# 坐标
x = r_[14,   15,  15,  16,  16,  17,  17,  18,  19,  20,  20,  21,  22,  23,  24,  25,  26,  26,  27,  28,  29,  30,  31,  32]
y = r_[-261, -260, -261, -259, -260, -258, -259, -258, -257, -256, -257, -255, -254, -254, -253, -253, -252, -253, -252, -251, -251, -250, -250, -250]

# 质心坐标
x_m = mean(x)
y_m = mean(y)

print(x_m, y_m)

method_3 = "正交距离回归法"
def calc_R(xc, yc):
    """ 计算s数据点与圆心(xc, yc)的距离 """
    return sqrt((x - xc) ** 2 + (y - yc) ** 2)

def f_3(beta, x):
    """ 圆的隐式定义 """
    return (x[0] - beta[0]) ** 2 + (x[1] - beta[1]) ** 2 - beta[2] ** 2


"""参数初始化"""
R_m = calc_R(x_m, y_m).mean()
beta0 = [x_m, y_m, R_m]

lsc_data = odr.Data(row_stack([x, y]), y=1)
lsc_model = odr.Model(f_3, implicit=True)
lsc_odr = odr.ODR(lsc_data, lsc_model, beta0)
lsc_out = lsc_odr.run()

xc_3, yc_3, R_3 = lsc_out.beta

# Ri_3 = calc_R(xc_3, yc_3)
# residu_3 = sum((Ri_3 - R_3) ** 2)


fmt = '%-22s  %10.5f  %10.5f  %10.5f  '
print('-' * (22 + 4 * (10 + 1)))
print(fmt % (method_3, xc_3, yc_3, R_3))

def plot_all(residu=False):
    plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
    plt.axis('equal')
    theta_fit = linspace(-pi, pi, 180)

    x_fit3 = xc_3 + R_3 * cos(theta_fit)
    y_fit3 = yc_3 + R_3 * sin(theta_fit)
    plt.plot(x_fit3, y_fit3, 'r2-', label=method_3, lw=2)
    plt.plot([xc_3], [yc_3], 'gD', mec='r', mew=1)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
    plt.legend(loc='best', labelspacing=0.1)

plot_all(residu=True)
plt.show()