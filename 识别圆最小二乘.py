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

x_m = mean(x)
y_m = mean(y)

method_2 = "最小二乘法    "


# 修饰器：用于输出反馈
def countcalls(fn):
    "decorator function count function calls "

    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls += 1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped


def calc_R(xc, yc):
    """ 计算s数据点与圆心(xc, yc)的距离 """
    return sqrt((x - xc) ** 2 + (y - yc) ** 2)


@countcalls
def f_2(c):
    """ 计算半径残余"""
    Ri = calc_R(*c)
    return Ri - Ri.mean()


# 圆心估计
center_estimate = x_m, y_m
center_2, _ = optimize.leastsq(f_2, center_estimate)

xc_2, yc_2 = center_2
Ri_2 = calc_R(xc_2, yc_2)
# 拟合圆的半径
R_2 = Ri_2.mean()
residu_2 = sum((Ri_2 - R_2) ** 2)
residu2_2 = sum((Ri_2 ** 2 - R_2 ** 2) ** 2)
ncalls_2 = f_2.ncalls

fmt = '%-22s  %10.5f  %10.5f  %10.5f  '
print('-' * (22 + 4 * (10 + 1)))
print(fmt % (method_2, xc_2, yc_2, R_2))


def plot_all(residu=False):
    plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
    plt.axis('equal')
    theta_fit = linspace(-pi, pi, 180)

    x_fit2 = xc_2 + R_2 * cos(theta_fit)
    y_fit2 = yc_2 + R_2 * sin(theta_fit)
    plt.plot(x_fit2, y_fit2, 'bo-', label=method_2, lw=2)
    plt.plot([xc_2], [yc_2], 'gD', mec='r', mew=1)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
    plt.legend(loc='best', labelspacing=0.1)

plot_all(residu=True)

plt.show()













