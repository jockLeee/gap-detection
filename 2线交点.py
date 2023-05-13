def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c


def get_line_cross_point(line1, line2):
    # x1y1x2y2
    a0, b0, c0 = calc_abc_from_line_2d(*line1)

    a1, b1, c1 = calc_abc_from_line_2d(*line2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    # print(x, y)
    return x, y


def get_line_crosspoint(x1, y1, x2, y2, x3, y3, x4, y4):
    # a0 = y1 - y2
    # b0 = x2 - x1
    # c0 = x1*y2-x2*y1
    # a1 = y3 - y4
    # b1 = x4 - x3
    # c1 = x3*y4 - x4*y3
    x = ((x2 - x1) * (x3 * y4 - x4 * y3) - (x4 - x3) * (x1 * y2 - x2 * y1)) / ((y1 - y2) * (x4 - x3) - (y3 - y4) * (x2 - x1))
    y = ((y4 - y3) * (x1 * y2 - x2 * y1) - (y2 - y1) * (x3 * y4 - x4 * y3)) / ((y1 - y2) * (x4 - x3) - (y3 - y4) * (x2 - x1))
    return x, y


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return (px, py)


if __name__ == '__main__':
    # x1y1x2y2
    line1 = [0, 3, 5, -2]
    line2 = [-4, -7, 10, 7]
    cross_pt = get_line_cross_point(line1, line2)
    print(cross_pt)

    temp = findIntersection(0, 3, 5, -2, -4, -7, 10, 7)
    print(temp)
    temp1 = get_line_crosspoint(0, 3, 5, -2, -4, -7, 10, 7)
    print(temp1)
