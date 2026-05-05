# 定義Point2D(二維點), Line2D(二維線段)兩個類別，是simple_playground運作的數學基礎
import math as m


class Point2D:
    ...


class Line2D:
    ...


# 點座標和向量運算
# (負責處理所有與座標(x, y)相關的運算，包含向量加減、長度、旋轉、距離計算和矩形區域檢查)
class Point2D():
    # 初始化點的(x, y)座標 -> 車輛中心、軌道節點
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 計算該點作為向量的長度 -> 計算該點到原點的距離
    @property
    def length(self):  # length when seem as a vector
        return m.sqrt(self.x**2 + self.y**2)

    def __str__(self) -> str:
        return f'{self.x} {self.y}'

    # 重載加減乘除的運算子 -> 進行向量或點的運算
    def __sub__(self, point):
        difx = self.x - point.x
        dify = self.y - point.y
        return Point2D(difx, dify)

    def __add__(self, point):
        sumx = self.x + point.x
        sumy = self.y + point.y
        return Point2D(sumx, sumy)

    def __mul__(self, num: float):
        return Point2D(self.x*num, self.y*num)

    def __div__(self, num: float):
        return Point2D(self.x/num, self.y/num)

    # 計算當前點到另一個點的歐幾里得距離 -> 在playground中計算感測器交點到車輛的距離
    def distToPoint2D(self, p2: Point2D):
        diff = self - p2
        return diff.length

    # 計算當前點到線段所在值線的垂直距離 -> 用於做碰撞檢測
    def distToLine2D(self, line: Line2D):
        p1 = line.p1
        lineTop1 = Line2D(self, p1)
        angle = lineTop1.angleToLine(line)
        return m.sin(angle/180*m.pi) * lineTop1.length

    # 繞原點旋轉點座標 -> 在Car.getPosition()中用於計算感測器位置
    def rorate(self, angle) -> Point2D:
        '''
        |cos a -sin a|       |x|
        |sin a  cos a| = RM, |y| = P
        |newx|
        |newy| = RM @ P

        new_x = x*cos(a) - y*sin(a)
        new_y = x*sin(a) + y*cos(a)
        '''
        rad = angle/180*m.pi
        new_x = self.x*m.cos(rad) - self.y*m.sin(rad)
        new_y = self.x*m.sin(rad) - self.y*m.cos(rad)
        return Point2D(new_x, new_y)

    # 檢查當前點是否在由p1(左上/右下)和p2(右上/左下)定義的矩形區域內
    # -> 用於檢查車輛中心是否抵達終點區域
    def isInRect(self, p1, p2):
        if p1.x > p2.x:
            rx = p1.x
            lx = p2.x
        else:
            rx = p2.x
            lx = p1.x

        if p1.y > p2.y:
            uy = p1.y
            dy = p2.y
        else:
            uy = p2.y
            dy = p1.y

        return lx <= self.x <= rx and dy <= self.y <= uy


# 線段與幾何交點
# (負責處理所有與線段(即跑道牆壁)相關的運算)
class Line2D():
    # 初始化線段的兩端點p1,p2 -> 跑道邊界和感測器射線
    def __init__(self, *arg):
        if len(arg) == 2:
            self.p1 = arg[0]
            self.p2 = arg[1]
        else:
            self.p1 = Point2D(arg[0], arg[1])
            self.p2 = Point2D(arg[2], arg[3])

    # 計算線段p1到p2的長度 -> 碰撞和距離計算
    @property
    def length(self):
        return (self.p1-self.p2).length

    def __str__(self) -> str:
        return f'{self.p1} {self.p2}'

    # 計算兩個線段作為向量之間的夾角 -> 在PointD.distToLine2D中被用來計算垂直距離
    def angleToLine(self, line: Line2D):
        p1, p2 = line.p1, line.p2
        p3, p4 = self.p1, self.p2

        v1 = p1-p2
        len_line = v1.length

        v2 = p3-p4
        len_line2 = v2.length
        angle_diff = m.acos((v1.x * v2.x + v1.y * v2.y) /
                            (len_line * len_line2 + 1e-10))

        return angle_diff/m.pi*180

    # 核心的幾何運算 -> 感測器讀數和碰撞檢測的關鍵
    # 檢查兩條線段(由p1 ~ p4定義)是否相交，當0 <= t <= 1且0 <= u <= 1時，
    # 兩線段才相交。其中輸出的t, u參數，線段P1P2上的點P可表示為P1 + t*(P2 - P1)，
    # 當0 <= t <= 1時，交點在P1P2線段上；當0 <= u <= 1時，交點在P3P4線段上
    def lineOverlap(self, line2: Line2D):
        '''
        input:
            line2: Line2D
                Check if this line is overlapped with line2
        output:
            isOverlapped: bool
                if two line are overlapped, return True

            t: float
                [x1 + t(x2-x1), y1 + t(y2-y1)]
            u: float
                [x3 + u(x4-x3), y3 + u(y4-y3)]
        '''

        p1, p2 = self.p1, self.p2
        p3, p4 = line2.p1, line2.p2

        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        x4, y4 = p4.x, p4.y

        # point intersect = [x1 + t(x2-x1), y1 + t(y2-y1)] = [x3 + u(x4-x3), y3 + u(y4-y3)]
        x13 = x1 - x3
        x21 = x2 - x1
        x43 = x4 - x3
        y21 = y2 - y1
        x34 = x3 - x4
        y34 = y3 - y4
        y31 = y3 - y1

        try:
            t = (x13*y34 + y31*x34)/(-x21*y34+y21*x34)
            u = (x13*y21 + y31*x21)/(x43*y21+y34*x21)
        except ZeroDivisionError as e:
            if (x13*y34 + y31*x34) == 0 and (x13*y21 + y31*x21) == 0:
                return True, None, None  # two lines are parallel and overlapped
            else:
                # two lines are parallel but no overlapped, or they intersect at the point out of range
                return False, None, None

        if 0 <= t <= 1 and 0 <= u <= 1:
            return True, t, u
        else:
            return False, t, u
