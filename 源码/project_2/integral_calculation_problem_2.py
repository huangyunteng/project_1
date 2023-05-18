"""
## ===== 问题分析 =====
## 1.图最终计算的是1元1次方程；
## 2.我不需要理论推导，只需要打点连线就可以，就是得到1个点的值，其实就相当于可以得到所有点的值；
## 3.两个难点：（1）python求实根；（2）python求积分；

## （1）方程改写
f(x) = 40*h - 5*x**8 + 4*x**5 + 10*x**2

## 参考：https://blog.csdn.net/charlie_C9299/article/details/122196162
python 求实根的3种方法（带马）：
## ======================
import sympy as sy
a=math.pi
#-----------------方法----------------------
x = sp.Symbol('x') # 定义符号变量
h = -8/40
f = 40*h - 5*x**8 + 4*x**5 + 10*x**2
x = sp.solve(f) # 调用solve函数求解方程
print("x1=",x)

## (2)求积分
"""
import numpy as np
import pandas as pd
from scipy import integrate as inte_sci
import sympy as sy
from sympy import integrate as inte_sym
from sympy import Float


# 求实根
def get_roots(h):
    x = sy.Symbol('x')  # 定义符号变量
    f = h + 1/8 * x**8 - 1/4 * x**5 + 1/8 * x**2
    return sy.solve(f)  # 调用solve函数求解方程


# 求积分
def f_u(x, h):
    y = x * (h + 1/8 * x ** 8 - 1/4 * x**5 + 1/8 * x**2) ** (1 / 2)
    return y

if __name__ == "__main__":

    cuts = 2000
    _dict = {}
    for h in np.arange(-0.0279035341 + (0 + 0.0279035341) / cuts, 0, (0 + 0.0279035341) / cuts):
        print("h:{}".format(h))
        roots_lst = get_roots(h)
        real_roots = [_root for _root in roots_lst if type(_root)==Float]
        real_roots.sort()
        if len(real_roots) == 2:
            root_dn = real_roots[0]
            root_up = real_roots[1]
        elif len(real_roots) == 4:
            root_dn = real_roots[1]
            root_up = real_roots[2]

        def f_u(x):
            y1 = x * (h + 1 / 8 * x ** 8 - 1 / 4 * x ** 5 + 1 / 8 * x ** 2) ** (1 / 2)
            return y1

        def f_d(x):
            y2 = (h + 1 / 8 * x ** 8 - 1 / 4 * x ** 5 + 1 / 8 * x ** 2) ** (1 / 2)
            return y2

        y_u, err = inte_sci.quad(f_u, root_dn, root_up)
        y_d, err = inte_sci.quad(f_d, root_dn, root_up)

        if y_d == 0:
            P = 0
        else:
            P = y_u / y_d
        _dict[h] = P

        rs = pd.DataFrame(columns=["h", "P"])
        rs["h"] = _dict.keys()
        rs["P"] = _dict.values()
        rs.to_csv("rs_pro2.csv", index=False)

