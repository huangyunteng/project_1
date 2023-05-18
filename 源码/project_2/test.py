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
from scipy import integrate
import sympy as sy
from sympy import *

# 求实根
def get_roots(h):
    x = sy.Symbol('x')  # 定义符号变量
    f = 40 * h - 5 * x ** 8 + 4 * x ** 5 + 10 * x ** 2
    return sy.solve(f)  # 调用solve函数求解方程

# 求积分
def f_u(x, h=-8/40):
    y = x * (h-1/8*x**8+1/10*x**5+1/4*x**2)**(1/2)
    return y

def f_d(x, h=-8/40):
    y = (h-1/8*x**8+1/10*x**5+1/4*x**2)**(1/2)
    return y

if __name__ == "__main__":

    _lst = []
    _dict = {}
    for h in np.arange(-0.19, 0, (0+9/40)/1000): # -0.19997999999999042
        print("h:{}".format(h))
        roots_lst = get_roots(h)
        real_roots = [_root for _root in roots_lst if (len(str(_root).split("-")) <= 1) and (len(str(_root).split("+")) <= 1)]
        real_roots.sort()
        if len(real_roots) == 2:
            root_dn = real_roots[0]
            root_up = real_roots[1]
        elif len(real_roots) == 4:
            root_dn = real_roots[1]
            root_up = real_roots[2]

        if True:

            for x in np.arange(root_dn, root_up, (root_up - root_dn)/50):
                y = (h-1/8*x**8+1/10*x**5+1/4*x**2)**(1/2)
                print(y)
                _lst.append(y)
            import matplotlib.pyplot as plt
            # plt.plot(_lst)
            # plt.show()
            y_u, err = integrate.quad(f_u, root_dn, root_up)


            # y_u, err = integrate.quad(f_u, root_dn, root_up)
            # y_d, err = integrate.quad(f_d, root_dn, root_up)

            x = symbols('x')
            y_u = integrate(x * (h - (1/8) * x**8 + (1/10) * x**5 + (1/4) * x**2)**(1/2), (x, root_dn, root_up))
            # print("y_u:{}".format(y_u))

            # integrate.fixed_quad(f_u, root_dn, root_up)
            # integrate.fquadrature(f_u, root_dn, root_up)





