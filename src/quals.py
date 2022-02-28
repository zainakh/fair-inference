import numpy as np
import matplotlib.pyplot as plt
import math


def gen_gamma(x, a, d, p):
    num = (p/(a**d)) * (x**(d-1)) * np.exp(-((x/a)**p))
    denom = math.gamma(d/p)
    return num/denom


xs = np.linspace(0, 10, num=1000)
y = [gen_gamma(x, 1, 1, 1) for x in xs]

plt.plot(y, xs)