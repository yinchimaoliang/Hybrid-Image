import math
import numpy as np

def get_cv(r, sigma):
    return 1 / (2 * math.pi * sigma ** 2) * math.exp((-r**2) / (2 * sigma ** 2))


def get_window():
    # 模糊半径为 2, sigma 为 1.5
    radius, sigma = 2, 1.5
    window = np.zeros((radius * 2 + 1, radius * 2 + 1))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            r = (i ** 2 + j ** 2) ** 0.5
            window[i + radius][j + radius] = get_cv(r, sigma)
    return window / np.sum(window)


if __name__ == '__main__':
    print(get_window())