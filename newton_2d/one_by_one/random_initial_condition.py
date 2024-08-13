import random
import math


def random_initial_condition(val_r, val_v):
    val_x = random.random()
    val_y = random.random()
    val_xy_div = 1 / math.sqrt((val_x**2 + val_y**2))
    val_x = val_x*val_xy_div*val_r
    val_y = val_y*val_xy_div*val_r

    val_x_v = random.random()
    val_y_v = random.random()
    val_xy_div = 1 / math.sqrt((val_x_v ** 2 + val_y_v ** 2))
    val_x_v = val_x_v * val_xy_div * val_v
    val_y_v = val_y_v * val_xy_div * val_v

    print(val_x, val_y, val_x_v, val_y_v)
    return val_x, val_y, val_x_v, val_y_v
