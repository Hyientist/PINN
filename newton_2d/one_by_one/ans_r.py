

def ans_r(x_, y_, u_, v_, cd_, dt, x_a=0.0, y_a=-9.8):
    if u_ > 0:
        u_dir = 1
    else:
        u_dir = -1
    if v_ > 0:
        v_dir = 1
    else:
        v_dir = -1
    x_ = x_ + u_ * dt + 0.5 * (x_a - u_ ** 2 * cd_ * u_dir) * dt ** 2
    y_ = y_ + v_ * dt + 0.5 * (y_a - v_ ** 2 * cd_ * v_dir) * dt ** 2
    u_ = u_ + x_a * dt - u_ ** 2 * cd_ * dt * u_dir
    v_ = v_ + y_a * dt - v_ ** 2 * cd_ * dt * v_dir

    return x_, y_, u_, v_
