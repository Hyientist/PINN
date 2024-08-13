import torch


def ans_r_torch(x_, y_, u_, v_, cd_, dt_, x_a=0.0, y_a=-9.8):
    """
    Updates the position and velocity of an object under the influence of
    acceleration and drag using tensor operations, ensuring compatibility with
    PyTorch for efficient computation, especially when dealing with batches of data.

    Parameters:
    x_ (torch.Tensor): The initial x-coordinate(s) of the object(s).
    y_ (torch.Tensor): The initial y-coordinate(s) of the object(s).
    u_ (torch.Tensor): The initial x-axis velocity component(s) of the object(s).
    v_ (torch.Tensor): The initial y-axis velocity component(s) of the object(s).
    cd_ (torch.Tensor): The drag coefficient(s) for the object(s).
    dt_ (torch.Tensor or float): The time step(s) over which to compute the updates.
    x_a (float, optional): The constant acceleration in the x direction, default is 0.0.
    y_a (float, optional): The constant acceleration in the y direction, default is -9.8
                           (simulating gravity).

    Returns:
    torch.Tensor: A tensor with the updated x, y, u, and v values, stacked and transposed
                  such that each row corresponds to a single object with its updated
                  position and velocity in the format [x, y, u, v].

    Details:
    - The function first computes `u_dir` and `v_dir`, which represent the direction
      of the x and y velocity components, respectively. This is determined using
      the sign of the velocity values (`u_` and `v_`).
    - The positions `x_` and `y_` are then updated using the current velocity,
      acceleration, drag, and the time step `dt_`. The equation accounts for the
      influence of acceleration and drag on the object's movement over the time step.
    - Similarly, the velocities `u_` and `v_` are updated, factoring in the effects
      of acceleration and drag. The drag force is assumed to be proportional to
      the square of the velocity and acts in the opposite direction of motion.
    - Finally, the updated position and velocity values are stacked into a single tensor
      and transposed so that each row contains the values [x, y, u, v] for a corresponding
      object.
    """

    # Calculate the direction of the velocity components using vectorized operations
    u_dir = torch.sign(u_)  # Returns 1 if u_ > 0, -1 if u_ < 0, and 0 if u_ == 0
    v_dir = torch.sign(v_)  # Returns 1 if v_ > 0, -1 if v_ < 0, and 0 if v_ == 0

    # Update the position using vectorized operations
    x_ = x_ + u_ * dt_ + 0.5 * (x_a - u_ ** 2 * cd_ * u_dir) * dt_ ** 2
    y_ = y_ + v_ * dt_ + 0.5 * (y_a - v_ ** 2 * cd_ * v_dir) * dt_ ** 2

    # Update the velocity using vectorized operations
    u_ = u_ + x_a * dt_ - u_ ** 2 * cd_ * dt_ * u_dir
    v_ = v_ + y_a * dt_ - v_ ** 2 * cd_ * dt_ * v_dir

    # Stack the updated position and velocity components and transpose the result
    return torch.vstack([x_, y_, u_, v_]).T
