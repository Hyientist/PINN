import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def set_random_seed(random_seed):
    try:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        print(f"random seed set as {random_seed}")
    except Exception as e:
        print(f"can't set random seed check following error statement: {e}")


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


# prepare data to learn and check the data
random_seed_ = 43
set_random_seed(random_seed_)
val_x_, val_y_, val_x_v_, val_y_v_ = random_initial_condition(5.0, 10.0)
answers = list()
answers.append([val_x_, val_y_, val_x_v_, val_y_v_, 0])
val_cd = 0.1
dt = 0.1
for i in range(1, 51):
    val_x_, val_y_, val_x_v_, val_y_v_ = ans_r(val_x_, val_y_, val_x_v_, val_y_v_, val_cd, dt)
    answers.append([val_x_, val_y_, val_x_v_, val_y_v_, dt*i])

answers_np = np.array(answers)
plt.plot(answers_np[:, 4], answers_np[:, 0], label="x")
plt.plot(answers_np[:, 4], answers_np[:, 1], label="y")
plt.plot(answers_np[:, 4], answers_np[:, 2], label="u")
plt.plot(answers_np[:, 4], answers_np[:, 3], label="v")
plt.legend()
plt.show()

# Torch Dataset Part


class DatasetNewton2D(Dataset):
    def __init__(self, input_chunk, output_chunk):
        self.data = input_chunk.clone().detach().requires_grad_(True)
        self.labels = output_chunk

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset_ts = torch.tensor(answers, dtype=torch.float)
dataset_noise_ratio = 0.03  # 1%
noise = torch.rand_like(dataset_ts)*torch.tensor([1, 1, 1, 1, 0]) * dataset_noise_ratio
dataset_noise = dataset_ts + noise
data_input = dataset_noise[:, 4:]
data_label = dataset_noise[:, :4]
dataset = DatasetNewton2D(data_input, data_label)
dataloader = DataLoader(dataset, shuffle=False, batch_size=data_input.shape[0]) # Full Batch

# Model part


class ModelNewton2D(torch.nn.Module):
    def __init__(self):
        super(ModelNewton2D, self).__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(1, 32))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(32, 16))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(16, 32))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(32, 5))
        self.net = torch.nn.Sequential(*self.layers)

    def forward(self, input_chunk):
        output = self.net(input_chunk)
        return output


model = ModelNewton2D()

# Loss and Optimizer Part


def ans_r_torch(x_, y_, u_, v_, cd_, dt_, x_a=0.0, y_a=-9.8):
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


def lbfgs_closure(dataloader_, optimizer_, model_, equation_, dt_):
    """
    Creates a closure for use with the LBFGS optimizer in PyTorch. This closure
    computes the loss for a batch of data, performs backpropagation to calculate
    the gradients, and returns the total loss.

    Parameters:
    dataloader_ (torch.utils.data.DataLoader): The data loader providing batches
                                               of input data and labels.
    optimizer_ (torch.optim.Optimizer): The optimizer used for updating the model
                                        parameters. Typically an instance of LBFGS.
    model_ (torch.nn.Module): The neural network model that predicts outputs
                              from the input data.
    equation_ (callable): A function that takes model outputs and other parameters
                          to compute additional constraints or physical equations
                          relevant to the problem.
    dt_ (torch.Tensor or float): The time step used in the physical equation
                                 or the model update.

    Returns:
    callable: A closure function that computes the total loss for the current
              batch of data, performs backpropagation, and returns the total loss.

    Details:
    - The closure function is designed to be used with the `LBFGS` optimizer in PyTorch,
      which requires a closure function to be passed to its `step` method.
    - The closure resets the gradients using `optimizer_.zero_grad()` at the start of
      each iteration to ensure that gradients are not accumulated across batches.
    - For each batch of data provided by `dataloader_`, the model predictions are
      obtained by passing `input_chunk` through the model. The first four columns
      of the model output (x, y, u, v) are extracted, and the observed loss is
      computed using the mean squared error between the model predictions and
      the true labels.
    - The mean value of the fifth column, `model_cd`, is extracted from the model
      output, representing a parameter related to drag or another physical property.
    - The physical consistency loss (`loss_eqn`) is computed by passing the model
      outputs (x, y, u, v), the drag coefficient, and the time step `dt_` to the
      `equation_` function, which returns expected results based on the physical
      model. The loss is then computed as the mean squared error between these
      results and the true labels.
    - The total loss is the sum of the observed loss (`loss_obs`) and the physical
      consistency loss (`loss_eqn`). This total loss is used to perform backpropagation.

    CAUTION:
    - The code contains sections that compute gradients with respect to the input
      (`grad_x`, `grad_y`) and additional loss terms (`loss_u`, `loss_v`).
      Uncommenting these lines will add gradient-based constraints to the total loss
      function.

      WARNING: Including these additional loss terms can significantly alter the
      learning dynamics of the model. In particular, the predicted value of `cd`
      (drag coefficient) may change, as these terms enforce stricter physical
      consistency constraints. It is essential to understand the impact of these
      additional losses on the overall training process. To understand this impact:

      1. **Run Training Without `loss_u` and `loss_v`**: Comment out the additional
         loss terms and train the model. Observe the predicted `cd` value.

      2. **Run Training With `loss_u` and `loss_v`**: Uncomment the additional loss
         terms, include them in the total loss, and train the model. Compare the
         predicted `cd` value with the previous step.

      3. **Compare Results**: Analyze how the inclusion of these gradient-based loss
         terms affects the `cd` value and overall model performance.

    Example Usage:
    ```
    optimizer = torch.optim.LBFGS(model_.parameters())
    closure = lbfgs_closure(dataloader_, optimizer_, model_, equation_, dt_)
    optimizer.step(closure)
    ```
    """

    def closure():
        optimizer_.zero_grad()  # Reset gradients to avoid accumulation
        total_loss = 0  # Initialize total loss
        lossfn = torch.nn.MSELoss()  # Define the loss function (Mean Squared Error)

        # Iterate over each batch of data from the dataloader
        for input_chunk, label_chunk in dataloader_:
            # Forward pass: compute the model output
            model_output_chunk = model_(input_chunk)
            # Extract the first four components (x, y, u, v) of the output
            model_output_xyuv = model_output_chunk[:, :4]
            # Compute the observed loss between the model output and the true labels
            loss_obs = lossfn(model_output_xyuv, label_chunk)

            # Extract individual components for further computation
            model_x = model_output_chunk[:, 0]
            model_y = model_output_chunk[:, 1]
            # model_u and model_v are the velocity components
            model_u = model_output_chunk[:, 2]
            model_v = model_output_chunk[:, 3]
            # model_cd is a drag coefficient or another physical parameter
            model_cd = torch.mean(model_output_chunk[:, 4])

            # Compute the equation loss based on physical consistency
            eqn_ans = equation_(model_x, model_y, model_u, model_v, model_cd, dt_)
            loss_eqn = lossfn(eqn_ans, label_chunk)

            # Combine observed loss and equation loss into total loss
            total_loss = loss_eqn + loss_obs

            # ------------------------- CAUTION -------------------------
            # Uncomment the following lines to include additional gradient-based losses
            # WARNING: Including these additional losses will change the predicted `cd` value.
            # grad_x = torch.autograd.grad(
            #     model_x, input_chunk,
            #     grad_outputs=torch.ones_like(model_x),
            #     retain_graph=True,
            #     create_graph=True
            # )[0].squeeze()
            # grad_y = torch.autograd.grad(
            #     model_y, input_chunk,
            #     grad_outputs=torch.ones_like(model_y),
            #     retain_graph=True,
            #     create_graph=True
            # )[0].squeeze()
            # loss_u = lossfn(grad_x, model_u)
            # loss_v = lossfn(grad_y, model_v)
            # total_loss = loss_eqn + loss_obs + loss_u + loss_v
            # -----------------------------------------------------------

            # Print debugging information for the current batch
            print("----------------------------")
            print("Value of Cd:", model_cd.item())
            print("loss of obs:", loss_obs.item())
            print("loss of eqn:", loss_eqn.item())
            # print("loss of u:", loss_u.item())
            # print("loss of v:", loss_v.item())
            print("")

        total_loss.backward()  # Compute gradients via backpropagation
        return total_loss  # Return the total loss for the optimizer

    return closure  # Return the closure function for the optimizer


# Learning Part
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=5000)
optimizer.step(lbfgs_closure(dataloader, optimizer, model, ans_r_torch, dt))

model_output = model(data_input)
data_label_np = data_input.numpy()
model_output_np = model_output.detach().numpy()
plt.plot(data_label_np, model_output_np[:, 0], label="model_x")
plt.plot(data_label_np, model_output_np[:, 1], label="model_y")
plt.plot(data_label_np, model_output_np[:, 2], label="model_u")
plt.plot(data_label_np, model_output_np[:, 3], label="model_v")

dataset_with_noise = dataset_noise.numpy()
plt.plot(dataset_with_noise[:, 4], dataset_with_noise[:, 0], label="x")
plt.plot(dataset_with_noise[:, 4], dataset_with_noise[:, 1], label="y")
plt.plot(dataset_with_noise[:, 4], dataset_with_noise[:, 2], label="u")
plt.plot(dataset_with_noise[:, 4], dataset_with_noise[:, 3], label="v")

plt.legend()
plt.show()
