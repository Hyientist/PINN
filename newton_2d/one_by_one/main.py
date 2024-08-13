import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from set_random_seed import set_random_seed
from random_initial_condition import random_initial_condition
from ans_r import ans_r
from DatasetNewton2D import DatasetNewton2D
from ModelNewton2D import ModelNewton2D
from ans_r_torch import ans_r_torch
from lbfgs_closure import lbfgs_closure


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

dataset_ts = torch.tensor(answers, dtype=torch.float)
dataset_noise_ratio = 0.03  # 1%
noise = torch.rand_like(dataset_ts)*torch.tensor([1, 1, 1, 1, 0]) * dataset_noise_ratio
dataset_noise = dataset_ts + noise
data_input = dataset_noise[:, 4:]
data_label = dataset_noise[:, :4]
dataset = DatasetNewton2D(data_input, data_label)
dataloader = DataLoader(dataset, shuffle=False, batch_size=data_input.shape[0]) # Full Batch

# Model part
model = ModelNewton2D()

# Learning Part
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=5000)
model_save_path = './model_newton2d.pth'
optimizer_save_path = './optimizer_newton2d.pth'
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
