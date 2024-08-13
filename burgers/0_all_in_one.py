import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DataBurgers:
    def __init__(self):
        # Hardcoded Data
        x = np.linspace(-1, 1, 100)
        t = np.linspace(0, 1, 100)
        xx, tt = np.meshgrid(x, t)
        self.xx = xx.flatten()
        self.tt = tt.flatten()
        self.yy = np.full_like(self.xx, np.nan)
        self.yy[0:100] = self.__u_input(self.xx[0:100])

        boundary_indices = np.where((self.xx == -1) | (self.xx == 1))
        self.yy[boundary_indices] = 0

        input_chunk = np.vstack([self.xx, self.tt])
        self.input_chunk = torch.tensor(input_chunk, dtype=torch.float).T
        self.output_chunk = torch.tensor([self.yy], dtype=torch.float).T

    def __u_input(self, x):
        y = -np.sin(np.pi * x)
        return y

    def get_data(self):
        return self.input_chunk, self.output_chunk

    def plot_valid_data(self):
        valid_indices = ~np.isnan(self.yy)
        plt.scatter(self.tt[valid_indices], self.xx[valid_indices], marker='o', color='blue', label='Valid data points')
        plt.xlabel('Time (t)')
        plt.ylabel('Space (x)')
        plt.title('Scatter Plot of Valid Data Points')
        plt.legend()
        plt.show()


class ModelBurgers(torch.nn.Module):
    def __init__(self):
        super(ModelBurgers, self).__init__()

        self.layers = []
        self.layers.append(torch.nn.Linear(2, 32))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(32, 16))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(16, 32))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(32, 1))

        self.net = torch.nn.Sequential(*self.layers)

    def forward(self, input_chunk):
        output = self.net(input_chunk)
        return output


class LossBurgers:
    def __init__(self):
        self.input_chunk, self.output_chunk, self.model_output = torch.Tensor, torch.Tensor, torch.Tensor
        self.u_x, self.u_t, self.u_xx = torch.Tensor, torch.Tensor, torch.Tensor
        self.nu = torch.tensor([3.184713375796178e-3], dtype=torch.float).cuda()
        print(self.nu)
        self.criterion = torch.nn.MSELoss()

    def __update(self, input_chunk, output_chunk, model_output_chunk):
        self.input_chunk = input_chunk
        self.output_chunk = output_chunk
        self.u = model_output_chunk

        self.u_x, self.u_t, self.u_xx = self.__get_gradient(self.u)

    def __get_gradient(self, target):
        input_dataset = self.input_chunk

        target_x = torch.autograd.grad(
            target, input_dataset,
            grad_outputs=torch.ones_like(target),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]

        target_t = torch.autograd.grad(
            target, input_dataset,
            grad_outputs=torch.ones_like(target),
            retain_graph=True,
            create_graph=True
        )[0][:, 1]

        target_xx = torch.autograd.grad(
            target_x, input_dataset,
            grad_outputs=torch.ones_like(target_x),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]

        return target_x, target_t, target_xx

    def get_loss(self, input_chunk, output_chunk, model_output_chunk):
        self.__update(input_chunk, output_chunk, model_output_chunk)
        # Burgers Equations
        value = self.u_t + self.u.squeeze() * self.u_x
        answer = self.nu * self.u_xx
        loss_eqn = self.criterion(value, answer)

        valid_mask = ~torch.isnan(output_chunk)
        valid_output = output_chunk[valid_mask]
        valid_model_output = model_output_chunk[valid_mask]

        # 관측 손실
        loss_obs = self.criterion(valid_model_output, valid_output)

        loss = loss_eqn + loss_obs

        return loss


class DatasetBurgers(Dataset):
    def __init__(self, input_chunk, output_chunk):
        self.data = input_chunk.clone().detach().requires_grad_(True)
        self.labels = output_chunk

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 데이터 및 모델 초기화
data = DataBurgers()
input_chunk, output_chunk = data.get_data()
dataset = DatasetBurgers(input_chunk.cuda(), output_chunk.cuda())
dataloader = DataLoader(dataset, shuffle=False, batch_size=10000)

model = ModelBurgers().cuda()
loss_func = LossBurgers()
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=5000)

# 상대 경로 설정
model_save_path = './model_burgers.pth'
optimizer_save_path = './optimizer_burgers.pth'


# 학습 루프
def closure():
    optimizer.zero_grad()
    total_loss = 0
    for input_chunk, output_chunk in dataloader:
        model_output_chunk = model(input_chunk)
        loss = loss_func.get_loss(input_chunk, output_chunk, model_output_chunk)
        total_loss += loss
    total_loss.backward()
    global loss_for_value
    loss_for_value = total_loss
    print(total_loss.item())
    return total_loss


optimizer.step(closure)

# 마지막 모델 및 옵티마이저 상태 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_save_path)


# 모델을 불러와서 평가하는 함수
def load_model(model_save_path):
    # 모델과 옵티마이저를 불러올 때 동일한 구조를 사용해야 합니다.
    model = ModelBurgers()  # MyModel()을 실제 모델 클래스명으로 변경하세요.

    # 저장된 상태 불러오기
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


model = load_model(model_save_path)

# 예측값을 사용하여 contour map 그리기
with torch.no_grad():
    model.eval()
    input_chunk = input_chunk.cuda()
    model_output_chunk = model(input_chunk).cpu().numpy()

# 컨투어 맵 그리기
xx = input_chunk.cpu().numpy()[:, 0].reshape(100, 100)
tt = input_chunk.cpu().numpy()[:, 1].reshape(100, 100)
uu = model_output_chunk.reshape(100, 100)

plt.contourf(tt, xx, uu, levels=100)
plt.colorbar()
plt.title('Contour Map of Model Output')
plt.xlabel('t')
plt.ylabel('x')
plt.show()
