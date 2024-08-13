import torch
import matplotlib.pyplot as plt
from modelBurgers import ModelBurgers
from dataBurgers import DataBurgers
from plot_specific_t import plot_specific_t

model_save_path = './model_burgers.pth'
model = ModelBurgers()

# 저장된 상태 불러오기
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])

data = DataBurgers()
input_chunk, output_chunk = data.get_data()

# 예측값을 사용하여 contour map 그리기
with torch.no_grad():
    model.eval()
    input_chunk = input_chunk
    model_output_chunk = model(input_chunk).numpy()

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

t_values = [0.25, 0.5, 0.75]  # 특정 t 값들 설정
plot_specific_t(input_chunk, model_output_chunk, t_values)
