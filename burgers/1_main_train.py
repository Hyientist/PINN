import torch
from torch.utils.data import DataLoader

from dataBurgers import DataBurgers
from modelBurgers import ModelBurgers
from datasetBurgers import DatasetBurgers
from lossBurgers import LossBurgers
from lbfgs_closure import LBFGSClosure


# 데이터 및 모델 초기화
data = DataBurgers()
input_chunk, output_chunk = data.get_data()
dataset = DatasetBurgers(input_chunk.cuda(), output_chunk.cuda())
dataloader = DataLoader(dataset, shuffle=False, batch_size=10000)

model = ModelBurgers().cuda()
loss_func = LossBurgers()
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=5000)
lbfgs_closure = LBFGSClosure(optimizer=optimizer, dataloader=dataloader, model=model, loss_func=loss_func)

# 상대 경로 설정
model_save_path = './model_burgers.pth'
optimizer_save_path = './optimizer_burgers.pth'

optimizer.step(lbfgs_closure.closure)

# 마지막 모델 및 옵티마이저 상태 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, model_save_path)
