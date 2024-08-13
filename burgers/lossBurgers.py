import torch


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
