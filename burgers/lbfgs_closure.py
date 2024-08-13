
class LBFGSClosure:
    def __init__(self, optimizer, dataloader, model, loss_func):
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.model = model
        self.loss_func = loss_func

    def closure(self):
        self.optimizer.zero_grad()
        total_loss = 0
        for input_chunk, output_chunk in self.dataloader:
            model_output_chunk = self.model(input_chunk)
            loss = self.loss_func.get_loss(input_chunk, output_chunk, model_output_chunk)
            total_loss += loss
        total_loss.backward()
        print(total_loss.item())
        return total_loss
