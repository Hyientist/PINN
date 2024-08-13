import numpy as np
import torch
import matplotlib.pyplot as plt


class DataBurgers:
    def __init__(self):
        """
        초기 데이터를 준비하는 객체입니다.
        초기 조건과 드리클레 조건을 적용한 데이터를 얻을 수 있습니다.
        """
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

        self.plot_valid_data()

    @staticmethod
    def __u_input(x):
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
