import numpy as np
import matplotlib.pyplot as plt


def plot_specific_t(input_chunk, model_output_chunk, t_values):
    # 데이터 변환 및 리쉐이프
    xx = input_chunk.cpu().numpy()[:, 0].reshape(100, 100)
    tt = input_chunk.cpu().numpy()[:, 1].reshape(100, 100)
    uu = model_output_chunk.reshape(100, 100)

    # 서브플롯 생성
    fig, axes = plt.subplots(1, len(t_values), figsize=(15, 5), sharey=True)

    for i, t_value in enumerate(t_values):
        # 특정 t 값에 대한 인덱스 찾기
        t_index = np.abs(tt[:, 0] - t_value).argmin()

        # 해당 t 값에 대한 x와 u 값 추출
        x_values = xx[t_index, :]
        u_values = uu[t_index, :]

        # 서브플롯에 그래프 그리기
        axes[i].plot(x_values, u_values)
        axes[i].set_title(f't={t_value}')
        axes[i].set_xlabel('x')
        if i == 0:
            axes[i].set_ylabel('u(t,x)')

    plt.tight_layout()
    plt.show()
