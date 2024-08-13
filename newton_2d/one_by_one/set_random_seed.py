import random
import numpy as np
import torch


def set_random_seed(random_seed):
    try:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        print(f"random seed set as {random_seed}")
    except Exception as e:
        print(f"can't set random seed check following error statement: {e}")
