import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def find_smaller_than_T(nums, T):
    indices = [index for index, num in enumerate(nums) if num < T]
    if indices:
        return True, indices
    else:
        return False, None