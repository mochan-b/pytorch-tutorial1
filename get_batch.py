import torch
from torch import Tensor
from typing import Tuple

bptt = 35


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


if __name__ == '__main__':
    """
    Test get_batch and print out the shape of data and target
    """
    test_data = torch.arange(0, (bptt + 3) * 10).reshape(-1, 10)
    batch_data, batch_target = get_batch(test_data, 0)
    print(batch_data.shape)
    print(batch_target.shape)
    print(batch_data)
    print(batch_target)
