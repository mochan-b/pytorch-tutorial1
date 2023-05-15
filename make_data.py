import torch
from torch import Tensor
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset

from get_batch import bptt, get_batch


def get_data():
    """
    Get data from torchtext.datasets.WikiText2 ready for training
    :return: train_data, val_data, test_data which are ready to be used
    """
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    ntokens = len(vocab)  # size of vocabulary

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    def batchify(data: Tensor, bsz: int) -> Tensor:
        """Divides the data into ``bsz`` separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Arguments:
            data: Tensor, shape [N]
            bsz: int, batch size

        Returns:
            Tensor of shape ``[N // bsz, bsz]``
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    return train_data, val_data, test_data, ntokens


if __name__ == '__main__':
    """
    Test the get_data function that gets the iterator for training, validation and test data from WikiText2
    """
    train_data, val_data, test_data, ntokens = get_data()

    for i in range(0, test_data.size(0) - 1, bptt):
        data, targets = get_batch(test_data, i)
        print(data.shape)
        print(targets.shape)
        # Print the first training example and the target for it
        print(data[:, 0])
        print(targets[::10])
        break
