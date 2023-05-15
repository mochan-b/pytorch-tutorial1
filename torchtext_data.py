from torchtext.datasets import WikiText2

if __name__ == '__main__':
    '''
    WikiText2 is a dataset of Wikipedia articles with tokens separated by spaces.
    Each element in the iterator is a string a line.
    '''
    train_iter = WikiText2(split='train')
    for i, item in enumerate(train_iter):
        print(i, item)
        if i > 10:
            break
