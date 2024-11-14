from dgl.data import DGLDataset


def get_splits_window(dataset: DGLDataset, n_slot, splits):
    """
    Given the data, split it into random subsets of train, val, and test as given by splits
    :param dataset: DGLDataset object to split
    :param n_slot: Number of possible sliding windows in a day
    :param splits: (train, val, test) ratios
    """
    split_train, split_val, _ = splits
    i = n_slot * split_train
    j = n_slot * split_val
    train = dataset[:i]
    val = dataset[i : i + j]
    test = dataset[i + j :]

    return train, val, test


def get_splits(dataset: DGLDataset, ratios):
    """
    Given the data, split it into train, val, and test subsets.
    :param dataset: DGLDataset object to split
    :return: A tuple containing train, val, and test subsets
    """

    # Calculate split indices
    num_graphs = len(dataset)
    num_train = int(ratios[0] * num_graphs)
    num_val = int(ratios[1] * num_graphs)

    # Split without shuffling to maintain order
    train_set = dataset[:num_train]
    val_set = dataset[num_train : num_train + num_val]
    test_set = dataset[num_train + num_val :]

    return train_set, val_set, test_set
