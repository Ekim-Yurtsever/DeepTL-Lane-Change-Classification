import numpy as np

def n_fold_sample(labels, n, itr, sort_idx=False):
    """
    Separates a binary label list into balanced training and validation splits for n-fold cross validation.
    :param labels: 1d vector of binary labels
    :param n: n-fold cross validation
    :param itr: current fold itr (from 0 to n-1)
    :return: training_label_idx and validation_label_idx for the current fold itr
    """
    if itr>=n:
        print("Iteration number is 0-indexed and must be smaller than n. Returning nothing")
        return None, None
    # Setup
    labels = np.array(labels)
    pos_idx = np.where(labels==1)[0]
    neg_idx = np.where(labels==0)[0]
    num_pos_idx_itr = np.floor(pos_idx.size / n).astype(np.int)
    num_neg_idx_itr = np.floor(neg_idx.size / n).astype(np.int)
    pos_idx_remainder = pos_idx.size % n
    neg_idx_remainder = neg_idx.size % n
    # print(pos_idx.size, neg_idx.size, num_pos_idx_itr, num_neg_idx_itr, pos_idx_remainder, neg_idx_remainder)

    # Get idx
    if itr < pos_idx_remainder:
        idx = np.arange(i*num_pos_idx_itr+itr, (i+1)*num_pos_idx_itr+itr+1)
        pos_valid_idx = pos_idx[idx]
        pos_train_idx = np.delete(pos_idx, idx)
    else:
        idx = np.arange(i*num_pos_idx_itr+pos_idx_remainder, (i+1)*num_pos_idx_itr+pos_idx_remainder)
        pos_valid_idx = pos_idx[idx]
        pos_train_idx = np.delete(pos_idx, idx)
    if itr < neg_idx_remainder:
        idx = np.arange(i*num_neg_idx_itr+itr, (i+1)*num_neg_idx_itr+itr+1)
        neg_valid_idx = neg_idx[idx]
        neg_train_idx = np.delete(neg_idx, idx)
    else:
        idx = np.arange(i*num_neg_idx_itr+neg_idx_remainder,(i+1)*num_neg_idx_itr+neg_idx_remainder)
        neg_valid_idx = neg_idx[idx]
        neg_train_idx = np.delete(neg_idx, idx)

    # Build lists
    training_idx = np.concatenate((pos_train_idx, neg_train_idx))
    validation_idx = np.concatenate((pos_valid_idx, neg_valid_idx))
    if sort_idx:
        training_idx.sort()
        validation_idx.sort()
    return training_idx, validation_idx

if __name__== "__main__":
    # np.random.seed(192837465)
    labels = np.random.randint(0,2,50)*np.random.randint(0,2,50)*np.random.randint(0,2,50)
    n = 5
    i = 0
    print("pos labels:", np.where(labels==1))
    print("neg labels:", np.where(labels==0))
    for i in range(n):
        print("iteration {} of {}".format(i+1,n))
        train_idx, valid_idx = n_fold_sample(labels, n, i)
        print("validation idx:", valid_idx)
        print("training idx:", train_idx)