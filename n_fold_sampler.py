import numpy as np

def n_fold_sample(labels, n, itr, train_ratio=None, sort_idx=True):
    """
    Separates a binary label list into balanced training and validation splits for n-fold cross validation.
    :param labels: 1d vector of binary labels
    :param n: n-fold cross validation
    :param itr: current fold itr (from 0 to n-1)
    :param train_ratio: percentage ratio ( ]0.1,1] ) of data in training set
    :return: training_label_idx and validation_label_idx for the current fold itr
    """
    # Bad usage handling
    if itr >= n:
        print("Iteration number is 0-indexed and must be smaller than n. Returning nothing")
        return None, None

    # Setup
    if train_ratio == None:
        train_ratio = 1 - 1/n
        valid_ratio = 1/n
    else:
        valid_ratio = 1 - train_ratio
    labels = np.array(labels)
    pos_label_idx = np.where(labels==1)[0]
    neg_label_idx = np.where(labels==0)[0]
    num_pos_idx_itr = np.floor(pos_label_idx.size * valid_ratio).astype(np.int)
    num_neg_idx_itr = np.floor(neg_label_idx.size * valid_ratio).astype(np.int)

    # Get idx range
    if n == 1:
        pos_start_idx = 0
        neg_start_idx = 0
    else:
        pos_start_idx = np.floor((pos_label_idx.size - num_pos_idx_itr) * itr / (n-1)).astype(np.int)
        neg_start_idx = np.floor((neg_label_idx.size - num_neg_idx_itr) * itr / (n-1)).astype(np.int)
    pos_idx = np.arange(pos_start_idx, pos_start_idx + num_pos_idx_itr)
    neg_idx = np.arange(neg_start_idx, neg_start_idx + num_neg_idx_itr)

    # Get idx
    pos_valid_idx = pos_label_idx[pos_idx]
    pos_train_idx = np.delete(pos_label_idx, pos_idx)
    neg_valid_idx = neg_label_idx[neg_idx]
    neg_train_idx = np.delete(neg_label_idx, neg_idx)

    # Build lists and return
    training_idx = np.concatenate((pos_train_idx, neg_train_idx))
    validation_idx = np.concatenate((pos_valid_idx, neg_valid_idx))
    if sort_idx: # This sorts indices back, otherwise ordered as positive and negative labels
        training_idx.sort()
        validation_idx.sort()
    return training_idx, validation_idx

if __name__== "__main__":
    # np.random.seed(192837465)
    labels = np.random.randint(0,2,100)*np.random.randint(0,2,100)*np.random.randint(0,2,100)
    n = 10
    print("pos labels:", np.where(labels==1)[0], "num. labels = ", np.where(labels==1)[0].size)
    print("neg labels:", np.where(labels==0)[0], "num. labels = ", np.where(labels==0)[0].size)
    for i in range(n):
        print("iteration {} of {}".format(i+1,n))
        train_idx, valid_idx = n_fold_sample(labels, n, i, train_ratio=0.6)
        # print("validation idx:", valid_idx)
        # print("training idx:", train_idx)