import numpy as np
import itertools
from sklearn import cross_validation
from sklearn import linear_model
from scipy import linalg


def test_permutation(activation_img, condition, modality, verbose=False, n_perm=10000):
    """
    Parameters
    ==========
    activation_img: array, shape (n_sub, n_samples, n_features)
        The activation image (aka beta-map)
    condition: integer array, shape (n_sub, n_samples,)
        The condition for every sample.
    modality: integer array, shape (n_sub, n_samples,)
        Modality for each sample. For now this is limited to two
        modalities.
    subj: integer array, shape (n_samples,)
        Array indicating to which subject do the activation_img
        correspond. Correlations will only be computed within
        the same subject. In case all images correspond to a single
        subject set this array to a constant value.
    n_perm : integer
        number of permutations.
    Returns
    =======
    pval: float
    T0 : test statistic
    T_perm : all permuted statistics
    """
    condition = np.array(condition)
    modality = np.array(modality)

    # check the data
    n_sub = len(activation_img)
    unique_modalities = np.unique(modality)
    assert len(unique_modalities) == 2
    unique_conditions = np.unique(condition)
    assert np.unique(condition).size == 2

    if verbose:
        print('%s subjects were given' % n_sub)

    for s in range(n_sub):
        n_samples, n_features = activation_img[s].shape
        assert len(condition[s]) == n_samples
        assert len(modality[s]) == n_samples

    if verbose:
        print('All sub data are fine')

    # compute test statistic
    T0 = 0.0
    for s in range(n_sub):
        img_cond_modality_1 = []
        img_cond_modality_2 = []
        idx_mod_1 = (modality[s] == unique_modalities[0])
        idx_mod_2 = (modality[s] == unique_modalities[1])
        for cond in unique_conditions:
            idx_cond_m1 = (condition[s] == cond) & idx_mod_1
            idx_cond_m2 = (condition[s] == cond) & idx_mod_2
            img_cond_modality_1.append(activation_img[s][idx_cond_m1].mean(0))
            img_cond_modality_2.append(activation_img[s][idx_cond_m2].mean(0))
        img_cond_modality_1 = np.array(img_cond_modality_1)
        img_cond_modality_2 = np.array(img_cond_modality_2)

        # should you here keep a separate T0 for each sub?
        T0 += test_stat(img_cond_modality_1, img_cond_modality_2)
      #  print ("test_stat", T0)
    # compute the permuted statistic
    T_perm = []
    for perm_count in range(n_perm):
        T_subj = 0.0

        # permute one modality
        # permute the conditions (on non-averaged betas to have more permutations)
        # idx_1 = np.arange(n_samples)[modality == unique_modalities[0]]
        idx_2 = np.arange(n_samples)[modality[0] == unique_modalities[0]]
        condition_perm = condition[0].copy()
        condition_perm[idx_2] = condition_perm[idx_2][np.random.permutation(idx_2.size)]

        for s in range(n_sub):
            img_cond_modality_1 = []
            img_cond_modality_2 = []
            idx_mod_1 = (modality[s] == unique_modalities[0])
            idx_mod_2 = (modality[s] == unique_modalities[1])
            for cond in unique_conditions:
                idx_1 = (condition_perm == cond) & idx_mod_1
                idx_2 = (condition_perm == cond) & idx_mod_2
                img_cond_modality_1.append(activation_img[s][idx_1].mean(0))
                img_cond_modality_2.append(activation_img[s][idx_2].mean(0))
            img_cond_modality_1 = np.array(img_cond_modality_1)
            img_cond_modality_2 = np.array(img_cond_modality_2)
            T_subj += test_stat(img_cond_modality_1, img_cond_modality_2)

        T_perm.append(T_subj)
    T_perm = np.array(T_perm)

    pval =1-((T0 >= T_perm).sum())/n_perm  #1-(T0 >= T_perm).mean() #1 - (T0 >= T_perm).mean()
    return pval, T0, T_perm


# test statistic
def test_stat(img_cond_modality_1, img_cond_modality_2):
    """
    This is the test statistic used by the permutation test
    Parameters
    ==========
    img_cond_modality_1 : array, shape (n_conditions, n_features)
    img_cond_modality_2 : array, shape (n_conditions, n_features)
    """
    n_conditions = img_cond_modality_1.shape[0]
    conditions = np.arange(n_conditions)

    # initialize
    within_condition_t = 0
    within_condition_counter = 0
    cross_condition_t = 0
    cross_condition_counter = 0

    # generate all pairwise comparisons
    for [a, b] in itertools.product(conditions, conditions):
    #    print ("a, b:", a, b)
        A = img_cond_modality_1[a]
        B = img_cond_modality_2[b]
      #  print (a, b)
        if a == b:
            
            within_condition_counter += 1
            within_condition_t += np.corrcoef(A, B)[0, 1]
       #     print ("tic")
        else:
            
            cross_condition_counter += 1
            cross_condition_t += np.corrcoef(A, B)[0, 1]
         #   print ("tac")
    assert within_condition_counter > 0
    assert cross_condition_counter > 0
  #  print ("counters within, between:", within_condition_counter, cross_condition_counter)
    ret = within_condition_t / float(within_condition_counter) - \
        cross_condition_t / float(cross_condition_counter)
    assert np.isfinite(ret)
   # print ("test stat:", ret)
    return ret


def test_decoding(img, condition, modality, n_perm=10000):
    # Data
    modality = np.array(modality)
    condition = np.array(condition)
    m1, m2 = np.unique(modality)
        
    # number of subjects
    n_sub = modality.shape[0]

    # initialization
    score = 0.
    scores_perm = []

    # create permutations beforehand so they are shared
    # across subjects

    for i in range(n_sub):
       # print (n_sub)
     #   print (i, modality[i], m1)
        X1 = img[i][modality[i] == m1]
     #   print (X1.shape)
        y1 = condition[i][modality[i] == m1]
      #  print (y1.shape)
        X2 = img[i][modality[i] == m2]
     #   print (X2.shape)
        y2 = condition[i][modality[i] == m2]
      #  print (y2.shape)
        cv = cross_validation.ShuffleSplit(X1.shape[0],50)
        # clf = linear_model.LogisticRegression(cv=cv, Cs=5)
        clf = linear_model.LogisticRegression()
        clf.fit(X1, y1.ravel())
        score += clf.score(X2, y2.ravel())
        perms = [np.random.permutation(X2.shape[0]) for _ in range(n_perm)]

        # compute the permuted test statistic
        scores_perm_sub = []
        for pi in perms:
            tmp = clf.score(X2, y2.ravel()[pi])
            scores_perm_sub.append(tmp)
        scores_perm.append(scores_perm_sub)

    # add across subjects
    scores_perm = np.array(scores_perm).sum(0)

    pval = 1- ((score > scores_perm).sum())/n_perm #1-(score > scores_perm).mean() #1 - (score > scores_perm).mean()

    return pval, score, scores_perm


def generate_synthetic_data2(n_samples, width, alpha, noise):
    """
       alpha = modality effect
       beta = condition effect
    """
    M1 = np.random.randn(width, width)
    M2 = np.random.randn(width, width)
    C1 = np.random.randn(width, width)
    C2 = np.random.randn(width, width)

    M1 /= linalg.norm(M1)
    M2 /= linalg.norm(M2)
    C1 /= linalg.norm(C1)
    C2 /= linalg.norm(C2)
    
    ground_truth = (M1, M2, C1, C2)

    samples_A1 = []
    for i in range(n_samples):
        tmp = M1 + noise * np.random.randn(width, width) + alpha * C1
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_A1.append(tmp.ravel())

    samples_A2 = []
    for i in range(n_samples):
        tmp = M1 + noise * np.random.randn(width, width) + alpha * C2
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_A2.append(tmp.ravel())

    samples_B1 = []
    for i in range(n_samples):
        tmp = M2 + noise * np.random.randn(width, width) + alpha * C1
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_B1.append(tmp.ravel())

    samples_B2 = []
    for i in range(n_samples):
        tmp = M2 + noise * np.random.randn(width, width) + alpha * C2
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_B2.append(tmp.ravel())
        
    samples = np.concatenate((samples_A1, samples_A2, samples_B1, samples_B2), axis=0)
    assert np.isnan(samples).sum() == 0

    condition = [0] * n_samples + [1] * n_samples + [0] * n_samples + [1] * n_samples
    modality = [0] * (2 * n_samples) + [1] * (2 * n_samples)
    return ground_truth, samples, condition, modality

        
def generate_synthetic_data(n_samples, width, noise_amplitude=0.5, only_noise=False):

    if only_noise:
        w_A1 = np.zeros((width, width))
        w_A2 = np.zeros((width, width))
        w_B1 = np.zeros((width, width))
        w_B2 = np.zeros((width, width))
    else:
        w_A1 = np.zeros((width, width))
        w_A1[20:30, 20:30] = 1.
        w_A1[160:170, 160:170] = 1.

        w_A2 = np.zeros((width, width))
        w_A2[20:30, 20:30] = 1.
        #     w_A2[50:60, 20:30] = 1.
        w_A2[20:30, 160:170] = 1.
        #     w_A2[160:170, 20:30] = 1.

        w_B1 = np.zeros((width, width))
        w_B1[160:170, 20:30] = 1.
        #     w_B1[50:60, 20:30] = 1.
        w_B1[160:170, 160:170] = 1.
        w_B1[20:30, 160:170] = 1.

        w_B2 = np.zeros((width, width))
        w_B2[160:170, 20:30] = 1.
        w_B2[20:30, 160:170] = 1.

    ground_truth = (w_A1, w_A2, w_B1, w_B2)

    samples_A1 = []
    for i in range(n_samples):
        tmp = w_A1 + noise_amplitude * np.random.randn(*w_A1.shape)
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_A1.append(tmp.ravel())

    samples_A2 = []
    for i in range(n_samples):
        tmp = w_A2 + noise_amplitude * np.random.randn(*w_A1.shape)
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_A2.append(tmp.ravel())

    samples_B1 = []
    for i in range(n_samples):
        tmp = w_B1 + noise_amplitude * np.random.randn(*w_A1.shape)
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_B1.append(tmp.ravel())

    samples_B2 = []
    for i in range(n_samples):
        tmp = w_B2 + noise_amplitude * np.random.randn(*w_A1.shape)
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_B2.append(tmp.ravel())


    samples = np.concatenate((samples_A1, samples_A2, samples_B1, samples_B2), axis=0)
    assert np.isnan(samples).sum() == 0

    condition = [0] * n_samples + [1] * n_samples + [0] * n_samples + [1] * n_samples
    modality = [0] * (2 * n_samples) + [1] * (2 * n_samples)
    return ground_truth, samples, condition, modality

if __name__ == '__main__':
    
    img_1='VS_delay_3_l+r_OSC_betamaps'  #VS_delay_0
    img_2='Im_l+r_OSC_betamaps'
    labels_1='VS_delay_3_labels'
    labels_2='Im_labels'
    mod_1=np.zeros([9, 32]).astype('int32')
    
    mod_2=np.ones([9, 32]).astype('int32')
  #  mod_2=(mod_1+1).astype('int32')
    X1 = np.load('data/' + img_1 + '.npy')
    y1 = np.load('data/' + labels_1 + '.npy')
 #   print (y1[2])
  #  print (y1.shape) #, y2.shape)
    y1=y1.reshape(9, 32)  #y1.reshape(9, 32)
  #  print (y1[2])
    X2 = np.load('data/' + img_2 + '.npy')
    y2 = np.load('data/' + labels_2 + '.npy')  
 #   print (y2[1])
    y2= y2.reshape(9, 32) #y2.reshape(9, 32)
  #  print (y2[1])
   # print (y1.shape, y2.shape)
    activation_img=np.hstack([X1, X2])
    condition=np.hstack([y1, y2])
  #  print (condition.shape)
    modality=np.hstack([mod_1, mod_2])
    
    
    print(img_1, img_2)
    # compute score
    pval, score1, score2 = test_decoding(activation_img, condition, modality, n_perm=10000)
    print('p-value decoding: ', pval)

    pval1, T, Tperm= test_permutation(activation_img, condition, modality, verbose=False, n_perm=10000)
    print('p-value permutation: ', pval1)