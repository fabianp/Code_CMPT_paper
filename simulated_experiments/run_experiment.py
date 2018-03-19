import os
from CMPT import *
from joblib import delayed, Parallel
np.random.seed(0)

run = 0
n_jobs = -1
alpha_max = 6
# all_width = [100, 1000, 10000]
all_width = [1000, 10000]
for width in all_width:
    print('width', width)
    n_alphas = 20
    n_iters = 100
    n_perm = 10000
    alphas = np.linspace(0, alpha_max, n_alphas)
    noise = 0.5
    n_samples = 40
    
    if not os.path.exists('data'):
        os.mkdir('data')

    np.save('data/alphas_%s.npy' % width, alphas)

    def column_errors(i):
        pvals = []
        for j in range(n_alphas):
            _, samples, condition, modality = generate_synthetic_data2(
                n_samples, width, alphas[j], noise)
            pval, _, _ = test_permutation(
                [samples], [condition], [modality], verbose=False, n_perm=n_perm)
            pvals.append(pval)
        return pvals
    out_cmpt = Parallel(n_jobs=n_jobs, verbose=1)(delayed(column_errors)(i) for i in range(n_iters))
    out_cmpt = np.array(out_cmpt)
    np.save('data/cmpt_%s_%s.npy' % (width, run), out_cmpt)

    # def column_errors_decoding(i):
    #     pvals = []
    #     for j in range(n_alphas):
    #         _, samples, condition, modality = generate_synthetic_data2(
    #             n_samples, width, alphas[j], noise)
    #         pval, _, _ = test_decoding([samples], [condition], [modality], n_perm=n_perm)
    #         pvals.append(pval)
    #     return pvals
    # out_decoding = Parallel(n_jobs=n_jobs, verbose=2)(delayed(column_errors_decoding)(i) for i in range(n_iters))
    # out_decoding = np.array(out_decoding)
    # np.save('data/decoding_%s_%s.npy' % (width, run), out_decoding)
