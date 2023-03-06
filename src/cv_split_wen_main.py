import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2.robjects import r, numpy2ri

# Load R packages
glmnet = rpackages.importr('glmnet')
Matrix = rpackages.importr('Matrix')

def CV_SWEN_Main(x, y, type_, G, include_intercept, alpha_s, alpha_d, n_lambda_sparsity,
                 n_lambda_diversity, tolerance, max_iter, n_folds, active_set, full_diversity, n_threads):

    # Case for a single model
    if G == 1:
        model = glmnet.cv_wen(x, y, type_, include_intercept, alpha_s, n_lambda_sparsity,
                              tolerance, max_iter, n_folds, active_set, n_threads)

        return model

    # Case for multiple models
    else:
        split = glmnet.split_wen(x, y, G, include_intercept, alpha_s, alpha_d, n_lambda_sparsity,
                                 n_lambda_diversity, tolerance, max_iter, n_folds, active_set,
                                 full_diversity, n_threads)

        beta_hat = np.array(split.rx2("beta_hat"))
        num_splits = np.array(split.rx2("num_splits"))

        return {"beta_hat": beta_hat, "num_splits": num_splits}
