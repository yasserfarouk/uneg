import os
import random

random.seed(0)
from math import sqrt, log
from typing import Tuple, List, Optional

import numpy as np
import negmas

np.random.seed(0)
from scipy.optimize import minimize, fsolve
import scipy

scipy.random.seed(0)
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

eps = 1e-12


def generate_ufun(n_outcomes: int) -> np.array:
    """
    Generates a random ufun as an n_outcomes vector

    Args:
        n_outcomes:

    """
    return np.random.uniform(size=n_outcomes)


def generate_genius_ufuns(folder="./data/1d", max_n_outcomes=None) -> List[np.array]:
    ufuns = []
    folder = os.path.abspath(folder)
    for dir in sorted(os.listdir(folder)):
        full_name = os.path.join(folder, dir)
        if not os.path.isdir(full_name):
            continue
        if max_n_outcomes is not None and int(dir[:6]) > max_n_outcomes:
            continue
        mechanism, agent_info, issues = negmas.load_genius_domain_from_folder(folder_name=full_name)
        for negotiator in agent_info:
            ufuns.append(np.asarray(list(negotiator["ufun"].mapping.values())))
    return ufuns


def generate_random_ufuns(n_outcomes, n_trials) -> List[np.array]:
    return [generate_ufun(n_outcomes) for _ in range(n_trials)]


def generate_data(ufun: np.array, n: int, sigma: float) -> np.array:
    """
    Generate comparisons given a ufun

    Args:
        ufun: the utility function as a n_outcomes vector
        n: number of comparisons
        sigma: the std. dev. of the noise to be added to ufun

    Returns:

    """
    ufun = np.asarray(ufun, dtype=float).flatten().copy()
    n_outcomes = len(ufun)

    noisy_comparisons, gt_comparisons = [], []

    def relation(f0_, f1_):
        return 1 if f0_ > f1_ else 0 if f0_ == f1_ else -1

    for _ in range(n):
        w0, w1 = random.randint(0, n_outcomes - 1), random.randint(0, n_outcomes - 1)
        while w0 == w1:
            w0, w1 = random.randint(0, n_outcomes - 1), random.randint(0, n_outcomes - 1)
        f0 = ufun[w0] + (np.random.randn(1))[0] * sigma
        f1 = ufun[w1] + (np.random.randn(1))[0] * sigma
        c = relation(f0, f1)
        noisy_comparisons.append([w0, w1, c])
        c = relation(ufun[w0], ufun[w1])
        gt_comparisons.append([w0, w1, c])
    return np.array(noisy_comparisons), np.array(gt_comparisons)


def extract_ws(data: np.array) -> np.array:
    return np.unique(np.vstack((data[:, 0], data[:, 1]))).transpose()


def create_f0(data: np.array) -> np.array:
    ws = extract_ws(data)
    ys = np.random.uniform(0.0, 1.0, *ws.shape)  # Amy made me write it this way :-(
    return create_table(ws, ys)


def create_table(ws, ys) -> np.array:
    """
    Takes ws and ys and generates a table of them (ws in first column)

    Args:
        ws:
        ys:

    Returns:

    """
    ws = np.asarray(ws).flatten().reshape(len(ws), 1)
    ys = np.asarray(ys).flatten().reshape(len(ws), 1)
    return np.hstack((ws, ys))


def z(f0, f1, sigma):
    return (f0 - f1) / (sqrt(2) * sigma)


def correct_order(w0, w1, c):
    if c == 0:
        return None, None
    if c < 0:
        return w1, w0
    return w0, w1


def likelihood(ys: np.array, data: np.array, ws: np.array, sigma: float, epsilon: float = eps) -> float:
    """
    Calculates the likelihood

    Args:
        ys: The parameters to be optimized ( f(w) for w in ws )
        ws: The outcomes at which we are estimating f(w)
        data: list (or ndarray) of comparisons in the format (w0, w1, i) with the quicksort() meaning
        sigma: The noise std. dev.
        epsilon: a small real number to avoid numerical problems
    """

    if ws is None:
        ws = extract_ws(data)

    l = 0  # likelihood

    f = dict(zip(ws, ys))

    for w0, w1, relation in data:
        w0, w1 = correct_order(w0, w1, relation)
        if w0 is None:
            continue
        phi = norm.cdf(z(f[w0], f[w1], sigma))
        if phi < epsilon:
            continue
        l -= log(phi)
    return l


def jacobian(ys: np.array, data: np.array, ws: np.array, sigma: float
             , epsilon: float = eps) -> np.array:
    """First derivative of the likelihood"""
    jac = np.zeros(len(ws))

    f = dict(zip(ws, ys))
    indx = dict(zip(ws, range(len(ws))))

    for w0, w1, relation in data:
        w0, w1 = correct_order(w0, w1, relation)
        if w0 is None:
            continue

        zk = z(f[w0], f[w1], sigma)

        phi = norm.cdf(zk) * sqrt(2) * sigma
        if phi < epsilon:
            # print(f'?', end='')
            continue
        delta = norm.pdf(zk) / phi
        jac[indx[w0]] -= delta
        jac[indx[w1]] += delta
    return jac


def hessian(ys: np.array, data: np.array, ws: np.array, sigma: float
            , epsilon: float = eps, delta: float = eps) -> np.array:
    """Second derivative of the likelihood"""
    h = np.eye(len(ws)) * delta

    f = dict(zip(ws, ys))
    indx = dict(zip(ws, range(len(ws))))

    s2sigma = (sqrt(2) * sigma * sigma)

    for w0, w1, relation in data:
        w0, w1 = correct_order(w0, w1, relation)
        if w0 is None:
            continue

        zk = z(f[w0], f[w1], sigma)

        phi = norm.cdf(zk)
        if phi < epsilon:
            continue
        pdf = norm.pdf(zk)

        delta = (pdf * pdf / (phi * phi) + zk * pdf / phi) / s2sigma
        i0, i1 = indx[w0], indx[w1]
        h[i0, i1] -= delta
        h[i1, i0] -= delta
        h[i0, i0] += delta
        h[i1, i1] += delta
    # print(h)
    return h


def test_likelihood(table, data, n_outcomes, sigma):
    l0 = likelihood(np.linspace(1, 0, n_outcomes), data, table[:, 0], sigma)
    l1 = likelihood(np.random.randn(n_outcomes), data, table[:, 0], sigma)
    l2 = likelihood(np.linspace(0, 1, n_outcomes), data, table[:, 0], sigma)
    assert l0 >= l1 >= l2


def find_errors(u: np.array, data: np.array, tolerance: float = 0.0) -> List[Tuple[int, float]]:
    """
    Tests that the result is compatible with ufun given comparisons in data

    Args:
        u: A table with w in column 1 and ufun value in column 2
        data: Should be generated with no noise
        tolerance: tolerance

    Returns:

    """

    errors = []
    f = dict(zip(u[:, 0], u[:, 1]))
    for i, (w0, w1, c) in enumerate(data):
        if c < 0 and (f[w0] - f[w1]) > tolerance:
            errors.append((i, abs(f[w0] - f[w1])))
        elif c > 0 and (f[w0] - f[w1]) < -tolerance:
            errors.append((i, abs(f[w0] - f[w1])))
    return errors


def create_gp(kernel=None, adaptive_sigma=1.0, normalize_y=False) -> GaussianProcessRegressor:
    """
    Creates a GP

    Args:
        kernel: The basic kernel (without noise)
        adaptive_sigma: The noise level (to be optimized as a hyper parameter)
        normalize_y: If true, Y will be normalized before fitting when fit_gp() is called

    Returns:

    """
    if kernel is None:
        kernel = Matern()
    if adaptive_sigma != 0.0:
        kernel += WhiteKernel(noise_level=adaptive_sigma)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y)


def fit_gp(gp: Optional[GaussianProcessRegressor], xy: np.array) -> GaussianProcessRegressor:
    """
    Fits a GP

    Args:
        gp: Gaussian process
        xy: Data as an n*2 matrix with first column giving xs and second giving ys (trivial to generalize)
    """
    if gp is None:
        gp = create_gp()
    xy = np.asarray(xy)
    gp.fit(X=xy[:, :-1], y=xy[:, -1])
    return gp


def sigma_inv(gp: GaussianProcessRegressor, ws) -> np.array:
    k = gp.kernel_
    n = len(ws)
    cov = np.zeros((n, n))
    for i in range(n):
        cov[i, i] = k(np.eye(1) * ws[i], np.eye(1) * ws[i])
        for j in range(i + 1, n):
            cov[i, j] = k(np.eye(1) * ws[i], np.eye(1) * ws[j])
            cov[j, i] = cov[i, j]
    return np.linalg.pinv(cov)


def prediction(gp: GaussianProcessRegressor, ws) -> np.array:
    return gp.predict(ws.reshape(-1, 1))


def prior(gp: GaussianProcessRegressor, ws) -> np.array:
    f = prediction(gp, ws)
    return 0.5 * f.transpose() @ sigma_inv(gp, ws) @ f


def max_apriori(ys: np.array, data: np.array, ws: np.array, sigma: float
                , epsilon: float = eps) -> float:
    gp = create_gp_from_table(ws, ys)
    l = likelihood(ys, data, ws, sigma, epsilon) + prior(gp, ws)
    return l


def create_gp_from_table(ws: np.array, ys: np.array) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(optimizer=None)
    gp.fit(ws.reshape(-1, 1), ys.reshape(-1, 1))
    return gp


def dsdf(ys: np.array, data: np.array, ws: np.array, sigma: float
         , epsilon: float = eps) -> np.array:
    gp = create_gp_from_table(ws, ys)
    # ys = prediction(gp, ws.reshape(-1, 1)).flatten()
    j = jacobian(ys, data, ws, sigma, epsilon)
    x = sigma_inv(gp, ws) @ ys
    return j + x


def main():
    # Ufun Generation Parameters
    genius_ufuns = True
    sigma_generator = 0.0
    n_trials = 100  # only used with random ufuns
    n_outcomes = 100  # interpreted as max_n_outcomes for genius ufuns

    # Comparison generation Parameters
    n_comparisons_per_outcome = 2

    # GP parameters
    do_gp = True
    kernel = Matern()
    learn_sigma = False

    # Learner Parameters

    use_their_map = False
    use_our_map = True
    if use_their_map:
        do_gp = False
    sigma_learner = 0.1
    tol = 1e-3

    optimizer_options_nm = {'maxiter': None, 'maxfev': None, 'disp': False, 'return_all': False,
                            'initial_simplex': None, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
    optimizer_options = {'xtol': tol, 'eps': 1.4901161193847656e-08, 'maxiter': 1000, 'disp': False, 'return_all': True}

    n_failed, n_failed_gp, n_failed_optimization = 0, 0, 0

    avg_errs = 0.0

    if genius_ufuns:
        ufuns = generate_genius_ufuns(max_n_outcomes=n_outcomes)
    else:
        ufuns = generate_random_ufuns(n_outcomes, n_trials)

    n_trials = len(ufuns)

    comparisons = [generate_data(ufun=ufun, n=n_comparisons_per_outcome * len(ufun), sigma=sigma_generator)
                   for ufun in ufuns]

    for trial, (ufun, (noisy_comparisons, gt_comparisons)) in enumerate(zip(ufuns, comparisons)):
        n_real_outcomes = len(ufun)
        ws = extract_ws(noisy_comparisons)
        errors = find_errors(create_table(ws, [ufun[w] for w in ws]), noisy_comparisons, tolerance=0.0)
        if len(errors) > 0 and sigma_generator == 0.0:
            print(f'failed on GT :{ufun}\n\n{[(e[1], noisy_comparisons[e[0], :]) for e in errors]}')
            return
        errors = find_errors(create_table(ws, [ufun[w] for w in ws]), gt_comparisons, tolerance=0.0)
        if len(errors) > 0:
            print(f'failed on GT with GT comparisons: {ufun}\n\n{[(e[1], gt_comparisons[e[0], :]) for e in errors]}')
            return

        table = create_f0(noisy_comparisons)
        if use_their_map:
            learned_ufun, result, ier, mesg = fsolve(dsdf, table[:, 1],
                                                     args=(noisy_comparisons, table[:, 0], sigma_learner)
                                                     , full_output=True)
            gp = create_gp_from_table(ws, learned_ufun)

            # print(dsdf(learned_ufun, data, table[:, 0], sigma_learner))
            successful = ier == 1
            result['message'] = mesg
        else:
            if use_our_map:
                result = minimize(max_apriori, x0=table[:, 1], args=(noisy_comparisons, table[:, 0], sigma_learner)
                                  , method='Nelder-Mead'
                                  , tol=tol, callback=None
                                  , options=optimizer_options_nm
                                  )
            else:
                result = minimize(likelihood, x0=table[:, 1], args=(noisy_comparisons, table[:, 0], sigma_learner)
                                  , method='Newton-CG'  # 'Nelder-Mead'
                                  , jac=jacobian, hess=hessian
                                  , tol=tol, callback=None
                                  , options=optimizer_options
                                  )
            successful = result.success
            learned_ufun = result.x
        if successful:
            table[:, 1] = learned_ufun
            errors = find_errors(table, noisy_comparisons, tolerance=1e-3)
            if len(errors) > 0:
                # print(f'\tfailed on :{ufun}\n got {learned_ufun}\n\n{[(e[1], noisy_comparisons[e[0], :]) for e in errors]}')
                print(f'\t failed {len(errors) / len(noisy_comparisons):0.03%} of comparisons')
                avg_errs += len(errors) / len(noisy_comparisons)
                n_failed += 1
            if do_gp:
                gp = create_gp(kernel=kernel, adaptive_sigma=sigma_learner if learn_sigma else 0.0)
                gp = fit_gp(gp, table)
                learned_gp = gp.predict(np.arange(n_real_outcomes).reshape(-1, 1))
                errors = find_errors(create_table(ws, [learned_gp[w] for w in ws]), noisy_comparisons, tolerance=1e-3)
                if len(errors) > 0:
                    # print(f'\tlearned gp has errors:\nGT: {ufun}\nGP: {learned_gp}\n\n'
                    #      f'\t{[(e[1], noisy_comparisons[e[0], :]) for e in errors]}')
                    print(f'\t GP failed {len(errors) / len(noisy_comparisons):0.03%} of comparisons')
                    n_failed_gp += 1
        else:
            print(f'\t{"Optimization" if not use_their_map else "Root finding"} failed!! :-(')
            n_failed_optimization += 1
            continue
            print(f'\tGT: {ufun}')
            print(f'\t{result}')
            table[:, 1] = learned_ufun
            errors = find_errors(table, noisy_comparisons, tolerance=1e-3)
            if len(errors) > 0:
                print(f'\tSome Errors on :{ufun}\n got {learned_ufun}\n\n'
                      f'{[(e[1], noisy_comparisons[e[0], :]) for e in errors]}')
            else:
                print('\tNO ERRORS :-)')

        print(f'{trial + 1:06} of {n_trials:06} completed [n={n_real_outcomes}]', flush=True, end='')
        if use_their_map:
            print(
                f' Failures: {n_failed / (trial + 1):0.03%}, RF Failures: {n_failed_optimization / (trial + 1):0.03%}',
                end='')
        else:
            print(f' Failures: {n_failed / (trial + 1):0.03%}'
                  f', GP failures: {n_failed_gp / (trial + 1):0.03%}, OP Failures: '
                  f'{n_failed_optimization / (trial + 1):0.03%} ', end='')
        print('', end='\n', flush=True)

    print(f'\n\nAverage Failure Rate: {avg_errs / len(noisy_comparisons)}')


# def main():
#     sigma = 1.0
#     data = np.array([
#         [0, 1, 1],
#         [1, 2, 1],
#         [2, 3, 1],
#     ])
#     for _ in range(1000):
#         table = create_f0(data)
#
#         result = minimize(likelihood, x0=table[:, 1], args=(data, table[:, 0], sigma)
#                           , method='Nelder-Mead'
#                           , tol=None, callback=None
#                           , options={'maxiter': None, 'maxfev': None, 'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False})
#         #print(result)
#         if result.success:
#             assert result.x[0] >= result.x[1] >= result.x[2] >= result.x[3]
#             table[:, 1] = result.x
#             #print(table)
#             return
#         #print(f'Failed to optimize')


if __name__ == '__main__':
    main()
