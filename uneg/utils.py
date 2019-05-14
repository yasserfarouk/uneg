import os
import random

random.seed(0)
from math import sqrt, log
from typing import Tuple, List, Optional, Union

import numpy as np
import negmas

np.random.seed(0)
from scipy.optimize import minimize, fsolve
import scipy

scipy.random.seed(0)
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, Kernel

eps = 1e-12

__all__ = [
    "generate_genius_ufuns_1d",
    "generate_genius_ufuns",
    "generate_comparisons",
    "generate_random_ufun",
    "generate_random_ufuns",
    "generate_ranking",
    "create_f0",
    "create_gp",
    "create_gp_from_table",
    "find_errors",
    "fit_gp",
    "ranking",
    "partial_ranking",
    "partial"
]


def generate_genius_ufuns_1d(folder=os.path.expanduser("~/datasets/genius/1d"), max_n_outcomes=None) -> List[np.array]:
    ufuns = []
    folder = os.path.abspath(folder)
    for dir in sorted(os.listdir(folder)):
        full_name = os.path.join(folder, dir)
        if not os.path.isdir(full_name):
            continue
        if max_n_outcomes is not None and int(dir[:6]) > max_n_outcomes:
            continue
        mechanism, agent_info, issues = negmas.load_genius_domain_from_folder(
            folder_name=full_name
        )
        for negotiator in agent_info:
            ufuns.append(np.asarray(list(negotiator["ufun"].mapping.values())))
    return ufuns


def generate_genius_ufuns(folder=os.path.expanduser("~/datasets/genius/nd")
                          , max_n_outcomes=None) -> Tuple[List[np.array], List[str], List[List[negmas.Issue]]]:
    folder = os.path.abspath(folder)
    names, issue_lists, ufuns = [], [], []
    for dir in sorted(os.listdir(folder)):
        full_name = os.path.join(folder, dir)
        if not os.path.isdir(full_name):
            continue
        if max_n_outcomes is not None and int(dir[:6]) > max_n_outcomes:
            continue
        mechanism, agent_info, issues = negmas.load_genius_domain_from_folder(
            folder_name=full_name, ignore_discount=True, ignore_reserved=True, force_numeric=True
        )
        for negotiator in agent_info:
            ufuns.append(negotiator["ufun"])
            names.append(negotiator["ufun_name"])
            issue_list = mechanism.issues
            issue_lists.append(issue_list)
    return ufuns, names, issue_lists


def generate_random_ufun(n_outcomes: int) -> np.array:
    """
    Generates a random ufun as an n_outcomes vector

    Args:
        n_outcomes:

    """
    return np.random.uniform(size=n_outcomes)


def generate_random_ufuns(n_outcomes, n_trials) -> List[np.array]:
    return [generate_random_ufun(n_outcomes) for _ in range(n_trials)]


def generate_comparisons(
    ufun: np.array, n: Optional[int], sigma: float
) -> Tuple[np.array, np.array]:
    """
    Generate comparisons given a ufun

    Args:
        ufun: the utility function as a n_outcomes vector
        n: number of comparisons. If None, ALL comparisons will be added
        sigma: the std. dev. of the noise to be added to ufun

    Returns:
        Two lists of comparisons. The first is noisy and the second is GT
    """
    ufun = np.asarray(ufun, dtype=float).flatten().copy()
    n_outcomes = len(ufun)

    noisy_comparisons, gt_comparisons = [], []

    def relation(f0_, f1_):
        return 1 if f0_ > f1_ else 0 if f0_ == f1_ else -1

    def add_comparison(w0, w1):
        f0 = ufun[w0] + (np.random.randn(1))[0] * sigma
        f1 = ufun[w1] + (np.random.randn(1))[0] * sigma
        c = relation(f0, f1)
        noisy_comparisons.append([w0, w1, c])
        c = relation(ufun[w0], ufun[w1])
        gt_comparisons.append([w0, w1, c])

    if n is None:
        for w0 in range(n_outcomes):
            for w1 in range(w0 + 1, n_outcomes):
                add_comparison(w0, w1)

    else:
        for _ in range(n):
            w0, w1 = (
                random.randint(0, n_outcomes - 1),
                random.randint(0, n_outcomes - 1),
            )
            while w0 == w1:
                w0, w1 = (
                    random.randint(0, n_outcomes - 1),
                    random.randint(0, n_outcomes - 1),
                )
            add_comparison(w0, w1)
    return np.array(noisy_comparisons), np.array(gt_comparisons)


def extract_ws(comparisons: np.array) -> np.array:
    return np.unique(np.vstack((comparisons[:, 0], comparisons[:, 1]))).transpose()


def create_f0(comparisons: np.array) -> np.array:
    ws = extract_ws(comparisons)
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


def likelihood(
    ys: np.array, data: np.array, ws: np.array, sigma: float, epsilon: float = eps
) -> float:
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

    l = 0.0  # likelihood

    f = dict(zip(ws, ys))

    if sigma == 0.0:
        for w0, w1, relation in data:
            w0, w1 = correct_order(w0, w1, relation)
            if w0 is None:
                continue
            elif f[w0] > f[w1]:
                l -= 1.0
        return l

    for w0, w1, relation in data:
        w0, w1 = correct_order(w0, w1, relation)
        if w0 is None:
            continue
        phi = norm.cdf(z(f[w0], f[w1], sigma))
        if phi < epsilon:
            continue
        l -= log(phi)
    return l


def jacobian(
    ys: np.array, data: np.array, ws: np.array, sigma: float, epsilon: float = eps
) -> np.array:
    """First derivative of the likelihood"""
    jac = np.zeros(len(ws))

    f = dict(zip(ws, ys))
    indx = dict(zip(ws, range(len(ws))))

    if sigma == 0.0:
        d = 0.05
        for w in ws:
            delta = np.zeros_like(ys)
            delta[indx[w]] = np.random.randn(1)[0] * d
            wsminus = ws - delta
            wsplus = ws + delta
            jac[indx[w]] = (
                likelihood(wsplus, data, ws, sigma, epsilon)
                - likelihood(wsminus, data, ws, sigma, epsilon)
            ) / (2 * d)
        return jac

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


def hessian(
    ys: np.array,
    data: np.array,
    ws: np.array,
    sigma: float,
    epsilon: float = eps,
    delta: float = eps,
) -> np.array:
    """Second derivative of the likelihood"""
    h = np.eye(len(ws)) * delta

    f = dict(zip(ws, ys))
    indx = dict(zip(ws, range(len(ws))))

    s2sigma = sqrt(2) * sigma * sigma

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


def find_errors(
    u: np.array, comparisons: np.array, tolerance: float = 0.0
) -> List[Tuple[int, float]]:
    """
    Tests that the result is compatible with ufun given comparisons in data

    Args:
        u: A table with w in column 1 and ufun value in column 2
        comparisons: Should be generated with no noise
        tolerance: tolerance

    Returns:

    """

    errors = []
    f = dict(zip(u[:, 0], u[:, 1]))
    for i, (w0, w1, c) in enumerate(comparisons):
        if c < 0 and (f[w0] - f[w1]) > tolerance:
            errors.append((i, abs(f[w0] - f[w1])))
        elif c > 0 and (f[w0] - f[w1]) < -tolerance:
            errors.append((i, abs(f[w0] - f[w1])))
    return errors


def create_gp(
    kernel=None, adaptive_sigma=1.0, normalize_y=False
) -> GaussianProcessRegressor:
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


def fit_gp(
    gp: Optional[GaussianProcessRegressor], xy: np.array
) -> GaussianProcessRegressor:
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


def max_apriori(
    ys: np.array, data: np.array, ws: np.array, sigma: float, epsilon: float = eps
) -> float:
    gp = create_gp_from_table(ws, ys)
    l = likelihood(ys, data, ws, sigma, epsilon) + prior(gp, ws)
    return l


def create_gp_from_table(
    ws: np.array, ys: np.array, *, kernel: Kernel = None
) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(optimizer=None, kernel=kernel)
    ys[np.abs(ys) < eps] = 0.0
    gp.fit(ws.reshape(-1, 1), ys.reshape(-1, 1))
    return gp


def dsdf(
    ys: np.array, data: np.array, ws: np.array, sigma: float, epsilon: float = eps
) -> np.array:
    gp = create_gp_from_table(ws, ys)
    # ys = prediction(gp, ws.reshape(-1, 1)).flatten()
    j = jacobian(ys, data, ws, sigma, epsilon)
    x = sigma_inv(gp, ws) @ ys
    return j + x


def generate_ranking(
    ufun: Union[negmas.UtilityFunction, List[float]],
    outcomes: List[negmas.Outcome],
    ascending=False,
    uniform_noise: float = 0.0,
    white_noise: float = 0.0,
    fraction=1.0,
) -> List[negmas.Outcome]:
    """
    Generates a partial/full ranking from the given ufun with noise

    Args:
        ufun:
        outcomes:
        ascending:
        uniform_noise: Range of a uniform noise to add before generating the ranking
        white_noise: Std. dev of white (gaussian) noise to add before generating the ranking
        fraction:

    Returns:

    """

    if isinstance(ufun, negmas.UtilityFunction):
        ufun = [float(ufun(outcome)) for outcome in outcomes]
    ufun = np.array(ufun)
    if uniform_noise > 0.0:
        mx, mn = ufun.max(), ufun.min()
        ufun = (
            (ufun - mn) / (mx - mn)
            + np.random.uniform(0.0, uniform_noise, ufun.shape)
            - 0.5 * uniform_noise
            + np.random.randn(*ufun.shape) * white_noise
        )
    r = ranking(ufun, outcomes, ascending=ascending)
    k = int(len(r) * fraction + 0.5)
    if k == len(r):
        return r
    return [r[_] for _ in random.sample(range(len(r)), k=k)]


def ranking(ufun, outcomes, ascending=False):
    """
    Gets a ranking of outcomes according to the ufun given

    Args:
        ufun:
        outcomes:
        ascending: Ascending or descending ranking

    Returns:

        - ties are broken randomly
    """
    outcomes = [
        outcome
        if isinstance(outcome, tuple)
        else tuple(v for v in outcome.values())
        if isinstance(outcome, dict)
        else outcome.astuple()
        for outcome in outcomes
    ]
    if isinstance(ufun, negmas.UtilityFunction):
        gt = {outcome: ufun[outcome] for outcome in outcomes}
    else:
        gt = dict(zip(outcomes, ufun))

    return [
        _[0]
        for _ in sorted(
            zip(gt.keys(), gt.values()), key=lambda x: x[1], reverse=not ascending
        )
    ]


def partial(ranked_outcomes, fraction, include_best_and_worst=True):
    """
    Creates a partial ranking by returning only the given fraction.

    Args:
        ranked_outcomes:
        fraction:
        include_best_and_worst: If true, the best and worst outcomes are always included
    """
    n = len(ranked_outcomes)
    if fraction < 0.0:
        raise ValueError(f"Negative faction {fraction}")
    if fraction < 1 / n:
        return []
    if include_best_and_worst:
        n_new = int(fraction * n + 0.5) - 2
        if n_new < 0:
            raise ValueError(f"cannot include best and worst outcomes (n={n})")
        return (
            [ranked_outcomes[0]]
            + partial(
                ranked_outcomes[1:-1],
                fraction=n_new / (n - 2),
                include_best_and_worst=False,
            )
            + [ranked_outcomes[-1]]
        )
    return [
        ranked_outcomes[_]
        for _ in sorted(random.sample(range(0, n), int(fraction * n + 0.5)))
    ]


def partial_ranking(
    ufun, outcomes, fraction, ascending=False, include_best_and_worst=True
):
    """
    Creates a partial ranking by returning only the given fraction.

    Args:
        outcomes:
        fraction:
        ascending: Rank from lowest to highest utility
        include_best_and_worst: If true, the best and worst outcomes are always included
    """
    n = len(outcomes)
    if fraction < 0.0:
        raise ValueError(f"Negative faction {fraction}")
    if fraction < 1 / n:
        return []
    if include_best_and_worst:
        return partial(
            ranking(ufun, outcomes, ascending),
            fraction=fraction,
            include_best_and_worst=include_best_and_worst,
        )
    return list(
        _[0]
        for _ in sorted(
            (
                (outcomes[_], ufun[outcomes[_]])
                for _ in random.sample(range(0, n), int(fraction * n + 0.5))
            ),
            key=lambda x: x[1],
        )
    )
