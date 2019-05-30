import copy
import functools
import itertools
from abc import ABC, abstractmethod
from math import sqrt
from typing import List, Tuple, Iterable, Optional, Union, Callable, Any, Dict
import numpy as np
from negmas import (
    Outcome,
    UtilityFunction,
    UtilityValue,
    Issue,
    OutcomeType,
    MappingUtilityFunction,
)
from negmas.helpers import Distribution
from negmas.outcomes import is_outcome
from scipy.optimize import fsolve, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from uneg.utils import (
    dsdf,
    create_f0,
    create_gp_from_table,
    extract_ws,
    create_table,
    max_apriori,
    likelihood,
    jacobian,
    hessian,
    create_gp,
    fit_gp,
    generate_comparisons,
    find_errors,
    ranking,
)

__all__ = [
    "ComparisonsUfun",
    "IndexGPUfun",
    "ChuIndexGPUfun",
    "CompressRegressIndexUfun",
    "RankingLAPUfunLearner",
    "IssueFunction",
    "linear",
    "quadratic",
    "polynomial",
    "unbiased_polynomial",
    "RankingUfunLearner",
    "RankingProjectLAPUfunLearner",
    "distributions_from_vals",
]


class ComparisonsUfun(UtilityFunction):
    """A ufun learned from a list of comparisons.

    Assumptions:

        - The outcomes are countable and known in advance (passed as outcomes to the constructor)
        - The comparisons are a list/nparray with the following record (row) format:

            - outcome1 index, outcome2 index, relation

        - Possible values of relation are:

            * 1  => utility(outcome 1) > utility(outcome 2)
            * 0  => utility(outcome 1) = utility(outcome 2)
            * -1  => utility(outcome 1) < utility(outcome 2)

    """

    def __init__(
        self,
        outcomes: Union[int, List[Outcome]],
        comparisons: List[Tuple[int, int, int]] = None,
        accept_outcomes_or_indices=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        else:
            outcomes = list(outcomes)
        self.outcomes = outcomes
        self.n_outcomes = len(outcomes)
        self.index = dict(zip(outcomes, range(self.n_outcomes)))
        self.accept_outcomes_or_indices = accept_outcomes_or_indices
        self.comparisons = []
        if comparisons is not None:
            self.fit_(comparisons, incremental=False)

    def fit_(self, comparisons: List[Tuple[int, int, int]], incremental=False):
        if self.accept_outcomes_or_indices:
            for i, comparison in enumerate(comparisons):
                comparison = list(comparison)
                for j in range(2):
                    if is_outcome(comparison[j]):
                        comparison[j] = self.index[comparison[j]]
                comparisons[i] = tuple(comparison)
            comparisons = list(
                set(
                    [
                        (o1, o2, r) if r >= 0 else (o2, o1, -r)
                        for o1, o2, r in comparisons
                    ]
                )
            )
            if not incremental:
                self.comparisons = list(set(self.comparisons + comparisons))
            else:
                self.comparisons = comparisons
            self.comparisons = np.asarray(self.comparisons)
        return self.fit(self.comparisons, incremental=incremental)

    @abstractmethod
    def fit(self, comparisons: List[Tuple[int, int, int]], incremental=False) -> bool:
        pass

    @abstractmethod
    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        pass

    def predict(self, outcomes: Iterable[Outcome]) -> List[UtilityValue]:
        return [self(outcome) for outcome in outcomes]

    def evaluate(
        self,
        comparisons: List[Tuple[Union[int, Outcome], Union[int, Outcome], int]] = None,
        gt: List[float] = None,
        max_comparisons: int = 10000,
        tolerance: float = 1e-3,
        return_errors=False,
    ) -> Union[float, Tuple[float, List[Tuple[float, Tuple[int, int, int]]]]]:
        """
        Evaluates the quality of the ufun (that must be fitted) against either a list of comparisons or the ground truth

        Args:
            comparisons: List of comparisons in the same format as the input to fit
            gt: List of ufun values for the outcomes (in order)
            max_comparisons: The maximum number of comparisons between the ufun learned and gt (if gt is given)
            tolerance: The tolerance in comparisons when GT is given
            return_errors: If True, return a list of the errors that happened

        Returns:
            Average Failure Rate: A value between 0 and 1 giving the fraction of comparisons tried that had different
                                  results in the learned ufun and the ground truth.

        Remarks:

            - The ufun (and gt if given) are normalized before trying comparisons
            - If return_errors is True, the second return value is a list of records with the following elements:

              error (float) == abs(f(w0), f(w1)), Tuple[w0, w1, relation], where f is the utility function learned
        """

        def normalize(ufun):
            ufun = np.asarray(ufun)
            range_ = ufun.max() - ufun.min()
            if abs(range_) < 1e-5:
                return ufun
            return (ufun - ufun.min()) / range_

        learned_ufun = normalize(self.predict(self.outcomes))
        if gt is not None:
            gt = normalize(gt)
            c, _ = generate_comparisons(gt, max_comparisons, sigma=0.0)
            if comparisons is not None:
                comparisons = list(np.asarray(comparisons).tolist()) + list(c.tolist())
            else:
                comparisons = list(c.tolist())

        table = create_table(list(range(self.n_outcomes)), learned_ufun)
        errors = find_errors(table, comparisons, tolerance=tolerance)
        if return_errors:
            err_report = [(e[1], comparisons[e[0], :]) for e in errors]
            return len(errors) / len(comparisons), err_report
        return len(errors) / len(comparisons)

    def xml(self, issues: List[Issue]):
        return self.__class__.__name__ + " cannot be saved as xml (or read from xml)"


class IndexGPUfun(ComparisonsUfun):
    """A comparison utility function that is based on a GP trained on outcome indices"""

    def __init__(
        self,
        outcomes: List[Outcome],
        comparisons: List[Tuple[int, int, int]] = None,
        gp: GaussianProcessRegressor = None,
        sigma: float = 0.1,
        kernel: Optional[Kernel] = None,
    ):
        self.gp = gp
        self.sigma = sigma
        self.result_ = None
        self.fitted = False
        self.kernel = kernel
        super().__init__(outcomes=outcomes, comparisons=comparisons)

    def predict(self, outcomes: Iterable[Outcome]) -> List[UtilityValue]:
        indices = np.array([self.index[outcome] for outcome in outcomes]).reshape(-1, 1)
        if self.gp is None or not self.fitted:
            raise ValueError(
                f"Trying to predict before fitting or after a failed fitting trial"
            )
        return self.gp.predict(indices).flatten().tolist()

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        return self.predict([offer])[0]


class ChuIndexGPUfun(IndexGPUfun):
    """A comparison utility function based on Chu and Ghahramani's 2005 paper"""

    def fit(self, comparisons: List[Tuple[int, int, int]], incremental=False):
        if incremental:
            raise ValueError(f"ChuIndexGPUfun does not support incremental learning")
        ws = extract_ws(comparisons)
        ys = np.random.uniform(0.0, 1.0, *ws.shape)  # Amy made me write it this way :-(
        table = create_table(ws, ys)
        learned_ufun, self.result_, ier, mesg = fsolve(
            dsdf,
            table[:, 1],
            args=(comparisons, table[:, 0], self.sigma),
            full_output=True,
        )
        self.gp = create_gp_from_table(ws, learned_ufun, kernel=self.kernel)
        self.fitted = ier == 1
        self.result_["message"] = mesg
        return self.fitted


class CompressRegressIndexUfun(IndexGPUfun):
    """A comparison utility function based on our MIS-reading of Chu and Ghahramani's 2005 paper :-)"""

    def __init__(
        self,
        use_map: bool = False,
        tol: Optional[float] = 1e-3,
        compress_only: bool = False,
        optimize_sigma: bool = True,
        *args,
        **kwargs,
    ):
        self.use_map = use_map
        self.tol = tol
        self.compress_only = compress_only
        self.optimize_sigma = optimize_sigma
        super().__init__(*args, **kwargs)

    def fit(self, comparisons: List[Tuple[int, int, int]], incremental=False):
        ws = extract_ws(comparisons)
        ys = np.random.uniform(0.0, 1.0, *ws.shape)  # Amy made me write it this way :-(
        table = create_table(ws, ys)

        optimizer_options_nm = {
            "maxiter": None,
            "maxfev": None,
            "disp": False,
            "return_all": False,
            "initial_simplex": None,
            "xatol": 1e-6,
            "fatol": 1e-6,
            "adaptive": True,
        }
        optimizer_options = {
            "xtol": self.tol,
            "eps": 1.4901161193847656e-08,
            "maxiter": 1000,
            "disp": False,
            "return_all": True,
        }

        if self.use_map:
            self.result_ = minimize(
                max_apriori,
                x0=table[:, 1],
                args=(comparisons, table[:, 0], self.sigma),
                method="Nelder-Mead",
                tol=self.tol,
                options=optimizer_options_nm,
            )
        else:
            if self.sigma == 0.0:
                self.result_ = minimize(
                    likelihood,
                    x0=table[:, 1],
                    args=(comparisons, table[:, 0], self.sigma),
                    method="Nelder-Mead",
                    tol=self.tol,
                    options=optimizer_options_nm,
                )
            else:
                self.result_ = minimize(
                    likelihood,
                    x0=table[:, 1],
                    args=(comparisons, table[:, 0], self.sigma),
                    method="Newton-CG",  # 'Nelder-Mead'
                    jac=jacobian,
                    hess=hessian,
                    tol=self.tol,
                    options=optimizer_options,
                )
        self.fitted = self.result_.success
        learned_ufun = self.result_.x
        if not self.fitted:
            return False
        if self.compress_only:
            self.gp = create_gp_from_table(ws, learned_ufun, kernel=self.kernel)
            return self.fitted

        if not incremental or self.gp is None:
            self.gp = create_gp(
                kernel=self.kernel,
                adaptive_sigma=self.sigma if self.optimize_sigma else 0.0,
            )
        table[:, 1] = learned_ufun
        self.gp = fit_gp(self.gp, table)
        return self.fitted


class RankingUfunLearner(UtilityFunction):
    """A utility function learnable from a full ordering of a subset of the outcomes"""

    def __init__(
        self,
        *args,
        issues: Optional[List[Issue]],
        outcomes: List[Outcome] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if issues is None and outcomes is None:
            raise ValueError("Neither issues nor outcomes was given")
        self.issues = issues
        if outcomes is not None:
            self.outcomes = [
                outcome.values()
                if isinstance(outcome, dict)
                else outcome
                if isinstance(outcome, tuple)
                else outcome.astuple()
                for outcome in outcomes
            ]
        else:
            self.outcomes = Issue.enumerate(issues=issues, astype=tuple)
        self.issue_names = [issue.name for issue in issues]

    def ranking_error(
        self,
        gt: Union[List[float], Dict[Outcome, float], UtilityFunction],
        tolerance: float = 0.0,
    ) -> float:
        """
        Evaluates the utility function against gt counting differences in ranking only

        Args:

            gt: Ground truth ufun
            tolerance: a small tolerance when finding if the ranking is respected

        Returns:

            A value between zero and one indicating the ranking error
        """
        if isinstance(gt, list):
            gt = dict(zip(self.outcomes, gt))
        if not isinstance(gt, UtilityFunction):
            gt = MappingUtilityFunction(gt)
        rank = ranking(self, self.outcomes)
        return sum(
            int(float(gt(w1)) + tolerance < float(gt(w2)))
            for w1, w2 in zip(rank[:-1], rank[1:])
        ) / (len(self.outcomes) - 1)

    def value_error(
        self, gt: Union[List[float], Dict[Outcome, float], UtilityFunction]
    ) -> Tuple[float, float]:
        """
        Finds the error in ufun values against gt (after normalization from 0-1)

        Args:
            gt: Ground truth

        Returns:

            Mean and std. dev of errors

        """
        if isinstance(gt, list):
            gt = dict(zip(self.outcomes, gt))
        if not isinstance(gt, UtilityFunction):
            gt = MappingUtilityFunction(gt)
        data = np.array(
            [[float(gt(outcome)), float(self(outcome))] for outcome in self.outcomes]
        )
        mx, mn = data.max(axis=0), data.min(axis=0)
        data = (data - mn) / (mx - mn)
        err = (data[:, 0] - data[:, 1]) ** 2
        return float(np.sqrt(np.mean(err))), float(np.std(err))

    @abstractmethod
    def fit(self, ranking_list: List[Outcome]) -> UtilityFunction:
        """
        Fits the ufun to given ranking
        
        Args:
            ranking_list: 

        Returns:

        """


IssueFunction = Callable[[Any, List[float]], float]


def linear(x, theta):
    return theta[0] + x * theta[1]


def quadratic(x, theta):
    return theta[0] + x * theta[1] + x * x * theta[2]


def polynomial(x, theta):
    return sum(alpha * x ** i for i, alpha in enumerate(theta))


def unbiased_polynomial(x, theta):
    return sum(alpha * x ** (i + 1) for i, alpha in enumerate(theta))


def _ufun_objective(
    theta: List[float],
    ranking: List[Outcome],
    fs: List[IssueFunction],
    n_params: List[int],
    tolerance=0.0,
    kind: str = "errors",
):
    """

    Args:
        theta:
        ranking:
        fs:
        n_params:
        tolerance:
        kind: The kind of quality to measure:

            - errors : the number of errors in the ranking
            - error_sizes: the sum of error sizes
            - differences: The sum of differences

    Returns:

    """
    m = len(fs)
    param_ranges = [(0, 0)] * m
    nxt_param = 0
    for i, n_p in enumerate(n_params):
        param_ranges[i] = (nxt_param, nxt_param + n_p)
        nxt_param += n_p
    fvals = [
        sum(
            f(outcome[j], theta[range_[0] : range_[1]])
            for j, (f, range_) in enumerate(zip(fs, param_ranges))
        )
        for outcome in ranking
    ]
    if kind == "errors":
        return sum(
            int(f1 + tolerance < f2) for f1, f2 in zip(fvals[:-1], fvals[1:])
        ) / (len(fvals) - 1)
    elif kind == "error_sums":
        rng = max(fvals) - min(fvals)
        return sum(
            (f2 - f1) / rng if f1 + tolerance < f2 else 0.0
            for f1, f2 in zip(fvals[:-1], fvals[1:])
        )
    elif kind == "differences":
        rng = max(fvals) - min(fvals)
        return sum(
            (f2 - f1) / rng if f1 < f2 else 0.1 * (f2 - f1) / rng
            for f1, f2 in zip(fvals[:-1], fvals[1:])
        )
    elif kind == "constraint":
        return 1
    else:
        raise ValueError(
            f"Unknown kind: {kind}. Supported kinds are: errors, error_sums, differences"
        )


def _ufun_constraint(
    theta: List[float],
    o1: Outcome,
    o2: Outcome,
    fs: List[IssueFunction],
    n_params: List[int],
    tolerance=0.0,
    kind: str = "constraints",
):
    """

    Args:
        theta:
        o1:
        o2:
        fs:
        n_params:
        tolerance:
        kind: The kind of quality to measure:

            - errors : the number of errors in the ranking
            - error_sizes: the sum of error sizes
            - differences: The sum of differences

    Returns:

    """
    ranking = (o1, o2)
    m = len(fs)
    param_ranges = [(0, 0)] * m
    nxt_param = 0
    for i, n_p in enumerate(n_params):
        param_ranges[i] = (nxt_param, nxt_param + n_p)
        nxt_param += n_p
    fvals = [
        sum(
            f(outcome[j], theta[range_[0] : range_[1]])
            for j, (f, range_) in enumerate(zip(fs, param_ranges))
        )
        for outcome in ranking
    ]
    if kind in ("errors", "error_sums", "differences"):
        return 1
    elif kind == "constraints":
        return -1 if fvals[0] + tolerance < fvals[1] else 1
    else:
        raise ValueError(
            f"Unknown kind: {kind}. Supported kinds are: errors, error_sums, differences"
        )


class RankingLAPUfunLearner(RankingUfunLearner, UtilityFunction):
    """Linearly Aggregated Polynomials"""

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        if isinstance(offer, OutcomeType):
            offer = offer.asdict()
        if isinstance(offer, dict):
            offer = (offer[n] for n in self.issue_names)
        return self.uvals.get(offer, None)

    def xml(self, issues: List[Issue]) -> str:
        return "Cannot convert to xml"

    def __init__(
        self,
        *args,
        issues: Optional[List[Issue]] = None,
        outcomes: List[Outcome] = None,
        degree: Union[int, List[int]] = 2,
        kind="constraints",
        tolerance: float = 0.0,
        learn_distributions: bool = False,
        **kwargs,
    ):
        super().__init__(*args, issues=issues, outcomes=outcomes, **kwargs)

        self.fs = [unbiased_polynomial] * len(issues)
        self.n_params = [degree] * len(issues) if isinstance(degree, int) else degree
        self.kind = kind
        self.tolerance = tolerance
        nxt_param = 0
        param_ranges = [(0, 0)] * len(issues)
        for i, n_p in enumerate(self.n_params):
            param_ranges[i] = (nxt_param, nxt_param + n_p)
            nxt_param += n_p
        self.param_ranges = param_ranges
        self.result_ = None
        self.learn_distributions = learn_distributions
        self.ranking_list = []
        self.fitted = False
        self.theta = np.random.randn(sum(self.n_params)) - 0.5
        uvals = [
            sum(
                f(offer[i], self.theta[r[0] : r[1]])
                for i, (f, r) in enumerate(zip(self.fs, self.param_ranges))
            )
            for offer in self.outcomes
        ]
        self.uvals = dict(zip(self.outcomes, uvals))

    def fit(self, ranking_list: List[Outcome]):
        self.ranking_list += ranking_list
        ranking_list = self.ranking_list
        optimizer_options = {
            "disp": None,
            "maxcor": 10,
            "ftol": 2.220446049250313e-09,
            "gtol": 1e-05,
            "eps": 1e-08,
            "maxfun": 15000,
            "maxiter": 15000,
            "iprint": -1,
            "maxls": 20,
        }
        if self.kind == "constraints":
            constraints = [
                {
                    "type": "ineq",
                    "fun": _ufun_constraint,
                    "args": (o1, o2, self.fs, self.n_params, self.tolerance, self.kind),
                }
                for o1, o2 in zip(ranking_list[:-1], ranking_list[1:])
            ]
            self.result_ = minimize(
                lambda x: (sqrt(sum([_ * _ for _ in x]))) ** 2,
                x0=self.theta,
                method="COBYLA",  # "SLSQP"
                bounds=[(-1.0, 1)] * len(self.theta),
                constraints=constraints,
                options={
                    "rhobeg": 1.0,
                    "maxiter": 1000,
                    "disp": False,
                    "catol": 0.0002,
                },
                # options={
                #     "func": None,
                #     "maxiter": 100,
                #     "ftol": 1e-06,
                #     "iprint": 1,
                #     "disp": False,
                #     "eps": 1.4901161193847656e-08,
                # },
            )
        else:
            self.result_ = minimize(
                _ufun_objective,
                x0=self.theta,
                args=(ranking_list, self.fs, self.n_params, self.tolerance, self.kind),
                method="L-BFGS-B",
                options=optimizer_options,
                bounds=[(-1.0, 1)] * len(self.theta),
            )
        if self.result_.success:
            self.theta = self.result_.x
            uvals = [
                sum(
                    f(offer[i], self.theta[r[0] : r[1]])
                    for i, (f, r) in enumerate(zip(self.fs, self.param_ranges))
                )
                for offer in self.outcomes
            ]
            if self.learn_distributions:
                self.uvals = distributions_from_vals(uvals, self.outcomes)
            else:
                self.uvals = dict(zip(self.outcomes, uvals))
            self.fitted = True
            return True
        return False


def distributions_from_vals(uvals, outcomes):
    """
    Creates uniform distributions around given values

    Args:
        uvals:
        outcomes:

    Returns:

    """
    ordered_vals = list(sorted(zip(outcomes, uvals), key=lambda x: x[1]))
    ordered_vals = (
        [(None, ordered_vals[0][1] - (ordered_vals[1][1] - ordered_vals[0][1]))]
        + ordered_vals
        + [(None, ordered_vals[-1][1] + (ordered_vals[-1][1] - ordered_vals[-2][1]))]
    )
    ranges = [
        (o2[1] / 2 + (o1[1] + o3[1]) / 4, (o3[1] - o1[1]) / 4)
        for o1, o2, o3 in zip(ordered_vals[:-2], ordered_vals[1:-1], ordered_vals[2:])
    ]
    uvals = [Distribution(dtype="uniform", loc=r[0], scale=r[1]) for r in ranges]
    return dict(zip((_[0] for _ in ordered_vals[1:-1]), uvals))


class RankingProjectLAPUfunLearner(RankingUfunLearner, UtilityFunction):
    """Linearly Aggregated Polynomials after projection on issue space"""

    def __call__(self, offer: Outcome) -> Optional[UtilityValue]:
        if isinstance(offer, OutcomeType):
            offer = offer.asdict()
        if isinstance(offer, dict):
            offer = (offer[n] for n in self.issue_names)
        return self.uvals.get(offer, None)

    def xml(self, issues: List[Issue]) -> str:
        return "Cannot convert to xml"

    def __init__(
        self,
        *args,
        issues: Optional[List[Issue]] = None,
        outcomes: List[Outcome] = None,
        degree: Union[int, List[int]] = 2,
        kind="error_sums",
        tolerance: float = 0.0,
        learn_distributions: bool = False,
        **kwargs,
    ):
        super().__init__(*args, issues=issues, outcomes=outcomes, **kwargs)

        self.n_params = [degree] * len(issues) if isinstance(degree, int) else degree
        self.kind = kind
        self.tolerance = tolerance
        self.result_ = None
        self.learn_distributions = learn_distributions
        self.ranking_list = []
        self._fitted = [False] * len(issues)
        self.fitted = False
        self.thetas = [np.random.randn(n) - 0.5 for n in self.n_params]
        self.fs = [unbiased_polynomial] * len(issues)
        uvals = [
            sum(f(v, theta) for v, f, theta in zip(offer, self.fs, self.thetas))
            for offer in self.outcomes
        ]
        self.uvals = dict(zip(self.outcomes, uvals))

    def fit(self, ranking_list: List[Outcome]):
        self.ranking_list += ranking_list
        ranking_list = self.ranking_list
        optimizer_options = {
            "disp": None,
            "maxcor": 10,
            "ftol": 2.220446049250313e-09,
            "gtol": 1e-05,
            "eps": 1e-08,
            "maxfun": 15000,
            "maxiter": 15000,
            "iprint": -1,
            "maxls": 20,
        }
        self.result_ = []
        for i, (issue, f, theta) in enumerate(
            zip(self.issue_names, self.fs, self.thetas)
        ):
            n_params = len(theta)
            if self.kind == "constraints":
                constraints = [
                    {
                        "type": "ineq",
                        "fun": _ufun_constraint,
                        "args": (o1, o2, [f], [n_params], self.tolerance, self.kind),
                    }
                    for o1, o2 in zip(ranking_list[:-1], ranking_list[1:])
                ]
                result = minimize(
                    lambda x: (sqrt(sum([_ * _ for _ in x]))) ** 2,
                    x0=theta,
                    method="COBYLA",  # "SLSQP"
                    bounds=[(-1.0, 1)] * len(theta),
                    constraints=constraints,
                    options={
                        "rhobeg": 1.0,
                        "maxiter": 1000,
                        "disp": False,
                        "catol": 0.0002,
                    },
                )
            else:
                result = minimize(
                    _ufun_objective,
                    x0=theta,
                    args=(ranking_list, [f], [n_params], self.tolerance, self.kind),
                    method="L-BFGS-B",
                    options=optimizer_options,
                    bounds=[(-1.0, 1)] * len(theta),
                )
            self.result_.append(result)
        if any(_.success for _ in self.result_):
            for i, result in enumerate(self.result_):
                self.thetas[i] = result.x
                uvals = [
                    sum(f(v, theta) for v, f, theta in zip(offer, self.fs, self.thetas))
                    for offer in self.outcomes
                ]
                if self.learn_distributions:
                    self.uvals = distributions_from_vals(
                        uvals=uvals, outcomes=self.outcomes
                    )
                else:
                    self.uvals = dict(zip(self.outcomes, uvals))
                self._fitted[i] = True
        self.fitted = all(self._fitted)
        return self.fitted
