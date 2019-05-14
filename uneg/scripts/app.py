import math
import os
import random
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed, TimeoutError

from negmas import Issue
from negmas.helpers import add_records
from sklearn.gaussian_process.kernels import Matern

random.seed(0)
import numpy as np

np.random.seed(0)
import scipy
import click

scipy.random.seed(0)
from uneg.utils import *
from uneg.learners import ChuIndexGPUfun, CompressRegressIndexUfun
import numpy as np
import pandas as pd
from uneg.learners import RankingLAPUfunLearner
from uneg.utils import generate_genius_ufuns_1d, extract_ws, create_table


@click.group()
def cli():
    pass


@cli.command(help="Evaluates GP based ufun learner")
def gp(
        genius_ufuns=True,
        sigma_generator=0.0,
        n_trials=100,  # only used with random ufuns
        n_outcomes=100,  # interpreted as max_n_outcomes for genius ufuns
        n_comparisons_per_outcome=2,
        compress_only=True,
        kernel=None,
        learn_sigma=False,
        use_their_map=True,
        use_our_map=False,
        sigma_learner=0.0,
        tol=1e-3,
):
    if kernel is None:
        kernel = Matern()
    if use_their_map:
        compress_only = True

    testing_error, training_error, comparison_error = 0.0, 0.0, 0.0
    n_failed_optimization = 0

    if genius_ufuns:
        ufuns = generate_genius_ufuns_1d(max_n_outcomes=n_outcomes)
    else:
        ufuns = generate_random_ufuns(n_outcomes, n_trials)

    n_trials = len(ufuns)

    comparisons = [
        generate_comparisons(
            ufun=ufun, n=n_comparisons_per_outcome * len(ufun), sigma=sigma_generator
        )
        for ufun in ufuns
    ]

    for trial, (gt_ufun, (noisy_comparisons, gt_comparisons)) in enumerate(
            zip(ufuns, comparisons)
    ):
        n_real_outcomes = len(gt_ufun)
        ws = extract_ws(noisy_comparisons)
        errors = find_errors(
            create_table(ws, [gt_ufun[w] for w in ws]), noisy_comparisons, tolerance=0.0
        )
        if len(errors) > 0 and sigma_generator == 0.0:
            print(
                f"failed on GT :{gt_ufun}\n\n{[(e[1], noisy_comparisons[e[0], :]) for e in errors]}"
            )
            return
        errors = find_errors(
            create_table(ws, [gt_ufun[w] for w in ws]), gt_comparisons, tolerance=0.0
        )
        if len(errors) > 0:
            print(
                f"failed on GT with GT comparisons: {gt_ufun}\n\n{[(e[1], gt_comparisons[e[0], :]) for e in errors]}"
            )
            return

        if use_their_map:
            ufun = ChuIndexGPUfun(
                outcomes=[(_,) for _ in range(n_real_outcomes)],
                comparisons=noisy_comparisons,
                sigma=sigma_learner,
                kernel=kernel,
            )
        else:
            ufun = CompressRegressIndexUfun(
                outcomes=[(_,) for _ in range(n_real_outcomes)],
                comparisons=noisy_comparisons,
                sigma=sigma_learner,
                kernel=kernel,
                use_map=use_our_map,
                tol=tol,
                compress_only=compress_only,
                optimize_sigma=learn_sigma,
            )
        if ufun.fitted:
            comparison_error += ufun.evaluate(comparisons=gt_comparisons)
            training_error += ufun.evaluate(comparisons=noisy_comparisons)
            testing_error += ufun.evaluate(gt=gt_ufun)
        else:
            # print(
            #     f'\t{"Optimization" if not use_their_map else "Root finding"} failed!! :-('
            # )
            n_failed_optimization += 1
        print(
            f"{trial + 1:06} of {n_trials:06} completed [n={n_real_outcomes}]",
            flush=True,
            end="",
        )
        n_t = trial + 1 - n_failed_optimization
        if n_t == 0:
            continue
        print(
            f" Training Error: {training_error / n_t:0.03%}"
            f" GT Comparisons Error: {comparison_error / n_t:0.03%}"
            f" Testing Error: {testing_error / n_t:0.03%}"
            f", OP Failures: {n_failed_optimization / (trial + 1):0.03%} ",
            end="",
        )
        print("", end="\n", flush=True)


ufuns, issues = [], []


def evaluate_ranking(n_ufuns, name, method, degree, fraction, outcomes, u, w
                     , n_rankings_to_run, n_trials_per_ranking, results_file_name, i):
    gt, issue_list = ufuns[i], issues[i]
    results = []
    for _ in range(n_rankings_to_run):
        if fraction in (0.0, 1.0) and _ > 0:
            continue
        for __ in range(n_trials_per_ranking):
            ranking = generate_ranking(ufun=gt, outcomes=outcomes
                                       , uniform_noise=u, white_noise=w
                                       , fraction=fraction)
            if len(ranking) < 2:
                print(" Cancelled (too small fraction)")
                continue
            ufun = RankingLAPUfunLearner(issues=issue_list, outcomes=outcomes, kind=method
                                         , degree=degree)
            strt = time.perf_counter()
            ufun.fit(ranking)
            duration = time.perf_counter() - strt
            if ufun.fitted:
                rerror = ufun.ranking_error(gt)
                verror_mean, verror_std = ufun.value_error(gt)
            else:
                rerror = float('nan')
                verror_mean, verror_std = float('nan'), float('nan')

            results.append({
                "method": method,
                "degree": degree,
                "fitted": ufun.fitted,
                "name": name,
                "n_outcomes": len(outcomes),
                "fraction": fraction,
                "white_noise": w,
                "uniform_noise": u,
                "ranking_error": rerror,
                "value_error_mean": verror_mean,
                "value_error_std": verror_std,
                "duration": duration,
            })
            print(f"{i}/{n_ufuns}: {name}:{method}({degree}) "
                  f"(fraction:{fraction:0.02}, noise: {w}-{u}, {_}/{__})"
                  f" DONE (r-error: {rerror:4.02%}, v-error: {verror_mean:4.02%}"
                  f"[std.dev. {verror_std:4.02%}]) in {duration}ns", flush=True)
    return results


@cli.command(help="Evaluates a Rank Learner")
@click.option(
    "--outcomes",
    "-o",
    default=1000,
    help='The maximum allowed number of outcomes',
)
@click.option(
    "--fractions",
    "-f",
    default=(0.0, 1.0, 10),
    type=(float, float, int),
    help='Fractions as a tuple: start end count.',
)
@click.option(
    "--n-rankings-per-fraction",
    "--rankings"
    "-r",
    default=5,
    help='Number of rankings per fraction',
)
@click.option(
    "--n-trials-per-ranking",
    "--trials"
    "-t",
    default=1,
    help='Number of trials per ranking',
)
@click.option(
    "--white-noise",
    "--white",
    "--gaussian",
    "-w",
    "-g",
    default=(0.0, 0.0, 1),
    type=(float, float, int),
    help='White noise: start end number',
)
@click.option(
    "--uniform-noise",
    "--uniform",
    "-u",
    default=(0.0, 0.0, 1),
    type=(float, float, int),
    help='Uniform noise: start end number',
)
@click.option(
    "--degrees",
    "-d",
    default=(2, 3),
    type=(int, int),
    help='Degrees to try: min max',
)
@click.option(
    "--serial/--parallel",
    default=True,
    help="run serially or in parallel"
)
def rank(
        outcomes=1000,
        fractions=(0.0, 1.0, 10),
        n_rankings_per_fraction=5,
        n_trials_per_ranking=1,
        white_noise=None,
        uniform_noise=None,
        degrees=(2, 3),
        serial=True,
):
    global ufuns
    global issues
    if white_noise is None:
        white_noise = (0.0, 0.0, 1)
    if uniform_noise is None:
        uniform_noise = (0.0, 0.0, 1)
    ufuns, names, issues = generate_genius_ufuns(max_n_outcomes=outcomes)
    n_ufuns = len(ufuns)
    methods = ("errors", "error_sums", "differences")
    results_file_name = os.path.expanduser("~/code/projects/uneg/data/accuracy.csv")
    n_all = 0
    if serial:
        for i, (gt, name, issue_list) in enumerate(zip(ufuns, names, issues)):
            outcomes = Issue.enumerate(issue_list, astype=tuple)
            for fraction in np.linspace(*fractions):
                k = int(fraction * len(outcomes) + 0.5)
                if k < 2:
                    print(f"{i}/{len(ufuns)}: {name}: "
                          f"(fraction:{fraction:0.02} Cancelled (too small fraction)", flush=True)
                    continue
                n_rankings = int(math.factorial(len(outcomes)) / (math.factorial(len(outcomes) - k) * math.factorial(k)))
                n_rankings_to_run = min((n_rankings_per_fraction, n_rankings))
                for w in np.linspace(*white_noise):
                    for u in np.linspace(*uniform_noise):
                        for method in methods:
                            for degree in degrees:
                                    results = evaluate_ranking(n_ufuns, name, method, degree, fraction, outcomes, u, w
                                                 , n_rankings_to_run, n_trials_per_ranking, results_file_name, i)
                                    add_records(results_file_name, pd.DataFrame(data=results))
    else:
        executor = ProcessPoolExecutor(max_workers=None)
        future_results = []
        for i, (gt, name, issue_list) in enumerate(zip(ufuns, names, issues)):
            outcomes = Issue.enumerate(issue_list, astype=tuple)
            for fraction in np.linspace(*fractions):
                k = int(fraction * len(outcomes) + 0.5)
                if k < 2:
                    print(f"{i}/{len(ufuns)}: {name}: "
                          f"(fraction:{fraction:0.02} Cancelled (too small fraction)", flush=True)
                    continue
                n_rankings = int(math.factorial(len(outcomes)) / (math.factorial(len(outcomes) - k) * math.factorial(k)))
                n_rankings_to_run = min((n_rankings_per_fraction, n_rankings))
                for w in np.linspace(*white_noise):
                    for u in np.linspace(*uniform_noise):
                        for method in methods:
                            for degree in degrees:
                                future_results.append(
                                    executor.submit(
                                        evaluate_ranking,
                                        n_ufuns, name, method, degree, fraction, outcomes, u, w
                                        , n_rankings_to_run, n_trials_per_ranking, results_file_name, i,
                                    )
                                )
                                n_all += 1
        print(f"Submitted all processes ({n_all})")
        for j, future in enumerate(as_completed(future_results)):
            try:
                results = future.result()
                add_records(results_file_name, pd.DataFrame(data=results))
            except TimeoutError:
                print("Tournament timed-out")
                break
            except Exception as e:
                print(traceback.format_exc())
                print(e)


if __name__ == "__main__":
    cli()
