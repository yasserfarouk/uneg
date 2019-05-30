import pytest
from negmas import Issue, MappingUtilityFunction

from uneg.learners import RankingLAPUfunLearner, _ufun_objective


def test_lap_ufun_full_ranking():
    issues = [Issue(5, "Price"), Issue(5, "Distance")]
    gt = {
        (o["Price"], o["Distance"]): 0.2 * o["Price"] - 0.45 * o["Distance"]
        for o in Issue.enumerate(issues, astype=dict)
    }
    full_ranking = [
        _[0]
        for _ in sorted(zip(gt.keys(), gt.values()), key=lambda x: x[1], reverse=True)
    ]
    for kind in ("error_sums", "errors"):
        ufun = RankingLAPUfunLearner(issues=issues, degree=1, kind=kind)
        ufun.fit(ranking_list=full_ranking)
        learned_ranking = [
            _[0]
            for _ in sorted(
                zip(ufun.uvals.keys(), ufun.uvals.values()),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        if kind == "error_sums":
            assert full_ranking == learned_ranking, f"Failed on {kind}"
            assert (
                ufun.ranking_error(
                    gt=MappingUtilityFunction(lambda o: 0.2 * o[0] - 0.45 * o[1])
                )
                == 0.0
            )


def test_lap_ufun_partial_ranking():
    issues = [Issue(5, "Price"), Issue(5, "Distance")]
    gt = {
        (o["Price"], o["Distance"]): 0.2 * o["Price"] - 0.45 * o["Distance"]
        for o in Issue.enumerate(issues, astype=dict)
    }
    ufun = RankingLAPUfunLearner(issues=issues, degree=1)
    ufun.fit(ranking_list=[(4, 0), (3, 0), (3, 1), (2, 4), (0, 4)])
    assert ufun.theta[0] > 0.0 > ufun.theta[1]


@pytest.mark.parametrize(("kind",), (("error_sums",), ("errors",)))
def test_ufun_quality_error_sums(kind):
    issues = [Issue(5, "Price"), Issue(5, "Distance")]
    gt = {
        (o["Price"], o["Distance"]): 0.2 * o["Price"] - 0.45 * o["Distance"]
        for o in Issue.enumerate(issues, astype=dict)
    }
    full_ranking = [
        _[0]
        for _ in sorted(zip(gt.keys(), gt.values()), key=lambda x: x[1], reverse=True)
    ]
    ufun = RankingLAPUfunLearner(issues=issues, degree=1, kind=kind)
    assert (
        _ufun_objective(
            [0.2, -0.45],
            ranking=full_ranking,
            fs=ufun.fs,
            n_params=ufun.n_params,
            tolerance=ufun.tolerance,
            kind=ufun.kind,
        )
        == 0.0
    )
    assert (
        _ufun_objective(
            [0.2, 0.4],
            ranking=full_ranking,
            fs=ufun.fs,
            n_params=ufun.n_params,
            tolerance=ufun.tolerance,
            kind=ufun.kind,
        )
        > 0.0
    )
