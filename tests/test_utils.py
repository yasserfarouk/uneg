import pytest

from negmas import Issue

from uneg.utils import likelihood, ranking, partial
import numpy as np
import negmas.utilities as util


@pytest.mark.parametrize(("ascending",), [(True,), (False,)])
def test_ranking(ascending):
    issues = [Issue(5, "Price"), Issue(5, "Distance")]
    outcomes = Issue.enumerate(issues)
    ufun = util.UtilityFunction.generate_random(1, outcomes=outcomes)[0]

    rank = ranking(ufun, outcomes, ascending=ascending)

    for r1, r2 in zip(rank[:-1], rank[1:]):
        assert (ascending and ufun(r1) <= ufun(r2)) or (
            not ascending and ufun(r1) >= ufun(r2)
        )


@pytest.mark.parametrize(
    ("ascending", "fraction"),
    [
        (True, 0.0),
        (False, 0.0),
        (True, 0.2),
        (False, 0.2),
        (True, 0.4),
        (False, 0.4),
        (True, 0.6),
        (False, 0.6),
        (True, 0.8),
        (False, 0.8),
        (True, 1.0),
        (False, 1.0),
    ],
)
def test_partial_ranking(ascending, fraction):
    issues = [Issue(5, "Price"), Issue(5, "Distance")]
    outcomes = Issue.enumerate(issues)
    ufun = util.UtilityFunction.generate_random(1, outcomes=outcomes)[0]

    rank = partial(ranking(ufun, outcomes, ascending=ascending), fraction)

    assert len(rank) == int(len(outcomes) * fraction + 0.5)

    for r1, r2 in zip(rank[:-1], rank[1:]):
        assert (ascending and ufun(r1) <= ufun(r2)) or (
            not ascending and ufun(r1) >= ufun(r2)
        )
