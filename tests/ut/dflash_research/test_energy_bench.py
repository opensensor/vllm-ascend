import pytest

from tools.dflash_research.energy_bench import integrate_power


def test_integrate_power_accounts_for_elapsed_time():
    # The second interval is twice as long; an unweighted mean would be wrong.
    assert integrate_power([(0.0, 100.0), (1.0, 200.0), (3.0, 200.0)]) == 550.0


@pytest.mark.parametrize(
    "samples",
    [[], [(0.0, 100.0)], [(0.0, 100.0), (0.0, 100.0)], [(0.0, -1.0), (1.0, 2.0)]],
)
def test_integrate_power_rejects_invalid_samples(samples):
    with pytest.raises(ValueError):
        integrate_power(samples)
