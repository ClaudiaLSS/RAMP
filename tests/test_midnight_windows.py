"""Tests for windows of use which cross midnight

An appliance used continuously from the evening until the next morning is declared with a
single window whose end time is greater than 1440, e.g. [1200, 1760] for 20:00 -> 05:20.
Switch-on events may then span midnight, and the part of an event which lies past minute
1440 is carried over to the morning of the next day instead of being cut off.
"""

import numpy as np
import pytest

from ramp.core.core import DAY_MINUTES, User, UseCase
from ramp.errors_logs.errors import InvalidWindow

WINDOW_START = 1200  # 20:00
WINDOW_END = 1760  # 05:20 of the next day
SPILL = WINDOW_END - DAY_MINUTES  # 320 minutes past midnight
DURATION = 540  # func_time == func_cycle -> a single 9 hour event
POWER = 100.0
NUM_DAYS = 6


def build_usecase(num_days=NUM_DAYS, parallel=False, **appliance_kwargs):
    """A use case with one appliance used continuously from 20:00 until ~05:00

    ``func_time == func_cycle == DURATION`` leaves the algorithm exactly one switch-on
    event per day, of exactly ``DURATION`` contiguous minutes, starting within
    ``[WINDOW_START, WINDOW_END - DURATION]``. Any such event necessarily crosses midnight,
    which makes the placement of its after-midnight part exactly checkable.
    """
    user = User(user_name="hh", num_users=1)
    kwargs = dict(
        number=1,
        power=POWER,
        num_windows=1,
        func_time=DURATION,
        time_fraction_random_variability=0,
        func_cycle=DURATION,
        name="overnight_light",
    )
    kwargs.update(appliance_kwargs)
    app = user.add_appliance(**kwargs)
    app.windows(window_1=[WINDOW_START, WINDOW_END], random_var_w=0)

    usecase = UseCase(users=[user], parallel_processing=parallel, random_seed=42)
    usecase.initialize(peak_enlarge=0.15, num_days=num_days)
    return usecase, app


def real_power(profiles):
    """Discard the 0.001 markers which flag the windows of use, keep actual power"""
    return np.where(profiles > 1.0, profiles, 0.0)


def switched_on_runs(flat_profiles):
    """Start and length of every maximal run of consecutive switched-on minutes"""
    on = np.concatenate(([False], flat_profiles > 1.0, [False]))
    edges = np.flatnonzero(np.diff(on.astype(int)))
    return [(edges[i], edges[i + 1] - edges[i]) for i in range(0, len(edges), 2)]


def continuous_midnights(flat_profiles):
    """Day boundaries at which the appliance stays switched on across midnight"""
    num_days = flat_profiles.size // DAY_MINUTES
    return [
        day_idx
        for day_idx in range(num_days - 1)
        if flat_profiles[(day_idx + 1) * DAY_MINUTES - 1] > 1.0
        and flat_profiles[(day_idx + 1) * DAY_MINUTES] > 1.0
    ]


# ---------------------------------------------------------------------------
# declaration and validation
# ---------------------------------------------------------------------------


def test_window_crossing_midnight_extends_the_simulated_profile():
    _, app = build_usecase()
    assert app.window_spill == SPILL
    assert app.profile_length == DAY_MINUTES + SPILL
    assert app.daily_use.size == DAY_MINUTES + SPILL


def test_window_within_the_day_is_not_extended():
    user = User(user_name="hh", num_users=1)
    app = user.add_appliance(number=1, power=POWER, num_windows=1, func_time=60)
    app.windows(window_1=[600, 900])
    assert app.window_spill == 0
    assert app.profile_length == DAY_MINUTES
    assert app.daily_use.size == DAY_MINUTES


def test_maximum_profile_stays_one_day_long():
    """The after-midnight part is folded back, so peak-time and plotting code is unaffected"""
    _, app = build_usecase()
    assert app.maximum_profile.size == DAY_MINUTES
    # the window covers 20:00-24:00 and, folded back, 00:00-05:20
    assert app.maximum_profile[WINDOW_START] > 0
    assert app.maximum_profile[SPILL - 1] > 0
    assert app.maximum_profile[SPILL] == 0
    # folding takes the maximum, so it never exceeds the fully switched-on value
    assert app.maximum_profile.max() == pytest.approx(0.001 * POWER)


@pytest.mark.parametrize(
    "window",
    [
        [1200, 2900],  # longer than one day
        [1500, 1600],  # starts after midnight
        [-10, 100],  # negative start
        [300, 100],  # end before start
    ],
)
def test_invalid_windows_are_rejected(window):
    user = User(user_name="hh", num_users=1)
    app = user.add_appliance(number=1, power=POWER, num_windows=1, func_time=60)
    with pytest.raises(InvalidWindow):
        app.windows(window_1=window)


# ---------------------------------------------------------------------------
# switch-on events actually cross midnight
# ---------------------------------------------------------------------------


def test_no_discontinuity_at_midnight():
    usecase, _ = build_usecase()
    profiles = usecase.generate_daily_load_profiles(flat=True)

    assert continuous_midnights(profiles) == list(range(NUM_DAYS - 1))


def test_switch_on_event_stays_in_one_piece_across_midnight():
    """A day boundary must fall inside a single DURATION long run, not between two runs"""
    usecase, _ = build_usecase()
    profiles = usecase.generate_daily_load_profiles(flat=True)
    runs = switched_on_runs(profiles)

    # skip the first and last day, whose runs are truncated by the ends of the array
    for day_idx in range(1, NUM_DAYS - 1):
        midnight = (day_idx + 1) * DAY_MINUTES
        containing = [r for r in runs if r[0] <= midnight < r[0] + r[1]]
        assert len(containing) == 1, f"midnight of day {day_idx} is not inside a run"
        assert containing[0][1] == DURATION


def test_last_day_spills_onto_the_first_one():
    """The horizon is periodic, so the morning of the first day is not left empty"""
    usecase, _ = build_usecase()
    profiles = usecase.generate_daily_load_profiles(flat=False)

    assert profiles[0, 0] == pytest.approx(POWER)


def test_energy_is_conserved_over_the_horizon():
    """Nothing is lost or duplicated at the day boundaries"""
    usecase, _ = build_usecase()
    profiles = usecase.generate_daily_load_profiles(flat=True)

    assert real_power(profiles).sum() == pytest.approx(NUM_DAYS * DURATION * POWER)


def test_no_power_outside_the_declared_window():
    """An event must not leak into minutes the user never declared as a window"""
    usecase, _ = build_usecase()
    profiles = usecase.generate_daily_load_profiles(flat=False)

    # the window covers 20:00-24:00 and 00:00-05:20, so 05:20 -> 20:00 is always outside
    assert np.count_nonzero(profiles[:, SPILL:WINDOW_START]) == 0


def test_flat_appliance_crosses_midnight():
    usecase, _ = build_usecase(flat="yes")
    profiles = usecase.generate_daily_load_profiles(flat=True)

    assert continuous_midnights(profiles) == list(range(NUM_DAYS - 1))
    # a flat appliance fills its whole window, without further stochasticity
    window_length = WINDOW_END - WINDOW_START
    assert real_power(profiles).sum() == pytest.approx(NUM_DAYS * window_length * POWER)


def test_parallel_processing_carries_over_midnight():
    usecase, _ = build_usecase(num_days=3, parallel=True)
    profiles = usecase.generate_daily_load_profiles(flat=True)

    assert continuous_midnights(profiles) == [0, 1]
    assert real_power(profiles).sum() == pytest.approx(3 * DURATION * POWER)


# ---------------------------------------------------------------------------
# contrast with the two-window formulation this replaces
# ---------------------------------------------------------------------------


def test_split_windows_break_at_midnight_while_a_single_window_does_not():
    """The same use declared as two windows, as was necessary before, is discontinuous"""
    user = User(user_name="hh", num_users=1)
    app = user.add_appliance(
        number=1,
        power=POWER,
        num_windows=2,
        func_time=DURATION,
        time_fraction_random_variability=0,
        func_cycle=60,
        name="overnight_light",
    )
    app.windows(
        window_1=[WINDOW_START, DAY_MINUTES], window_2=[0, SPILL], random_var_w=0.1
    )
    split = UseCase(users=[user], parallel_processing=False, random_seed=42)
    split.initialize(peak_enlarge=0.15, num_days=NUM_DAYS)
    split_profiles = split.generate_daily_load_profiles(flat=True)

    merged, _ = build_usecase()
    merged_profiles = merged.generate_daily_load_profiles(flat=True)

    assert len(continuous_midnights(merged_profiles)) == NUM_DAYS - 1
    assert len(continuous_midnights(split_profiles)) < NUM_DAYS - 1


# ---------------------------------------------------------------------------
# duty cycles
# ---------------------------------------------------------------------------


def test_duty_cycle_appliance_with_window_crossing_midnight():
    user = User(user_name="hh", num_users=1)
    app = user.add_appliance(
        number=1,
        power=200,
        num_windows=1,
        func_time=DURATION,
        time_fraction_random_variability=0,
        func_cycle=30,
        fixed_cycle=1,
        name="overnight_freezer",
    )
    app.windows(window_1=[WINDOW_START, WINDOW_END], random_var_w=0)
    app.specific_cycle_1(p_11=200, t_11=20, p_12=5, t_12=10)
    app.cycle_behaviour(cw11=[WINDOW_START, WINDOW_END])

    usecase = UseCase(users=[user], parallel_processing=False, random_seed=42)
    usecase.initialize(peak_enlarge=0.15, num_days=NUM_DAYS)
    profiles = usecase.generate_daily_load_profiles(flat=True)

    assert profiles.size == NUM_DAYS * DAY_MINUTES
    # the appliance runs, on both sides of midnight
    assert len(continuous_midnights(profiles)) > 0
