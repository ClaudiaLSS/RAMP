"""Tests that the theoretical maximum profile does not depend on the simulation state.

``UseCase.calc_peak_time_range`` derives the peak window from the aggregated
``maximum_profile`` of the users, which is meant to describe the windows of use of their
appliances. It used to be computed from ``Appliance.daily_use``, an attribute overwritten
with the realised load of the day being simulated on every call to ``generate_load_profile``.
Recomputing the peak time range after a generation run therefore read a realised profile
instead of the windows of use, and scaled an already power-weighted profile by the power a
second time. The result was a peak window shifted by hours, or, when the realised maximum
happened to fall on a single minute, an ``IndexError`` from indexing a 0-dimensional array.
"""

import random

import numpy as np
import pytest

from ramp import UseCase, User


def make_usecase(num_days=10):
    user = User(user_name="household", num_users=3)
    morning = user.add_appliance(
        name="morning",
        number=2,
        power=100,
        num_windows=2,
        func_time=200,
        func_cycle=10,
        occasional_use=0.8,
    )
    morning.windows(window_1=[400, 700], window_2=[1000, 1300], random_var_w=0.1)
    evening = user.add_appliance(
        name="evening",
        number=1,
        power=500,
        num_windows=1,
        func_time=90,
        func_cycle=15,
        occasional_use=0.5,
    )
    evening.windows(window_1=[1050, 1350], random_var_w=0.1)

    usecase = UseCase(users=[user])
    usecase.initialize(num_days=num_days)
    return usecase, user


def aggregated_maximum_profile(usecase):
    total = np.zeros(1440)
    for user in usecase.users:
        total = total + user.maximum_profile
    return total


def test_windows_mask_matches_daily_use_before_any_simulation():
    """The two are identical until the first day is simulated, which is what makes the fix
    behaviour-preserving for any use case that computes its peak time range up front."""
    _, user = make_usecase()
    for app in user.App_list:
        assert np.array_equal(app.windows_mask, app.daily_use)


def test_maximum_profile_is_unchanged_by_generation():
    random.seed(11)
    usecase, _ = make_usecase()
    before = aggregated_maximum_profile(usecase)
    usecase.generate_daily_load_profiles(flat=False)
    after = aggregated_maximum_profile(usecase)
    assert np.array_equal(before, after)


def test_appliance_maximum_profile_ignores_realised_load():
    random.seed(12)
    usecase, user = make_usecase()
    app = user.App_list[0]
    before = app.maximum_profile.copy()
    usecase.generate_daily_load_profiles(flat=False)
    # daily_use now holds a realised profile, and must not leak into maximum_profile
    assert app.daily_use.sum() != app.windows_mask.sum()
    assert np.array_equal(app.maximum_profile, before)


def test_peak_time_range_can_be_recomputed_after_generation():
    random.seed(13)
    usecase, _ = make_usecase()
    before = usecase.calc_peak_time_range()
    usecase.generate_daily_load_profiles(flat=False)
    # must not raise, and must be drawn from the same theoretical maximum profile
    after = usecase.calc_peak_time_range()
    assert after.size > 0
    assert isinstance(before, np.ndarray)


def test_usecase_can_be_rebuilt_from_already_simulated_users():
    """The pattern any parameter sweep uses: reuse User objects across UseCase instances."""
    random.seed(14)
    usecase, user = make_usecase()
    usecase.generate_daily_load_profiles(flat=False)

    for _ in range(5):
        rebuilt = UseCase(users=[user])
        rebuilt.initialize(num_days=10)  # calls calc_peak_time_range internally
        assert rebuilt.peak_time_range.size > 0
        rebuilt.generate_daily_load_profiles(flat=False)


def test_single_minute_peak_window_does_not_crash():
    """A theoretical maximum attained at exactly one minute collapses to a 0-dimensional
    array under np.squeeze, which used to raise IndexError when indexed."""
    random.seed(15)
    user = User(user_name="solo", num_users=1)
    spike = user.add_appliance(
        name="spike", number=1, power=100, num_windows=1, func_time=1, func_cycle=1
    )
    spike.windows(window_1=[600, 601], random_var_w=0.0)

    usecase = UseCase(users=[user])
    usecase.initialize(num_days=2)
    assert usecase.peak_time_range.size > 0


def test_peak_enlarge_setter_after_generation():
    """Assigning peak_enlarge recomputes the peak time range, so it hit the same defect."""
    random.seed(16)
    usecase, _ = make_usecase()
    usecase.generate_daily_load_profiles(flat=False)
    usecase.peak_enlarge = 0.25  # must not raise
    assert usecase.peak_time_range.size > 0
