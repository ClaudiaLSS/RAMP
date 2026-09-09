"""Tests for the household-level occupancy mask (``User.prob_home``).

The occupancy mask adds a single presence draw per household per day which gates all of
that household's appliances at once. The properties worth pinning down are:

* it is opt-in, and a use case which does not use it behaves exactly as before;
* it gates a whole household coherently, rather than each appliance independently;
* it composes multiplicatively with ``occasional_use`` and never forces an appliance on.
"""

import random

import numpy as np
import pytest

from ramp import UseCase, User


SEED = 4321


def make_user(prob_home=None, occasional_use=1.0, num_appliances=3, num_users=1):
    """A household with several always-available appliances in a wide common window"""
    user = User(user_name="household", num_users=num_users, prob_home=prob_home)
    for i in range(num_appliances):
        app = user.add_appliance(
            name=f"appliance_{i}",
            number=1,
            power=100,
            num_windows=1,
            func_time=120,
            func_cycle=10,
            occasional_use=occasional_use,
        )
        app.windows(window_1=[400, 1200], random_var_w=0.0)
    return user


def make_usecase(prob_home=None, occasional_use=1.0, num_days=400, **kwargs):
    user = make_user(prob_home=prob_home, occasional_use=occasional_use, **kwargs)
    usecase = UseCase(name="occupancy test", users=[user])
    usecase.initialize(num_days=num_days)
    usecase.peak_time_range = usecase.calc_peak_time_range()
    return usecase, user


def daily_energy(usecase):
    """Total load per day, as a 1D array of length num_days"""
    return usecase.generate_daily_load_profiles(flat=False).sum(axis=1)


# --------------------------------------------------------------------------------------
# opt-in behaviour
# --------------------------------------------------------------------------------------


def test_disabled_by_default():
    user = User(user_name="household")
    assert user.prob_home is None
    assert user.occupancy_mask_enabled is False
    assert user.is_home_today is True


def test_disabled_mask_consumes_no_random_number():
    """With the mask off the random stream must be untouched, so that existing use cases
    reproduce their previous results exactly."""
    user = make_user(prob_home=None)
    usecase = UseCase(users=[user])
    peak_time_range = usecase.calc_peak_time_range()

    random.seed(SEED)
    user.generate_single_load_profile(0, peak_time_range, 0)
    state_without_mask = random.getstate()

    # the same sequence of appliance draws, reached without going through the user method
    random.seed(SEED)
    for app in user.App_list:
        app.generate_load_profile(0, peak_time_range, 0, power=app.power[0])
    state_reference = random.getstate()

    assert state_without_mask == state_reference


def test_enabled_by_setting_prob_home():
    user = User(user_name="household", prob_home=0.4)
    assert user.occupancy_mask_enabled is True


@pytest.mark.parametrize("bad_value", [-0.1, 1.5, 2])
def test_invalid_prob_home_rejected(bad_value):
    with pytest.raises(ValueError, match="prob_home"):
        User(user_name="household", prob_home=bad_value)


@pytest.mark.parametrize("good_value", [0, 0.5, 1])
def test_boundary_prob_home_accepted(good_value):
    assert User(user_name="household", prob_home=good_value).prob_home == good_value


# --------------------------------------------------------------------------------------
# gating behaviour
# --------------------------------------------------------------------------------------


def test_always_absent_produces_zero_load():
    random.seed(SEED)
    usecase, _ = make_usecase(prob_home=0.0, num_days=30)
    assert daily_energy(usecase).sum() == 0


def test_always_home_produces_load_every_day():
    random.seed(SEED)
    usecase, _ = make_usecase(prob_home=1.0, occasional_use=1.0, num_days=30)
    assert np.all(daily_energy(usecase) > 0)


def test_absent_days_are_exactly_zero_not_merely_small():
    """An absent day must contain no load at all, including the small non-zero values
    RAMP writes into the windows of use as a mask."""
    random.seed(SEED)
    usecase, _ = make_usecase(prob_home=0.5, occasional_use=1.0, num_days=200)
    energy = daily_energy(usecase)
    absent = energy[energy < energy.max() / 2]
    assert np.all(absent == 0)
    # with prob_home=0.5 we do expect to have seen some absent days
    assert absent.size > 0


def test_whole_household_is_gated_together():
    """The defining property: on an absent day *every* appliance of the household is off,
    and on a home day the appliances follow their own occasional_use. A per-appliance
    absence draw would produce days where only some appliances are off."""
    random.seed(SEED)
    user = make_user(prob_home=0.5, occasional_use=1.0, num_appliances=4)
    usecase = UseCase(users=[user])
    usecase.initialize(num_days=200)
    peak_time_range = usecase.calc_peak_time_range()

    absent_days = 0
    for day in range(200):
        user.generate_single_load_profile(day, peak_time_range, 0)
        active = [app.daily_use.sum() > 0 for app in user.App_list]
        if user.is_home_today:
            # occasional_use is 1, so every appliance of a present household runs
            assert all(active), f"day {day}: home but not all appliances ran"
        else:
            assert not any(active), f"day {day}: absent but some appliance ran"
            absent_days += 1

    assert 0 < absent_days < 200


def test_households_of_same_category_are_gated_independently():
    """num_users identical households must each roll their own presence draw, otherwise a
    whole user category would go absent in lockstep."""
    random.seed(SEED)
    usecase, _ = make_usecase(
        prob_home=0.5, occasional_use=1.0, num_days=120, num_users=20
    )
    energy = daily_energy(usecase)
    # if all 20 households shared one draw, every day would be either zero or full load
    assert np.unique(energy).size > 3


# --------------------------------------------------------------------------------------
# composition with occasional_use
# --------------------------------------------------------------------------------------


def test_probability_is_multiplicative():
    """P(appliance runs on a given day) == prob_home * occasional_use"""
    prob_home, occasional_use = 0.6, 0.5
    num_days = 3000

    random.seed(SEED)
    user = make_user(
        prob_home=prob_home, occasional_use=occasional_use, num_appliances=1
    )
    usecase = UseCase(users=[user])
    usecase.initialize(num_days=num_days)
    peak_time_range = usecase.calc_peak_time_range()

    app = user.App_list[0]
    runs = 0
    for day in range(num_days):
        user.generate_single_load_profile(day, peak_time_range, 0)
        runs += app.daily_use.sum() > 0

    observed = runs / num_days
    expected = prob_home * occasional_use
    assert observed == pytest.approx(expected, abs=0.03)


def test_occupancy_never_forces_an_appliance_on():
    """The mask may only remove usage days; on home days occasional_use keeps its meaning,
    so the run frequency must not be pushed towards certainty."""
    occasional_use = 0.4
    num_days = 3000

    random.seed(SEED)
    user = make_user(prob_home=0.5, occasional_use=occasional_use, num_appliances=1)
    usecase = UseCase(users=[user])
    usecase.initialize(num_days=num_days)
    peak_time_range = usecase.calc_peak_time_range()

    app = user.App_list[0]
    runs_when_home = 0
    home_days = 0
    for day in range(num_days):
        user.generate_single_load_profile(day, peak_time_range, 0)
        if user.is_home_today:
            home_days += 1
            runs_when_home += app.daily_use.sum() > 0

    # conditional on being home, the appliance still runs on occasional_use of the days
    assert runs_when_home / home_days == pytest.approx(occasional_use, abs=0.04)


def test_variability_increases_with_absence():
    """The metric the feature targets: day-to-day variability of the daily energy should
    be higher for an absence-prone household than for an always-present one."""
    num_days = 400

    random.seed(SEED)
    usecase_always_home, _ = make_usecase(
        prob_home=1.0, occasional_use=0.5, num_days=num_days, num_users=10
    )
    baseline = daily_energy(usecase_always_home)

    random.seed(SEED)
    usecase_absent, _ = make_usecase(
        prob_home=0.5, occasional_use=0.5, num_days=num_days, num_users=10
    )
    masked = daily_energy(usecase_absent)

    cv_baseline = baseline.std() / baseline.mean()
    cv_masked = masked.std() / masked.mean()
    assert cv_masked > cv_baseline


# --------------------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------------------


def test_parallel_processing_is_rejected():
    user = make_user(prob_home=0.5)
    usecase = UseCase(users=[user], parallel_processing=True)
    usecase.initialize(num_days=5)
    with pytest.raises(NotImplementedError, match="occupancy mask"):
        usecase.generate_daily_load_profiles()


def test_parallel_processing_allowed_without_mask():
    """The guard must only fire for users which actually enable the mask."""
    user = make_user(prob_home=None)
    usecase = UseCase(users=[user], parallel_processing=True)
    usecase.initialize(num_days=2)
    usecase.peak_time_range = usecase.calc_peak_time_range()
    # should not raise
    usecase.generate_daily_load_profiles(flat=False)


def test_saving_warns_that_prob_home_is_not_persisted():
    user = make_user(prob_home=0.5)
    with pytest.warns(UserWarning, match="prob_home"):
        user.save()


def test_saving_without_mask_does_not_warn():
    user = make_user(prob_home=None)
    with warnings_as_errors():
        user.save()


class warnings_as_errors:
    """Context manager turning UserWarnings about prob_home into failures"""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.filterwarnings("error", message=".*prob_home.*")
        return self

    def __exit__(self, *args):
        return self._ctx.__exit__(*args)
