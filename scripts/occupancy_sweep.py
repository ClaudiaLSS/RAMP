#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep the household occupancy mask (``prob_home``) and report its effect on
day-to-day variability.

For each value of ``prob_home`` the script generates a full set of daily load profiles and
writes out

* the aggregate (mean over days) load curve, one column per setting;
* a table of variability metrics, one row per setting;
* a two panel figure showing both.

The metric the occupancy mask is meant to move is day-to-day variability. Two are
reported:

``mean_rsd``
    Mean relative standard deviation across days, computed minute by minute::

        mean_t [ std_d( L[d, t] ) / mean_d( L[d, t] ) ]

    over the minutes of the day whose mean load is non-zero. This is the shape-aware
    metric: it asks how much a given minute of the day varies from one day to the next.

``daily_energy_cv``
    Coefficient of variation of the total daily energy, i.e. ``std_d(E_d) / mean_d(E_d)``.
    A single scalar, useful for a quick comparison between settings.

On ``occasional_use``
---------------------
With the occupancy mask active, an appliance's ``occasional_use`` means "fraction of HOME
days" rather than "fraction of all days", because the two draws compose multiplicatively::

    P(appliance runs on a given day) = prob_home * occasional_use

A sweep over ``prob_home`` with fixed ``occasional_use`` therefore lowers the average load
as ``prob_home`` falls. That is correct behaviour, but it conflates two effects. Pass
``--preserve-mean`` to rescale each appliance's ``occasional_use`` to
``occasional_use / prob_home``, which holds the unconditional run probability constant so
that the average curve stays put and only the variability moves. That isolates the effect
the mask exists to produce.

Mean preservation is only achievable while ``occasional_use / prob_home <= 1``: an
appliance cannot run more often than every single home day. Below
``prob_home = occasional_use`` the rescaled value is clipped at 1.0 and the average load
necessarily falls. The script warns, per appliance, whenever that happens, so a sweep that
has silently stopped preserving the mean is visible rather than misleading.

Examples
--------
Run the built-in demo household::

    python scripts/occupancy_sweep.py --preserve-mean

Sweep your own model, defined in a RAMP python input file exposing ``User_list``::

    python scripts/occupancy_sweep.py --input ramp/example/input_file_1.py \\
        --prob-home 1.0 0.8 0.6 0.4 --days 730 --preserve-mean
"""

import argparse
import importlib.util
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# allow running the script straight from a checkout without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ramp.core.core import UseCase, User  # noqa: E402

DAY_MINUTES = 1440
DEFAULT_PROB_HOME = [1.0, 0.9, 0.8, 0.7, 0.6]


# --------------------------------------------------------------------------------------
# model definition
# --------------------------------------------------------------------------------------


def demo_users():
    """A small absence-prone settlement, used when no input file is given.

    The occasional_use values are present-conditional: they describe how often each
    appliance is used on a day the household is actually at home. They are kept at or below
    the lowest default prob_home so that --preserve-mean can hold across the whole default
    sweep without clipping.

    Only a handful of households is simulated, deliberately. Absence is drawn per household,
    so in a settlement of many identical households the zero-load days average out and the
    bimodality is only visible in a single household's profile; with a few households it
    shows up in the aggregate too.
    """
    household = User(user_name="absence prone household", num_users=4)

    lights = household.add_appliance(
        name="lights",
        number=6,
        power=15,
        num_windows=2,
        func_time=180,
        func_cycle=10,
        occasional_use=0.6,
    )
    lights.windows(window_1=[350, 500], window_2=[1080, 1400], random_var_w=0.2)

    tv = household.add_appliance(
        name="tv",
        number=1,
        power=90,
        num_windows=1,
        func_time=200,
        func_cycle=20,
        occasional_use=0.45,
    )
    tv.windows(window_1=[1080, 1400], random_var_w=0.25)

    stove = household.add_appliance(
        name="stove",
        number=1,
        power=1200,
        num_windows=2,
        func_time=70,
        func_cycle=15,
        occasional_use=0.55,
    )
    stove.windows(window_1=[400, 560], window_2=[1020, 1220], random_var_w=0.2)

    washing_machine = household.add_appliance(
        name="washing machine",
        number=1,
        power=650,
        num_windows=1,
        func_time=100,
        func_cycle=100,
        occasional_use=0.3,
    )
    washing_machine.windows(window_1=[540, 1020], random_var_w=0.2)

    return [household]


def load_users_from_input_file(path):
    """Import a RAMP python input file and return the ``User_list`` it defines"""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location("ramp_input_file", path)
    module = importlib.util.module_from_spec(spec)
    # input files import from ramp and may reference files next to themselves
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)

    if not hasattr(module, "User_list"):
        raise AttributeError(
            f"{path} does not define a 'User_list'; RAMP python input files are expected "
            "to collect their User instances in a module level list of that name"
        )
    return module.User_list


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def variability_metrics(daily_profiles):
    """Compute day-to-day variability metrics from a (num_days, 1440) array"""
    mean_curve = daily_profiles.mean(axis=0)
    std_curve = daily_profiles.std(axis=0)

    # relative standard deviation is only defined where the mean load is non-zero, which
    # excludes minutes of the day at which nothing ever runs
    active = mean_curve > 0
    mean_rsd = (
        float((std_curve[active] / mean_curve[active]).mean()) if active.any() else 0.0
    )

    daily_energy = daily_profiles.sum(axis=1)
    daily_energy_cv = (
        float(daily_energy.std() / daily_energy.mean())
        if daily_energy.mean() > 0
        else 0.0
    )

    return {
        "mean_rsd": mean_rsd,
        "daily_energy_cv": daily_energy_cv,
        "mean_daily_energy_kWh": float(daily_energy.mean() / 60 / 1000),
        "peak_W": float(mean_curve.max()),
        "zero_load_days_frac": float((daily_energy == 0).mean()),
    }


# --------------------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------------------


def run_single(usecase, prob_home, seed, preserve_mean, base_occasional_use):
    """Generate daily profiles for one prob_home setting

    The users of ``usecase`` are mutated in place (prob_home and possibly occasional_use),
    which is why the baseline occasional_use values are captured once by the caller and
    restored here.

    The use case itself is built once by the caller and reused across settings. It must not
    be rebuilt between settings: ``UseCase.calc_peak_time_range`` derives the theoretical
    maximum profile from each appliance's ``daily_use``, which after a generation run holds
    a realised profile rather than the window mask, so calling it a second time raises an
    IndexError. Holding the peak time range fixed is also what a sweep wants, since it
    leaves prob_home as the only thing varying between settings.
    """
    clipped = []
    for user in usecase.users:
        user.prob_home = prob_home
        for app in user.App_list:
            base = base_occasional_use[(id(user), id(app))]
            if preserve_mean and prob_home not in (None, 0):
                # hold P(run) = prob_home * occasional_use constant across the sweep
                rescaled = base / prob_home
                if rescaled > 1.0:
                    # an appliance cannot run more often than every home day, so the mean
                    # cannot be preserved for this appliance at this prob_home
                    clipped.append((user.user_name, app.name, base, prob_home * 1.0))
                    rescaled = 1.0
                app.occasional_use = rescaled
            else:
                app.occasional_use = base

    if clipped:
        total = sum(len(user.App_list) for user in usecase.users)
        worst = max(clipped, key=lambda row: row[2])
        examples = ", ".join(
            f"{app_name} ({user})" for user, app_name, _, _ in clipped[:3]
        )
        if len(clipped) > 3:
            examples += f", +{len(clipped) - 3} more"
        print(
            f"\n    WARNING: mean NOT preserved at prob_home={prob_home:g}. "
            f"{len(clipped)}/{total} appliances need occasional_use > 1 to compensate and "
            f"were clipped at 1.0: {examples}.\n"
            f"             Worst case '{worst[1]}' of '{worst[0]}': unconditional run "
            f"probability drops from {worst[2]:g} to {worst[3]:g}. "
            f"The average load will fall rather than stay constant.\n   ",
            end=" ",
        )

    random.seed(seed)
    return usecase.generate_daily_load_profiles(flat=False)


def sweep(users, prob_home_values, num_days, seed, preserve_mean):
    base_occasional_use = {
        (id(user), id(app)): app.occasional_use
        for user in users
        for app in user.App_list
    }

    # built once and reused for every setting, see run_single
    usecase = UseCase(name="occupancy sweep", users=users)
    usecase.initialize(num_days=num_days)

    curves = {}
    rows = []
    for prob_home in prob_home_values:
        print(f"  prob_home={prob_home:<5} ...", end="", flush=True)
        profiles = run_single(
            usecase, prob_home, seed, preserve_mean, base_occasional_use
        )
        label = f"prob_home={prob_home:g}"
        curves[label] = profiles.mean(axis=0)

        metrics = variability_metrics(profiles)
        rows.append({"prob_home": prob_home, **metrics})
        print(
            f" mean_rsd={metrics['mean_rsd']:.3f}"
            f"  daily_energy_cv={metrics['daily_energy_cv']:.3f}"
            f"  mean_daily_energy={metrics['mean_daily_energy_kWh']:.2f} kWh"
            f"  zero_load_days={metrics['zero_load_days_frac']:.2%}"
        )

    aggregate = pd.DataFrame(curves)
    aggregate.index.name = "minute_of_day"
    variability = pd.DataFrame(rows).set_index("prob_home")
    return aggregate, variability


def plot(aggregate, variability, path, preserve_mean):
    fig, (ax_curve, ax_var) = plt.subplots(1, 2, figsize=(13, 4.8))

    hours = aggregate.index / 60
    for column in aggregate.columns:
        ax_curve.plot(hours, aggregate[column] / 1000, label=column, linewidth=1.4)
    ax_curve.set_xlabel("hour of day")
    ax_curve.set_ylabel("mean load [kW]")
    ax_curve.set_xlim(0, 24)
    ax_curve.set_xticks(range(0, 25, 4))
    title = "Aggregate load curve"
    if preserve_mean:
        title += "\n(occasional_use rescaled: mean held constant)"
    ax_curve.set_title(title)
    ax_curve.legend(fontsize=8)
    ax_curve.grid(alpha=0.3)

    ax_var.plot(
        variability.index, variability["mean_rsd"], "o-", label="mean RSD across days"
    )
    ax_var.plot(
        variability.index,
        variability["daily_energy_cv"],
        "s--",
        label="daily energy CV",
    )
    ax_var.set_xlabel("prob_home")
    ax_var.set_ylabel("day-to-day variability")
    ax_var.set_title("Day-to-day variability vs. occupancy")
    ax_var.invert_xaxis()  # increasing absence to the right
    ax_var.legend(fontsize=8)
    ax_var.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        default=None,
        help="RAMP python input file defining a 'User_list'. Defaults to a built-in demo household.",
    )
    parser.add_argument(
        "--prob-home",
        nargs="+",
        type=float,
        default=DEFAULT_PROB_HOME,
        help=f"prob_home values to sweep, default: {DEFAULT_PROB_HOME}",
    )
    parser.add_argument(
        "--days", type=int, default=365, help="number of days to simulate"
    )
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--preserve-mean",
        action="store_true",
        help="rescale occasional_use by 1/prob_home so the average load curve stays put "
        "and only the variability changes",
    )
    parser.add_argument(
        "--output-dir",
        default="occupancy_sweep_results",
        help="directory for the CSV and PNG outputs",
    )
    args = parser.parse_args(argv)

    for value in args.prob_home:
        if not 0 <= value <= 1:
            parser.error(f"prob_home must lie within [0, 1], got {value}")

    os.makedirs(args.output_dir, exist_ok=True)

    users = load_users_from_input_file(args.input) if args.input else demo_users()
    print(
        f"Sweeping prob_home over {args.prob_home} for {len(users)} user "
        f"categor{'y' if len(users) == 1 else 'ies'} across {args.days} days"
        f"{' (mean preserved)' if args.preserve_mean else ''}"
    )

    aggregate, variability = sweep(
        users, args.prob_home, args.days, args.seed, args.preserve_mean
    )

    aggregate_path = os.path.join(args.output_dir, "aggregate_curves.csv")
    variability_path = os.path.join(args.output_dir, "variability.csv")
    figure_path = os.path.join(args.output_dir, "occupancy_sweep.png")

    aggregate.to_csv(aggregate_path)
    variability.to_csv(variability_path)
    plot(aggregate, variability, figure_path, args.preserve_mean)

    print(f"\nWrote {aggregate_path}\n      {variability_path}\n      {figure_path}")
    print("\n" + variability.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
