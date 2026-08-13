# Divergence from upstream RAMP

This fork is based on [RAMP-project/RAMP](https://github.com/RAMP-project/RAMP) v0.5.2.
This file records where its behaviour differs from upstream, for documentation and for any
future attempt to rebase onto, or contribute back to, the upstream project.

## 1. Household-level occupancy mask (`User.prob_home`)

**Status:** new feature, not present upstream in any form. Opt-in.

### What it adds

A `User` may be given a `prob_home`, the probability that a household of that category is
present on a given day. When set, one presence draw is made **per household, per day**, and
it gates every appliance belonging to that household simultaneously: an absent household
produces exactly zero load that day.

The motivation is that upstream RAMP has no notion of occupancy at all. The only way to
approximate an intermittently absent household is to lower each appliance's
`occasional_use`, which makes every appliance roll its *own independent* absence die. That
produces incoherent days — one appliance of a routine runs while others that belong to the
same routine do not — because the draws are independent. Absence is a household-level fact.

### Where it touches upstream code

All changes are confined to `ramp/core/core.py`:

| Location | Change |
|---|---|
| `User.__init__` | New optional `prob_home` kwarg (default `None`), validated to lie in [0, 1]. Sets `self.prob_home` and `self.is_home_today = True`. |
| `User.occupancy_mask_enabled` | New read-only property, `prob_home is not None`. |
| `User.generate_single_load_profile` | One presence draw per call, stored on `self.is_home_today`. |
| `Appliance.generate_load_profile` | `not self.user.is_home_today` added as the **first** clause of the existing skip condition. |
| `UseCase.generate_daily_load_profiles_parallel` | Raises `NotImplementedError` if any user has the mask enabled. |
| `User.save` | Warns that `prob_home` is not persisted. |

The design deliberately mirrors the existing `rand_daily_pref` mechanism, which is already a
per-household, per-day shared draw stored on `User` and read by appliances through their
`self.user` back-reference. No structural change to the object model was required.

### Semantics

Presence and `occasional_use` are independent draws and compose multiplicatively:

```
P(appliance runs on a given day) = prob_home * occasional_use
```

So with the mask active, `occasional_use` means "fraction of **home** days" rather than
"fraction of all days". Input values are expected to be supplied on that basis. The mask can
only remove days on which an appliance would otherwise have run; it never pushes an
appliance's usage towards certainty on home days.

`prob_home` is defined per user *category*, but each of the `num_users` households in that
category rolls its own independent draw, so a category does not go absent in lockstep.

### Backwards compatibility

With `prob_home=None` (the default) the presence draw is short-circuited and **no random
number is consumed**, so the random stream is identical to upstream. This was verified by
running an identical seeded 3-user, 6-appliance, 365-day use case against a clean worktree at
the pre-feature commit and comparing the SHA-256 of the resulting profile array: identical
(`bc2ba40b…`, total 150758277.876). `tests/test_occupancy_mask.py` also asserts the
equivalence at the level of the RNG state.

### Known limitations

- **Not supported with `parallel_processing=True`** — raises `NotImplementedError`. The
  parallel path (`generate_daily_load_profiles_parallel`) dispatches one task per
  `(appliance, day)`, repeated `num_users` times with no household identity, and pickles
  appliance copies into worker processes. There is no per-household object on which a shared
  draw could live. Supporting it means restructuring the task unit to be per-household, which
  was out of scope here.
- **Not part of the .xlsx model format.** `prob_home` is a `User` attribute, but users are
  serialized by replicating user-level columns onto each appliance row. Adding a column would
  change the on-disk schema relative to upstream, so the feature is python-API-only for now.
  `User.save` warns rather than silently dropping the setting. Adding the column later is
  purely additive: files without it would still load with the mask disabled.
- **`User.__eq__` does not compare `prob_home`.** Two users differing only in `prob_home`
  compare equal. This was left alone deliberately to avoid perturbing the xlsx round-trip
  equality tests, and is consistent with `prob_home` not being part of the saved format.
  (Note that upstream's `User.__eq__` is in any case ineffective — it builds its result with
  `np.append` without assigning the return value, so `answer.all()` is called on an empty
  array and yields `True` regardless.)
- **Absence gates every appliance of the household**, including `flat` ones such as a
  refrigerator, which in reality would keep running while the occupants are away. Model such
  appliances on a separate `User` without a `prob_home`.

### Tooling added

- `tests/test_occupancy_mask.py` — 21 tests covering opt-in behaviour, RNG-stream identity
  when disabled, household-coherent gating, independence between households of a category,
  multiplicativity of the two probabilities, the guarantee that occupancy never forces an
  appliance on, and the parallel/save guards.
- `scripts/occupancy_sweep.py` — sweeps `prob_home` and reports the aggregate load curve plus
  two day-to-day variability metrics per setting. Not part of the package.

## 2. Windows of use crossing midnight

**Status:** pre-existing on this branch, predates the occupancy-mask work.

Appliance windows may be declared with an end time beyond minute 1440. Switch-on events may
then span midnight, and the part of an event past 1440 is carried over to the morning of the
following day (`UseCase._accumulate_day`) rather than being truncated. The horizon is treated
as periodic, so the last day spills onto the first, conserving total energy. See
`docs/source/input_parameters.rst` and `tests/test_midnight_windows.py`.

## 3. `calc_peak_time_range` no longer depends on simulation state

**Status:** fix for an upstream defect, verified present in upstream 0.5.2 (`13e0cee`).

### The defect

`Appliance.maximum_profile` is derived from `self.daily_use`. Before any generation that
attribute holds the window-of-use mask, which is what the "theoretical maximum profile" is
supposed to be. After `generate_daily_load_profiles` has run, it holds the last simulated
day's *realised* profile instead. `UseCase.calc_peak_time_range` sums those into
`tot_max_profile` and takes

```python
peak_window = np.squeeze(np.argwhere(tot_max_profile == np.amax(tot_max_profile)))
```

Two things then go wrong. The profile is the wrong shape, and because a realised profile is
already expressed in watts, multiplying it by `np.mean(self.power) * self.number` scales it by
the power a second time. Occasionally the realised maximum falls on a single minute, in which
case `np.squeeze` collapses the result to a 0-dimensional array and the next line,
`peak_window[-1]`, raises `IndexError: too many indices for array`.

Measured over 200 trials of *generate, then recompute*, on the pre-fix tree:

| | pre-fix | post-fix |
|---|---|---|
| theoretical max profile unchanged by generation | 0/200 | 200/200 |
| drifted | 200/200 | 0/200 |
| raised `IndexError` | 2/200 | 0/200 |

The crash was therefore the rare and *lucky* outcome. Every recomputation was wrong; only 1%
of them said so, the rest returning a plausible peak window shifted by a median of about 150
minutes, sometimes past minute 1440.

### Exposure

A use case which is built, initialized and generated once — the normal workflow, and what all
the example scripts and the previous tests do — never recomputes the peak time range and is
therefore unaffected. The defect only bites when `calc_peak_time_range` runs *after* a
generation, which happens when a fresh `UseCase` is built from already-simulated `User`
objects (what a parameter sweep naturally does), when `initialize` is called a second time, or
when `peak_enlarge` is assigned, since its setter recomputes. It was found while building the
`prob_home` sweep harness.

### The fix

`Appliance.windows` now stores an immutable copy of the window mask in a new `windows_mask`
attribute, and `maximum_profile` reads that instead of `daily_use`. The two are identical
until the first day is simulated, so any use case that computes its peak time range before
generating — every existing one — is unaffected. Verified by comparing the SHA-256 of a seeded
365-day reference run against a clean worktree at the pre-fix commit: identical. Additionally
`np.atleast_1d` now wraps the `np.squeeze`, so a peak window of a single minute stays
indexable.

`windows_mask` is not in `APPLIANCE_ATTRIBUTES`, so it does not affect `Appliance.__eq__`, the
saved .xlsx format, or the round-trip tests.

Covered by `tests/test_peak_time_range_stability.py` (7 tests, 4 of which fail on the pre-fix
tree).

## 4. Upstream bug noticed but not fixed: parallel generation ignores `rand_daily_pref`

`generate_daily_load_profiles_parallel` does not reproduce the sequential path's behaviour for
appliances with `pref_index != 0`. Those appliances read `self.user.rand_daily_pref`, which is
only ever assigned inside `User.generate_single_load_profile` — a method the parallel path
never calls. In parallel mode it therefore keeps its initial value of `0` for the whole
simulation, so `pref_index` filtering behaves differently from a sequential run. This is
present upstream and unrelated to the occupancy mask, but it is the same root cause: the
parallel path discards user-level state. Left unfixed as out of scope.
