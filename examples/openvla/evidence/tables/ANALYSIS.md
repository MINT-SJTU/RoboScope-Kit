# OpenVLA experiment results and analysis

This directory summarizes three completed experiment groups: 200 historical Native episodes, 200 historical Kit 2/2 episodes, and 72 rollouts from the 36-pair diagnostic, totaling 472 executed episodes. The tables were extracted from recorded results and prepared on 2026-11-12. Other configurations and Autotune results are indexed in the [evidence guide](../README.md).

## Comparison files

| File | Contents |
| --- | --- |
| [Historical 200 pairs](historical-200.csv) | All task/state pairs across four suites, Native and Kit 2/2 outcomes and executed steps, and the 36 historical disagreements |
| [Diagnostic 36 pairs](diagnostic-36.csv) | Historical and diagnostic outcomes and steps, the 18 remaining disagreements, initial-image agreement, and first observation/input/action differences |
| [Runs and commands](../reproduction/RUNS.md) | Hardware, runtime versions, 8 historical commands, and 10 diagnostic batch commands |
| [Experiment metadata and findings](../reproduction/PROVENANCE.json) | Run identifiers, source revisions, table SHA256 hashes, coverage checks, and numerical diagnostic findings |

## Results

| Suite | Historical Native successes | Historical Kit successes | Historical disagreements | Diagnostic Native successes | Diagnostic Kit successes | Diagnostic disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial | 43/50 | 40/50 | 9 | 4/9 | 4/9 | 2 |
| Object | 42/50 | 44/50 | 6 | 2/6 | 6/6 | 4 |
| Goal | 33/50 | 39/50 | 8 | 7/8 | 7/8 | 2 |
| Long | 28/50 | 23/50 | 13 | 10/13 | 2/13 | 10 |
| Total | 146/200 | 146/200 | 36 | 23/36 | 19/36 | 18 |

Cases with remaining outcome disagreements, using zero-based task/state indices:

| Suite | Native success / Kit failure | Native failure / Kit success |
| --- | --- | --- |
| Spatial | 7/0 | 6/3 |
| Object | — | 3/0, 6/0, 6/4, 8/4 |
| Goal | 4/1 | 0/3 |
| Long | 0/2, 1/1, 2/0, 2/1, 2/2, 4/3, 6/1, 7/0, 8/1 | 7/1 |

## Changes from historical runs to the diagnostic

The diagnostic revisited the historical disagreement subset with the same task/state identities. Subset scheduling changed execution order and each Worker's reset history.

| Suite | Pairs revisited | Native outcome changes | Kit outcome changes |
| --- | ---: | ---: | ---: |
| Spatial | 9 | 6 | 3 |
| Object | 6 | 0 | 2 |
| Goal | 8 | 6 | 0 |
| Long | 13 | 1 | 2 |
| Total | 36 | 13 | 7 |

## Analysis

The full evaluation gives 146/200 successes on each path, with 36 paired outcome
disagreements. Goal favors Kit by six successes (39 vs. 33), while Long favors
Native by five (28 vs. 23). Equal overall totals therefore coexist with different
suite and episode outcomes.

The diagnostic revisits all 36 historical disagreements once on each path. Native
outcomes change in 13 cases and Kit outcomes in seven; 18 pairs still disagree.
Goal reaches 7/8 successes on both paths, with two opposing episode outcomes.
Long remains asymmetric at 10/13 Native successes and 2/13 Kit successes. These
rates describe the selected disagreement subset, not a second full-suite score.
Because the subset changes execution order and Worker reset history, the outcome
changes measure sensitivity across these runs rather than isolating variation
under identical execution conditions.

The diagnostic findings identify several points where trajectories can diverge:

- All 36 paired initial-state arrays match, while the initial images match in
  33 pairs. The exceptions are Spatial task/state 8/2 and 8/3, and Long 9/4.
  The Long 9/4 inspection measures a maximum initial body-position coordinate
  difference of about 7.17 mm despite equal initial-state arrays.
- The preprocessing inspection covers 34 milestones and reports matching CPU
  outputs against Native. The recorded color differences affect one to three
  components, with a maximum absolute difference of one intensity level. These
  findings locate numerical differences but do not establish their effect on
  every episode outcome.
- In the Spatial 7/0 action replay, Native-precision actions reproduce the Native
  images. Casting those actions to float32 first changes an image at step 7;
  both replays still succeed after 112 actions. The Kit Worker replay of those
  float32 actions matches the corresponding Native float32 replay at every
  observation and also succeeds after 112 actions. This case demonstrates
  sensitivity to action precision without attributing its closed-loop outcome
  difference solely to that conversion.
- Executed Worker actions match Policy actions in the diagnostic checks. The
  paired table distinguishes exact executed-action differences, including
  float32 rounding, from model-action differences larger than `1e-6`.

Together, these observations identify initial-observation, preprocessing, and
action-precision differences relevant to the comparison. The 18 remaining
outcome disagreements, especially the Long subset, are part of the measured
result; the diagnostic does not establish a single cause for all of them.

## Table conventions

- `True` / `False` indicate success/failure or whether a field's condition holds.
  Filter on `outcome_disagrees=True` to select disagreements.
- `task_id` and `initial_state_index` are zero-based; their pair identifies an
  episode within a suite. Both tables include the task name and episode ID.
- `*_steps` counts executed Policy actions, excluding the ten settling steps.
- Difference step indices start at zero. Images are observations before the
  corresponding action; `none` means that difference was absent within the
  jointly recorded steps. The first model-action difference uses a maximum
  component difference greater than `1e-6`. Exact executed-action differences
  are listed separately and include float32 rounding.
- Native initially ran two diagnostic cases each for Spatial and Object, then
  added seven and four respectively. Each case has one diagnostic result after
  combining these batches. `native_batch` identifies its batch of origin.

## Experiment configuration

[RUNS.md](../reproduction/RUNS.md) lists the recorded hardware and runtime versions,
eight full-evaluation commands, and ten diagnostic batch commands.
[PROVENANCE.json](../reproduction/PROVENANCE.json) summarizes the experiment metadata,
coverage checks, diagnostic findings, and checksums of the accompanying files.
