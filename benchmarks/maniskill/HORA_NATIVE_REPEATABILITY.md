# HORA native contact repeatability

On 2026-11-11, source `42e6b24bbde9e26bfe714e92786d2891bfc37d4b` used the
unchanged HORA checkpoint, seed 0, one Worker/Policy replica, vector width one
and the original 300-tick horizon. The strict observation comparison still **fails**;
no quantization or comparison tolerance was introduced.

Twelve independent SDK/native runs completed: three per path for each of levels
2 and 3, containing 3,600 actions and 3,612 observations. Every run retained its
coverage, actual boolean terminal masks and complete horizon. All action hashes,
outcomes, returns and record fields matched across all 30 pair comparisons. No
SDK Worker failed or ran out of memory. Each run had 0/1 success; native horizon
truncation is distinct from an execution timeout.

| Comparison | Pairs | Entirely exact pairs | Differing observation rows across pairs | Action / record differences |
| --- | ---: | ---: | ---: | ---: |
| Native versus native | 6 | 4 | 2 | 0 / 0 |
| SDK versus SDK | 6 | 1 | 6 | 0 / 0 |
| SDK versus native | 18 | 8 | 13 | 0 / 0 |

Level 2 native repeats were exact. Level 3 native repeat 1 differed from repeats
2/3 only at observation 260, independently demonstrating native variation. SDK
variation was confined to observations 212/260. Returns were identical per level:
23.79337501525879 for level 2 and 24.885221481323242 for level 3.

Separate numeric captures covered 15 inputs per path and level: tick 0,
208–216 and 258–262. All 60 captures matched their actual trajectory input/action
hashes. Level 2 captured pairs were exact. Level 3 differed only in state index
70, the fourth finger force-sensor value:

| Tick | SDK float32 | Native float32 | Absolute difference | SDK / native bits |
| --- | ---: | ---: | ---: | --- |
| 212 | 0.0285978801548481 | 0.028597882017493248 | 1.862645149230957e-9 | `3cea461a` / `3cea461b` |
| 260 | 0.034062307327985764 | 0.03406231105327606 | 3.725290298461914e-9 | `3d0b84eb` / `3d0b84ec` |

Both differences are one float32 ULP. The unchanged Policy consumes state[:16];
this joint-position prefix was bitwise equal in all 30 captured pairs. Offline
analysis reproduced every repeated-run hash variant at 212/260 by substituting
only the two actually observed state-70 alternatives. Original records were not
rewritten. Combined with exact hashes elsewhere, this localizes the differences
to a sensor field unused by this Policy.

A separate native diagnostic re-queried `get_fsr_impulse()` 64 times at each
frozen state at 212/260 in both levels, without stepping or calling `get_obs`,
`get_info` or `evaluate`. All before/after joint positions were bitwise unchanged.
At level 3/tick 212, the first read returned norm `3cea461b`; the next 63 returned
`3cea461a`, exactly the SDK/native alternatives above. The underlying x/z impulse
components also differed by one ULP. The other three frozen-state cases were
exact across all 64 queries. Contact-readback variability is therefore directly
observed on the native path, independently of SDK transport.

Pinned SAPIEN 3.0.0b1 uses float atomic additions for body-contact accumulation.
This is consistent with reduction-order variability; the kernel-level cause was
not proven. See the [pinned SAPIEN implementation](https://github.com/haosulab/SAPIEN/blob/d9031a351c39986e51bd0716bc3b51f9a05070d8/src/physx/physx_system.cu#L181-L223).

Raw records, commands, samples, all 64 vectors/norms per frozen state and source
identities remain outside Git, with verified local backups:

- `hora-repeatability-r12-evidence.tar.gz`: SHA256 `c15d176fc8bf90893be61aa5a680c335c33674760fb93a4814a27ac0ad830c8a`.
- `hora-followups-r12-evidence.tar.gz`: SHA256 `f4b274c03cb99db0d1898983d0570a24c71c715520f591b708186c977eb7b7ab`.

The [validation status](VALIDATION.md) tracks current source, coverage and publication.
This borrowed HORA model also does not satisfy intended-task policy acceptance. These findings explain a native numerical floor;
they do not turn the strict hash failures into passes or establish full benchmark
acceptance.
