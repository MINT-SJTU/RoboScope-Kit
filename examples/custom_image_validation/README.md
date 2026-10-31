# Custom-image validation demos

Custom images are how users evaluate on RoboScope: the image contains the user's `roboscope-kit` `Policy`,
its dependencies, and `/workspace/roboscope-policy.json`. The demos here check that this path works end to
end for different model architectures and benchmark action contracts: writing the `Policy`, baking and
saving the image, and running platform evaluations.

The demos are engineering evidence only. They are not part of the product support matrix, and their scores
are not official RoboScope results.

| Demo | What it covers |
| --- | --- |
| [`starvla/`](starvla/README.md) | The StarVLA Policy used in the RoboScope Kit technical report: one config-driven class for six benchmarks, with an observation stride equal to its action horizon |
| [`xvla/`](xvla/README.md) | One X-VLA image evaluated on several benchmarks, with model-specific observation and action adapters |
