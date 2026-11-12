# Shared SAM3 source

`sam3_http.py` and `sam3_local.py` are unmodified files from
[PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean) commit
`5238f76b2a2937cf461bca833d5d2a3b9b31d0d4`, distributed under the Apache License 2.0;
the upstream [LICENSE](LICENSE) and [NOTICE](NOTICE) are included here. The `sam3_local.py`
file is byte-identical to the version at the example's pinned PyRUA commit `c686120`.

The service binds to loopback by default and loads the same SAM3 checkpoint and segmentation
implementation as the local Policy. See the [PyRUA example](../README.md) for its use.
