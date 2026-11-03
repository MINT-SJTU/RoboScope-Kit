# 控制状态复用候选

`0008-control-state.patch` 仅修改 benchmark 的 robot manager。双臂在同一次控制调用中复用关节目标展开结果；CPU 状态直接生成独立 NumPy 数组，避免先克隆整个 Tensor 再复制。每次物理步后仍重新读取状态，GPU 路径维持原逻辑，不修改模型、控制频率、动作映射或物理参数。

默认关闭。单局验证 显示控制处理减少约 37%，整局实际 steps/s 仅增加约 1.3%；两组都失败，不构成成功率验收。

先按主 README 准备固定源码与环境，应用 isaac51-deps、episode-count 和 measurement；其他候选保持 before。所有评测进程退出后，使用已提交版本工具：

```bash
RECIPE=/path/to/roboscope-kit/examples/robodojo
ROOT=/autodl-fs/data/robodojo/RoboDojo
PYTHON=/workspace/robodojo-envs/controller/bin/python
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch control-state # before
# 先运行关闭组，输出标签必须未使用。
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 --episodes 1 --label control-off-NEW
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch control-state
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch control-state # after
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 --episodes 1 --label control-on-NEW --control-state-reuse
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch control-state
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch control-state # before
```

`run_series.py` 同样支持 `--control-state-reuse`。入口先清除继承的 `ROBODOJO_CONTROL_STATE_REUSE`，只由显式开关设置，并校验 patch 完整字节；不接受已经应用但没有声明的未知 runtime 修改。若需分项诊断，两组都增加 `--execution-profile`；与 profiling 关闭的性能验收分开报告。视频检查和逐局比较沿用主 recipe 工具。更新 patch 时用原版本工具回滚，不能强制覆盖现场。
