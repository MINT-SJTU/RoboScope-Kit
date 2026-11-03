# 观察 RPC 与有界 FIFO 候选

本页是 `0003-parallel-autotune.patch` 的可选 runtime 路径，不更换 SDK 搜索算法。同步是默认模式；`--overlap-observation` 启用异步回执等待，队列上限默认 1，显式 `--observation-queue-limit 10` 才启用本次 FIFO10 候选。

完整 50 局单待办候选出现成功率 60%→50%，原因未完全定位；FIFO10 仅三布局诊断。均不能作为成功率无回归的已验收优化。完整指标、失败布局和限制见 [阶段报告](PERFORMANCE_REPORT_2026-10-28.md)。

## 顺序与状态约束

- 提交前取得观察深拷贝，有界保存，不丢帧、合并或下采样。
- 每条连接只有一个在途 RPC；后继请求必须等待并验证前一个响应。前序错误向后传播，失败后的队列请求不得发送。
- 推理、reset、其他模型 RPC 和 close 前排空；中断等待仍保留未完成请求。
- 模型方法仍经过原来的全局锁；history、随机流分配、模型代码和动作队列执行不变。
- 仅支持该固定 recipe 已校验的 `update_observation` 返回 None 契约，不扩展为任意模型客户端语义。

同一 worker 的先后关系受保护，不代表跨 worker 全局 RNG 消费顺序与同步运行相同；这仍是成功率分析的限制。

## 应用与小范围对照

先完成主 README 准备和 [autotune 环境](autotune.md)，保持 isaac51-deps / episode-count 已应用；视频、render、线程和控制状态候选保持 before。退出评测和模型进程后，用原版本工具先回滚旧 parallel，再回滚旧 measurement，保留旧工具供恢复。新工具只接受其固定的原始或已应用状态，不能跳过检查覆盖旧 patch。

```bash
RECIPE=/path/to/roboscope-kit/examples/robodojo
ROOT=/autodl-fs/data/robodojo/RoboDojo
SDK=/autodl-fs/data/robodojo/roboscope-kit-2792b6e
PYTHON=/workspace/robodojo-envs/controller/bin/python
EVIDENCE=/autodl-fs/data/robodojo/evidence
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch parallel
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch parallel # after
export PYTHONPATH="$SDK/packages/roboscope-kit/src:$RECIPE:$ROOT"
# 本例重现三布局诊断口径，两组开同样的 profiling；输出目录必须全新。
export ROBODOJO_EXECUTION_PROFILE=1 ROBODOJO_PYTHON_PROFILE=1
unset ROBODOJO_CAUSAL_TRACE_DIR
"$PYTHON" "$ROOT/recipe_parallel_runtime.py" "$ROOT" \
  --recipe-dir "$RECIPE" --profile "$RECIPE/recipe.toml" --sdk-root "$SDK" \
  --mode fixed --workers 3 --replicas 1 --maximum-workers 3 --maximum-replicas 1 \
  --episodes 3 --output "$EVIDENCE/observation-off-NEW"
"$PYTHON" "$ROOT/recipe_parallel_runtime.py" "$ROOT" \
  --recipe-dir "$RECIPE" --profile "$RECIPE/recipe.toml" --sdk-root "$SDK" \
  --mode fixed --workers 3 --replicas 1 --maximum-workers 3 --maximum-replicas 1 \
  --episodes 3 --output "$EVIDENCE/observation-fifo10-NEW" \
  --overlap-observation --observation-queue-limit 10
```

旧单待办对照使用 `--overlap-observation --observation-queue-limit 1`。`fixed` 搜索耗时为 0；这不是 autotune on/off 对照。50 局历史单待办实验仅开 execution profiling、关闭 cProfile，不能与本例混为一个结果。

结束后按 [完整验证说明](full-series.md) 对 summary/result 路径运行 `verify_videos.py` 与 `compare_runs.py`；保留失败、不完整与逐布局差异。进程退出后逆序回滚：

```bash
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch parallel
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch parallel # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
# 如先前有旧测量版本，最后用其原工具恢复；不要对运行中的源码操作。
```

锁等待、executor 派发、方法体和返回 event loop 的计时属于 `0001-measurement.patch`；记录在释放原模型锁之后完成，不增加 GPU 同步。可选大体积因果 trace 只用于诊断，原始观察/RNG/动作记录留仓库外，不作为运行依赖提交。
