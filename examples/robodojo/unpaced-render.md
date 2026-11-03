# 固定时间步下取消 render 墙钟限速

这是默认关闭的 runtime 候选，归属 `0006-unpaced-render.patch`，只支持本 recipe
固定的 Isaac Sim 5.1 环境。单局诊断发现 Kit 主循环限速开启为 25 Hz，render
平均约 40.16 ms。不能把这个主机时间全部叫作 GPU 渲染时间。

2026-10-26 已完成未开 profiling 的 off/on/on/off 各六局。两次正式 steps/s
配对为 1.1174× / 0.9639×，尚无稳定整体收益，候选保持关闭，不进入 50 局验收。
完整指标、逐布局差异及局部 render 收益见阶段报告。

[Isaac Sim 5.1 官方示例](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html)
将主循环 FPS 限速与 PhysX 频率、timeline 设置分别列出。本候选只在
`BaseEnv.render()` 内暂时设置 `rateLimitEnabled=false`，调用原来的 render
一次，然后在 `finally` 恢复原值。不修改频率值或渲染参数。

启用时要求 Isaac loop runner 处于 manual 模式，其 step size 等于原来的
rendering dt；每次调用后检查物理步计数与 rendering dt 未改变。不满足就报错，
不改时间步来兼容。原来的相机 dispatcher fence、zero-delay 设置、物理推进、
成功判断、模型代码、history/chunk、三路 640×480 / 25 FPS 视频均保留。
取消墙钟等待可能影响非确定性渲染，不能仅凭成功总数相近声称轨迹等价。

## 应用、对照与回滚

先结束本次拥有的模型和仿真进程，确认 GPU 空闲。`measurement`、
`isaac51-deps`、`episode-count` 已按 recipe 应用；`parallel` 和 `async-video`
保持 before。原生入口不设置 `ROBODOJO_UNPACED_RENDER` 时行为不变。

```bash
BASE=/autodl-fs/data/robodojo
ROOT="$BASE/RoboDojo"
RECIPE=/path/to/committed/examples/robodojo
PYTHON=/workspace/robodojo-envs/policy/bin/python

"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch unpaced-render # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch unpaced-render
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch unpaced-render # after
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 \
  --num-envs 1 --episodes 1 --label unpaced-diagnostic \
  --unpaced-render --execution-profile --python-profile
"$PYTHON" "$RECIPE/verify_videos.py" /path/to/completed/diagnostic

# 性能对照关闭 profiling，使用同一 measurement 版本、seed 与 layouts 0–5。
export PYTHONPATH="$BASE/roboscope-kit-2792b6e/packages/roboscope-kit/src:$RECIPE"
"$PYTHON" "$RECIPE/run_series.py" "$ROOT" --rounds 1 --episodes-per-round 6 \
  --num-envs 1 --seed 1 --label unpaced-on --unpaced-render \
  --output "$BASE/evidence/unique-unpaced-on"
# 每组退出后解码全部视频，再切换；off 必须 rollback，并省略 --unpaced-render。
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch unpaced-render
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch unpaced-render # before
```

工具验证 pinned Git revision、完整 before/after 内容，拒绝未知修改和部分应用。
本 patch 与 measurement/episode-count 修改不同文件；仅针对固定 Isaac Sim 5.1 主路径验证，不与其他候选叠加。
