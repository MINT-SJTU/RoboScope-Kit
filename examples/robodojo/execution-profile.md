# 单进程执行开销定位

这一步定位旧 50 局基线中尚未细分的执行开销，不修改 parallel/autotune，
不修改 `XPolicyLab/policy/Simate_beta`。保持同一模型、seed、布局、动作映射、
550 步上限、history/chunk、物理推进、相机 fence 和视频参数。

`0001-measurement.patch` 新增可选 `--execution-profile`：在单个 episode 的
`run_eval` 内记录同步调用树的次数、累计主机时间、扣除同线程已测子调用后的
自身时间和最大耗时。记录范围包含观察、动作、控制插值、机器人状态、物理、
场景同步、渲染 fence、相机回读、视频以及 RPC。没有新增 CUDA synchronize，
所以 CUDA 异步工作可能计入后续等待；不能把这些时间称为 GPU kernel 时间。

同步路径的 inclusive 时间存在嵌套；主线程等待 RPC 与后台线程 codec 也有
重叠，不能相加。server 的异步时间单独以 `async/` 标记，只作包含等待的
累计区间解释。服务端统计是全生命周期，模拟器统计只覆盖正式执行；reset
和初始化沿用原 measurement 记录。对照中要分别报告这些范围。

`--python-profile` 额外保存主线程 cProfile，用于深入检查热点。它会增加
开销，不能与未开启的运行直接计算加速比。正式性能验收关闭两个 profiling
选项。诊断输出在外部 `events/execution-profile/`，不提交 JSON 或 `.prof`。

## 更新、运行与回滚

先确认所有本次模型和仿真进程退出、GPU 空闲。用原部署版本工具回滚原
measurement，确认 before；再用已提交的新工具应用。各阶段报告保留各自
measurement 提交，不把新版采样与旧版耗时拼接成同一次测量。

```bash
BASE=/autodl-fs/data/robodojo
ROOT="$BASE/RoboDojo"
OLD=/path/to/previous/recipe
RECIPE=/path/to/committed/examples/robodojo
PYTHON=/workspace/robodojo-envs/policy/bin/python

"$PYTHON" "$OLD/apply_runtime_patch.py" check "$ROOT" --patch parallel # before
"$PYTHON" "$OLD/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # after
# isaac51-deps 和 episode-count 保持已应用状态。
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 \
  --num-envs 1 --episodes 1 --label execution-diagnostic \
  --execution-profile --python-profile
# 定位完成后，使用独立证据目录跑关闭 profiling 的固定六局预实验和 50 局正式对照。
"$PYTHON" "$RECIPE/verify_videos.py" /path/to/completed/run

# 全部进程退出后恢复旧测量版本：
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$OLD/apply_runtime_patch.py" apply "$ROOT" --patch measurement
```

原生入口默认不启用 profiling。设置快照记录实际设备、物理 dt、decimation、
观察间隔、渲染间隔、Kit 墙钟限速及 zero-delay 选项，读取时不修改设置。
统计模块通过 benchmark patch 安装，不在用户推理代码中加速或改变运算。

## RPC 服务端分项与逐请求定位

最新 measurement 在保留原模型锁的前提下区分锁等待、executor 派发、方法体、返回 event loop 的耗时，记录放在释放锁之后，不增加 GPU 同步。计时和 cProfile 仍是诊断开关，两组须保持一致；线程池等待不能直接当作模型计算时间。

可选 `ROBODOJO_CAUSAL_TRACE_DIR` 启用观察/RNG/动作等逐请求诊断材料，体积较大且影响时序，只能指向仓库外新目录。正式吞吐实验应关闭该 trace，不能根据计数一致推断未记录的数值一致。详细结果与局限见 观察阶段报告，运行开关见 [观察 runtime](observation-runtime.md)。
