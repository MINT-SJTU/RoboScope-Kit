# 固定六局的并行与 autotune 对照

六局是历史预实验和搜索子集。当前正式范围为单任务 50 局；完整顺序及命令见
性能对照协议。
新入口可用 `--output` 指定尚不存在的证据目录。更新 patch 前用旧版本工具
回滚旧 parallel，检查 before 后再应用新版本，不覆盖正在运行的源码。

这是 [suica/evostudio-evaluation-kit#11](https://github.com/suica/evostudio-evaluation-kit/issues/11) 的可选实验入口，主 recipe 的验收仍使用单环境原生/测量对照。
复用 roboscope-kit 的 SDK autotune（`two-phase` 策略），固定 `2792b6e160f06e24d18499b5ca5f50cca5f3fc31`，不另写搜索算法。
下文的历史结果是在 roboscope-kit 前身 EvoStudio SDK
`3ee774ae2575255cc7077b88f8f2df388ba9d152` 上测得的；roboscope-kit 的 `two-phase` 沿用同一实现，但当前 pin 尚未重新跑 GPU 验证。
控制器需要 Python 3.11+；仿真和模型沿用主 README 的私有环境。

## 范围和正确性约束

- 固定 Simate-beta / stack_blocks / arx_x5、seed 1、layouts 0–5、550 步上限；
  每组均保留原生相机、视频、FP32 和 10 步 flow 推理。
- `off` 使用 K=1/G=1；`on` 让已有控制器按其搜索规则测试 K=1、2、3，G 默认上限为 1。
  K 为独立仿真进程数，每进程一个环境；G 为模型服务数。这是当前资源范围的
  有界实验，不宣称搜索了所有并行组合。`--maximum-workers 1|2|3` 可收紧上限；
  默认上限 3 是本次 49,140 MiB 机器的实验范围，不是 24 GB 卡的建议值。单个
  仿真进程约 8 GiB、模型约 15 GiB，应按实际容量选择；SDK 仍可提前拒绝候选。
- 每个候选都完整跑同一组六局。保留 SDK 的资源、OOM、失败、预算检查；
  成功率下降容忍度为 0。六局的门限只是工程筛查，不能证明统计等价。
- 分片按 `range(episodes)[worker_index::K]` 分配，保持原始 layout ID；结束后
  检查每片和合集均无缺失、重复或替换。异常记录为不完整，并清理本次子进程组。
- 共享模型时使用其已有 `reset_evaluation` 接口，只清理当前 evaluation 的
  历史；模型侧 forward、随机数、精度、chunk 和预处理代码不改。
- K 改变会改变共享模型 RNG 的消费顺序，并可能改变物理数值结果。因此需报告
  每局成败、步数、终止和重复测量；不能只看两组成功总数相同。

## 可选多副本对照（2026-10-26）

历史 b94 实验在入口和 SDK bounds 中固定了 G=1，未比较 G=2/3，不能解释为
autotune 已证明单副本最优。b96 的同一 `0003-parallel-autotune.patch` 支持
`--maximum-replicas 1|2`（默认 1），把副本维度交给原有 SDK 搜索，未新增调参算法。
每个副本使用同一 checkpoint/seed，独立端口和进程；worker 固定连接到
`worker_index % G`，每个副本至少服务一个 worker。模型实现及 scoped reset 不变。
独立副本拥有独立 RNG 流，轨迹不保证与共享服务相同，需要逐布局对照。

为了隔离 G 的影响，可用 `fixed` 直接测指定 K/G；它调用同一测量入口、没有搜索
开销，不是 autotune on 的结果。以下命令在既有环境空闲、上方 patch 应用完成后
运行。更新前先用旧工具回滚旧 parallel，确认 before，再用已本地提交的新工具
应用 parallel。不要在运行中的源码上切换版本。

```bash
for G in 1 2; do
  "$CONTROLLER" "$ROOT/recipe_parallel_runtime.py" "$ROOT" \
    --recipe-dir "$RECIPE" --profile "$RECIPE/recipe.toml" --sdk-root "$SDK" \
    --env-root "$ENV_ROOT" --simulator-env "$ENV_ROOT/simulator" \
    --mode fixed --workers 2 --replicas "$G" \
    --maximum-workers 2 --maximum-replicas 2 --episodes 6 \
    --output "$BASE/evidence/replicas-k2g${G}-NEW"
done
# 如需搜索副本维度，沿用原控制器：
# --mode on --maximum-workers 2 --maximum-replicas 2 --episodes 50 --probe-episodes 6
```

历史加载后单模型显存 13.11 GiB；K3/G1 正式峰值 39.07 GiB，K2/G1 探测峰值
30.75 GiB。加一个同样大小副本的粗估分别为 52.18 / 43.86 GiB，尚未计入额外
推理峰值。这张 49,140 MiB 卡只先测 K2/G2，不启动预计超容量的 K3/G2。
这里的 4090 有约 48 GiB 显存，不能把结果当作普通 24 GiB 4090 的容量结论。
按目标机器收紧 bounds；启动失败或 OOM 保留完整日志，清理本次所有子进程。

GPU 总量仍每 500 ms 采样；新增每 5 秒采样各模型/仿真 PID 的显存，并记录
各模型加载后显存、独立加载时间。CPU/RAM 每秒读取 cgroup；RAM 包含文件缓存和
容器其他进程，不等于模型常驻集。原始样本、进程 PID、端口及运行 JSON 均留仓库外。
完整结束后沿用本页回滚步骤；性能结果另行记录，不混入 b94 结果。

本次六局诊断：K2/G2
完整运行，峰值显存 44.09 GiB。正式 steps/s 5.33→6.63，GPU 均值
32.12%→43.45%；但两组为 3/6→5/6 成功，负载更均衡且轨迹不同，不能将
观测的 1.2419× 全归因于副本增加。容量验证通过，默认配置继续保持 G1；
没有用该短测替代固定 50 局验收。

## 应用顺序和运行

先完成主 README 的原生六局，再应用 measurement。所有源码修改只通过已提交
patch 工具。下列路径按机器替换；SDK checkout 与 recipe checkout 分开保存。

```bash
BASE=/autodl-fs/data/robodojo
ROOT="$BASE/RoboDojo"
RECIPE=/path/to/roboscope-kit/examples/robodojo
SDK="$BASE/roboscope-kit-2792b6e"
ENV_ROOT=/workspace/robodojo-envs
PYTHON="$ENV_ROOT/policy/bin/python"

git clone https://github.com/mint-sjtu/roboscope-kit.git "$SDK"
git -C "$SDK" checkout --detach 2792b6e160f06e24d18499b5ca5f50cca5f3fc31
# 如果 checkout 已存在，检查 revision 后复用，不覆盖本地修改。
env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_DEFAULT_INDEX \
  UV_PROJECT_ENVIRONMENT="$ENV_ROOT/controller" uv sync --locked --no-dev \
  --project "$SDK/packages/roboscope-kit" --python 3.11
CONTROLLER="$ENV_ROOT/controller/bin/python"
"$CONTROLLER" -c 'import sys; from roboscope_kit.runtime.benchmark import autotune; print(sys.version, autotune.__file__)'

"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch episode-count
# 上游自身支持的单进程双环境参考，必须在 parallel patch 应用前运行。
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --env-root "$ENV_ROOT" \
  --mode measured --seed 1 --episodes 6 --num-envs 2 --label upstream-vector-two

"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch parallel
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch parallel # after
export PYTHONPATH="$SDK/packages/roboscope-kit/src:$RECIPE:$ROOT"
for MODE in off on; do
  "$CONTROLLER" "$ROOT/recipe_parallel_runtime.py" "$ROOT" \
    --recipe-dir "$RECIPE" --profile "$RECIPE/recipe.toml" --sdk-root "$SDK" \
    --env-root "$ENV_ROOT" --simulator-env "$ENV_ROOT/simulator" \
    --mode "$MODE" --episodes 6 --probe-episodes 6 --maximum-workers 3 \
    --candidate-timeout-s 1800 --tune-budget-s 3600
done
```

`--vulkan-icd`、`--metrics-root` 可指定主机路径。默认搜索 NVIDIA ICD 的两个
常见目录。控制器拒绝错误源码 revision、未知修改、模型修改、错误 patch 摘要
或与 profile 不一致的模拟器/Python/PyTorch。无需修改通用 SDK 源码。

## 结果和计时

运行输出位于 `ROOT/../evidence/autotune/<off|on>-<id>/`。`experiment.json`
记录选定 K/G、搜索耗时、含搜索总耗时和正式运行路径；`selection/` 保存已有
SDK 的候选及拒绝原因。每个候选/正式运行的 `summary.json`、`result.json`、
worker 日志和 GPU 采样保留在仓库外。

正式评测窗口从最早 worker 的 evaluation_start 到最晚 evaluation_end，包含
reset、执行、窗口内输出及尾部不均衡；不是各 worker 时间之和。总墙钟还包含
模型加载、仿真启动和清理。`on` 的总成本须再包含本次所有搜索候选，不能只报
选中候选的执行时间。准备/下载另列。GPU 每 500 ms 采样；CPU 使用 cgroup
累计计数，不能据低 GPU 利用率直接断言瓶颈。

比较 serial、上游 vector-two、autotune off/on 时分别列成功次数/率、实际步数、
两种墙钟、episodes/s、steps/s、GPU 均值、显存均值/峰值和同口径加速比。
并发模式步数可能变化，需同时看 episode 吞吐量和 steps/s。当前入口向既有
控制器提供的候选吞吐信号是实际 steps/s；suica/evostudio-evaluation-kit#11 的验收仍使用完整 episodes/s，
不能把控制器选中某候选直接等同于达到 2×。`compare_runs.py`
默认用于相同环境数的原生/测量对照；跨并发配置须显式传入
`--allow-topology-change`，仍检查相同 profile、源码、测量版本、seed 和布局合集，
并按 layout ID 报告输入、动作、步数与终止差异，不能绕过其他配置一致性检查。
重复完整 off/on 才能进一步观察波动。

## 回滚

等本次全部进程退出，先回滚 parallel，再按主 patch 文档回滚 measurement。
不要在仍运行的模型/仿真进程中切换源码。

```bash
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch parallel
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch parallel # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
```

patch 不改用户模型，也不发布官方镜像。当前有界配置和门限属于实验设置；实际
收益、失败候选和限制以原始实验报告为准（不随仓库发布，仅作参考）。


## 各阶段实测结论

2026-10-24 原生 5.1 off/on 报告：
K=2 候选 2/6，低于本次 K=1 候选 3/6，控制器拒绝 K=2 并停止扩展，未测试
K=3，最终选 K=1。off/on 正式组均 4/6，正式窗口 531.049 / 535.972 s，
含搜索实验总耗时 597.953 / 1,863.942 s。本轮未实现加速或 2× 目标，也不能
用六局证明统计等价；不要把此可选入口当作已经验证有效的默认加速配置。

后续固定 50 局完整对照
使用 b94 patch 和同一 SDK autotune，K1/K2/K3 探测各 4/6 成功，最终选中
K3/G1。off/on 正式组 27/50→29/50 成功，正式 episode 吞吐 2.1948×、实际
steps/s 2.1405×，计入 27.50 分钟搜索后整体 episode 吞吐 1.3426×。完整
命令见该报告，正式 `--episodes 50`，探测仍为 6 局；原始证据保留在仓库外。
开启组峰值显存 40010 MiB，不能套用到普通 24 GB 卡。两组未采样系统 RAM，
逐布局图像/动作也不等价，不能声称成功率已被证明不受影响。实验后 parallel
已回滚，入口保持可选。
