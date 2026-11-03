# RoboDojo / Simate-beta 仿真评测 recipe

本目录交付 [suica/evostudio-evaluation-kit#10](https://github.com/suica/evostudio-evaluation-kit/issues/10)（原始实现见 [suica/evostudio-evaluation-kit#31](https://github.com/suica/evostudio-evaluation-kit/pull/31)） 的独立 recipe 与可运行样例，直接连接上游 RoboDojo 仿真和官方 XPolicyLab 模型服务。固定 Isaac Sim 5.1.0、官方 Simate-beta checkpoint，`stack_blocks / arx_x5 / seed 1`，当前完整范围为 50 个布局各一次；不是 MT50 或 RoboDojo 全任务套件。

文中的性能数字来自原始实现的内部实验（roboscope-kit 前身 EvoStudio SDK `3ee774a`），仅作参考，尚未在 roboscope-kit 上重新验证；性能报告不随仓库发布。并行入口复用既有 SDK autotune；观察 FIFO、控制状态复用、视频、render、线程候选均默认关闭。异步观察的 50 局成功率下降尚未完全定位，短测不能替代完整验收。这里不注册平台 benchmark、SDK Policy 或官方镜像。

## 1. 支持范围与固定输入

| 项目 | 本 recipe 声明的范围 |
| --- | --- |
| 模型 | 官方 Simate-beta 多任务 FP32 AE checkpoint，无需训练 |
| 仿真任务 | 仅 `stack_blocks`：将三个不同纹理积木堆起来，不要求颜色顺序 |
| 机器人 / 动作 | `arx_x5` 双臂，14 维 joint 动作 |
| 最小验证 | seed 1，layout 0，1 episode |
| 声明范围的完整验证 | seed 1，layouts 0–49，50 episodes，单进程单环境 |
| 历史预实验 | seed 1，layouts 0–5，6 episodes |
| 上限 / 成功 | 每局原生 550 个实际动作；三块堆叠且机械臂回位 |
| 仿真 / 观察 | physics dt=0.004 s，观察/控制名义频率 25 Hz |
| 模型设置 | FP32，10 flow steps；预测 50 个动作，执行前 10 个 |
| 记忆 | 模型保留 25 帧历史，每 20 个动作采样，AE grid 4×4 |

当前 50 局仍是单任务评测，不是 MT50、全部 RoboDojo 任务或完整官方排行榜
协议。六局历史结果保留为预实验，不能当作当前完整验证。

[recipe.toml](recipe.toml) 是准备和运行工具实际读取的配置，保留源码版本、下载源
和评测参数，属于运行输入，不是实验留档。逐文件校验清单、
运行输出与原始实验材料均不提交，patch 的检查方式见 [patch 文档](patches/README.md)。

固定输入如下：

| 来源 | 固定 revision |
| --- | --- |
| [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | `f57e1633ffed59761bc8e568cfe7530c4b33381c` |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `5d0be1499634d3176258c8f2e31f5fe816497b0a` |
| [IsaacLab fork](https://github.com/yuechen0614/IsaacLab) | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| [CuRobo fork](https://github.com/yuechen0614/curobo) | `d17b54ce32cba095c0b000c4c58777075d11de0e` |
| [官方资产与 checkpoint](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/35efbc7dedfdbeeb6e95fb749bd885d73d483e41) | `35efbc7dedfdbeeb6e95fb749bd885d73d483e41` |

Checkpoint 相对路径：
`ckpt/RoboDojo/Simate-beta/RoboDojo-sim-arx_x5-joint-0/`，四个文件合计
13,502,053,373 bytes。模型也与作者的
[AE release](https://huggingface.co/SIPAILab/sipai-robodojo-eval-ae/tree/0738aab54e0c873db76d4d2aec7a467ca1b04120)
一致；不要误换成 pooling-only 权重。

| 文件 | SHA-256 |
| --- | --- |
| `model.safetensors` | `11424ad091c37d6192d8fb18228c11b5f5d1fff30436ad8d20bbfebc89d00e3a` |
| `tokenizer.model` | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |
| `LICENSE` | `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |
| `README.md` | `42100c3470106a33c481f2eba0ca43feb95baf36e416178b5b393562ceab23ab` |

15,365 个资产文件合计 41,269,513,111 bytes。配置同时固定资产/checkpoint 文件
清单摘要；下载器先验证官方 revision 和清单，再逐文件检查大小及 SHA-256 或
Git blob SHA-1，校验失败不会接受下载文件。保留上游许可证。

## 2. 环境和安装

需要 Linux x86_64、可用的 NVIDIA RTX/Vulkan 渲染、Git、CUDA 编译工具链，以及
安装系统依赖的 root 权限。Isaac Sim 5.1 的官方最低配置包含 RTX 4080/16 GB、
32 GB RAM，Linux 验证驱动为 **580.65.06**，见
[官方要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。
本轮原生六局峰值 23,594 MiB，上游双环境峰值 23,934 MiB，24 GB 显存余量
有限。当前 AutoDL 报告 49,140 MiB 显存、20 CPU 配额、96 GiB RAM 上限；
实验室 D 机为 24,564 MiB 显存和 32 GB RAM，原生一局峰值 22,757 MiB。为资产、权重、仿真包和
缓存准备至少 150 GB 可用空间。这是工程容量建议，不是所有任务的保证。

准备工具使用 Python 3.11+；主路径的模拟器和模型各有一个 Python 3.11 私有环境。
固定 torch 2.7.0+cu128、Isaac Sim 5.1.0.0。两份 `constraints-*.txt` 是安装器
主动使用的依赖约束，不是纯留档清单；可编辑源码依赖由 Git revision 固定。
系统 apt 包仍随 Ubuntu 安全更新，不声称整个 OS/编译产物逐位可复现。

Isaac Sim 是 NVIDIA 的专有软件，受 [NVIDIA Omniverse License Agreement](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)
约束。本 recipe 不会替你接受该 EULA：阅读后自行执行 `export OMNI_KIT_ACCEPT_EULA=YES`，
`setup_environments.py`、`run_eval.py` 和并行入口在未设置时直接退出。是否设置
`PRIVACY_CONSENT` 由你自己决定，recipe 不设置。

以下命令在新 Linux 环境中执行。替换 `KIT`、`BASE`、`ENV_ROOT`、`CACHE_ROOT`
为自己的绝对路径；模型权重由本 recipe 下载，与 Meta-World 容器的 checkpoint
配置无关。保留 seed、任务、机器人和模型设置即可复现本声明范围。

```bash
python3 -m pip install --user 'uv==0.11.14'
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11.15
PYTHON=$(uv python find 3.11.15)

KIT=/path/to/roboscope-kit
RECIPE="$KIT/examples/robodojo"
BASE=/autodl-fs/data/robodojo
ROOT="$BASE/RoboDojo"
ENV_ROOT=/workspace/robodojo-envs
CACHE_ROOT=/workspace/robodojo-cache

"$PYTHON" "$RECIPE/prepare.py" sources "$ROOT"
"$PYTHON" "$RECIPE/prepare.py" assets "$ROOT" --workers 8
"$PYTHON" "$RECIPE/prepare.py" model "$ROOT" --workers 4
"$PYTHON" "$RECIPE/prepare.py" verify "$ROOT" --workers 8

# 先提交 patch 工具，再用它修改共享源码；不手改运行目录。
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch isaac51-deps
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch episode-count
"$PYTHON" "$RECIPE/setup_environments.py" simulator "$ROOT" \
  --python311 "$PYTHON" --env-root "$ENV_ROOT" --cache-root "$CACHE_ROOT"
"$PYTHON" "$RECIPE/setup_environments.py" policy "$ROOT" \
  --python311 "$PYTHON" --env-root "$ENV_ROOT" --cache-root "$CACHE_ROOT"
"$PYTHON" "$RECIPE/prepare_runtime_assets.py" apply "$ROOT"
"$PYTHON" "$RECIPE/prepare_runtime_assets.py" check "$ROOT"
```

安装器调用固定上游脚本的安装函数，跳过其移动分支的 submodule 更新；分别验证
`pip check`。`isaac51-deps` 只将 IsaacLab 的 Starlette 包元数据对齐到上游
5.1 安装器固定的 0.45.3，原生与测量入口使用同一修正。模拟器保留 Isaac Sim
要求的 `idna==3.10`，不安装早期探索环境遗留的 `httpx2`。复用旧环境时按
[补丁说明](patches/README.md)清理遗留包并刷新元数据。模型侧代码不改。

固定上游的 `stack_blocks` 默认上限是 **25 局**，`EVAL_NUM` 只能降低上限。
本 recipe 用 `episode-count` 将该任务配置扩展为 **50 局**，原生与测量入口
均使用同一配置；不改其他任务、布局或每局参数。这是声明的单任务扩展实验，
不是上游默认套件。启动前检查有效上限及布局文件，拒绝静默截断。

可选下载方式：`--endpoint https://hf-mirror.com`；有 aria2c 时，对模型命令加
`--downloader aria2`；`--transport modelscope` 只替换传输源，仍按锁定的官方
哈希验收。`--bundles /path/to/bundles` 可读取同名 Git bundle；原始清单可通过
`--inventory-file` 离线传入，清单摘要必须吻合。所有下载收据、安装日志和耗时
写入 `BASE/evidence/`，不进入 Git。

## 3. 先检查渲染器

必须先通过真实 RTX 启动检查，CUDA 张量能运行不等于渲染器可用。按自己的机器
修改 NVIDIA ICD 路径；下面示例适用于已检查的 AutoDL 容器。两条标记和退出 0
同时出现才算通过；不要通过关闭图像渲染绕过检查。

```bash
env -u CUDA_VISIBLE_DEVICES \
  ACCEPT_EULA=Y \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  "$ENV_ROOT/simulator/bin/python" -u - --allow-root <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'multi_gpu': False, 'active_gpu': 0})
print('RENDERER_READY', flush=True)
for _ in range(3):
    app.update()
print('RENDERER_UPDATES_COMPLETE', flush=True)
app.close()
PY
```

两条标记指 `RENDERER_READY` 与 `RENDERER_UPDATES_COMPLETE`。5.1 默认快速关闭会
直接结束进程，不能要求 `app.close()` 之后的 Python 输出。普通 Ubuntu 主机的
ICD 常位于 `/usr/share/vulkan/icd.d/nvidia_icd.json`，按实际文件调整上述命令。
Launcher 会在这两个目录查找 NVIDIA ICD，也可用 `--vulkan-icd /absolute/path.json`
显式指定，无需修改系统 Vulkan 配置。

更换机器后先重复 RTX 检查，再启动真实模型评测；CUDA 可用不代表渲染器一定可用。环境排障记录不作为本 PR 的 runtime 加速贡献。

## 4. 原生入口、测量入口和结果

`run_eval.py` 是可运行的串行样例，启动上游 Simate server 与 RoboDojo client，
通过 XPolicyLab WebSocket 连接。`--mode native` 不应用测量补丁；`measured`
使用同一模型、任务和控制循环，只增加计时及结果字段。两者完整命令与配置会写入
各自 `invocation.json`。它检查源码 revision、patch 状态及非预期代码改动，
不同 profile/仿真器的结果不能当作配对运行。

```bash
# 最小验证：原生 layout 0，与测量入口使用完全相同设置。
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --env-root "$ENV_ROOT" \
  --mode native --seed 1 --episodes 1 --label native-smoke
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --env-root "$ENV_ROOT" \
  --mode measured --seed 1 --episodes 1 --label measured-smoke
"$PYTHON" "$RECIPE/compare_runs.py" /path/to/native-smoke /path/to/measured-smoke

# 当前完整范围：同一 profile、checkpoint、seed 和 50 个布局。
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --env-root "$ENV_ROOT" \
  --mode native --seed 1 --episodes 50 --label native-fifty
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --env-root "$ENV_ROOT" \
  --mode measured --seed 1 --episodes 50 --label measured-fifty
"$PYTHON" "$RECIPE/compare_runs.py" /path/to/native-fifty /path/to/measured-fifty
```

`/path/to/...` 替换为启动器打印的 `BASE/evidence/evaluation/<run_id>/`。
`--timeout-s` 控制客户端等待期限，超时计为不完整，不改变每局动作上限。
默认只使用可见的 GPU 0。这里是单任务、单环境参考，不据此声称优于上游并行模式。
输入、输出与 reset 的完整说明见 [model-contract.md](model-contract.md)。

| 输出 | 读取方式 |
| --- | --- |
| `summary.json` | `complete`、请求/完成局数、成功次数/率、layout IDs、分阶段/总耗时及 GPU 指标 |
| `result.json` | 完整上游结果，每局 `success`、部分 `score`；测量模式另有实际动作数和终止原因 |
| `events/*.jsonl` | reset、执行和模型调用计时，供后续 suica/evostudio-evaluation-kit#11 使用 |
| `client.log` / `server.log` | 异常、模型加载、原生动作进度 |
| 上游任务运行目录 | `ROOT/eval_result/RoboDojo/stack_blocks/Simate_beta/arx_x5/<seed>_<label>/<run_id>/`，包含视频 |

完整结果必须满足退出 0、`complete=true`、完成数等于请求数、layout IDs 恰好为
0 至 N−1 且无重复。丢失、重复、替换布局、崩溃或超时都会使运行失败。失败局仍
计入分母；不稳定/放弃记录保留，不把缺失局默认为完成。SIGINT/SIGTERM 会清理
本次创建的子进程并保留失败记录。

单任务情况下，整体与该任务的成功次数/率相同；`score` 是部分完成得分，不能
替代成功率。上游 `eval_time` 是完成局数，不是秒。例如 `completed_episodes=50, successes=29, success_rate=0.58` 表示完成 50 局、29 局成功；并非后续运行的保证值。

## 5. 可选 runtime 及限制

- [完整 50 局运行](full-series.md)、[模型输入/动作/reset 契约](model-contract.md)。
- [并行与既有 autotune](autotune.md)、[观察 RPC/FIFO](observation-runtime.md)、[控制状态复用](control-state.md)。
- [分项 profiling](execution-profile.md)、[视频写入](async-video.md)、[render 限速](unpaced-render.md)、[PhysX 线程](physics-threads.md)。
- [patch 应用、检查与回滚](patches/README.md)。所有候选只在相关进程退出后应用；不手改共享 runtime 或保存镜像。

单次分数相同或接近不能证明正确性等价。模型 RNG 跨 episode 连续消费，原生渲染/物理也存在跨运行差异；必须保留逐布局成败、动作数、终止和输入差异。模型代码和精度/history/chunk 不属于本 PR 的优化范围。

修改任务、机器人或模型后必须重新固定配置并做原生对照，不能继承本任务的成功率结论。安装约束和 recipe 配置由工具实际读取，结果及原始证据在仓库外；性能结论统一写入上面的一个 Markdown，不另外提交证据清单。
