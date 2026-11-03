# Runtime patch 应用、检查与回滚

只通过已提交的本目录 patch 和 `apply_runtime_patch.py` 修改私有/共享源码。
源码版本统一由 `recipe.toml` 固定，文件范围直接从 `.patch` 读取。
工具在临时目录中用固定版本的 Git 原始文件重建 patch 前后内容，再逐字节检查
目标文件，拒绝部分应用、未知改动或错误版本；apply/rollback 可重复执行。
新增文件在应用前必须不存在。不另行提交逐文件哈希 JSON 清单。

| Patch | 用途 | 是否改变评测语义 |
| --- | --- | --- |
| `0000-isaac51-deps.patch` | 修正固定 IsaacLab 的 Starlette 声明，与上游 5.1 安装器及 Isaac Sim 依赖一致 | 仅包元数据；不改仿真、模型或任务 |
| `0001-measurement.patch` | opt-in JSONL 计时；添加实际动作数、上限和终止字段 | 不改图像、动作、RNG、reset、成功检查或视频 |
| `0003-parallel-autotune.patch` | 固定布局分片、按 evaluation reset、复用已固定 SDK autotune 的运行入口 | 并发改变共享 RNG 消费顺序，需独立正确性对照；模型代码不改 |
| `0004-async-video.patch` | 可选的有界后台视频 pipe，详见专用文档 | 保持帧顺序和编码参数，完整输出另行校验 |
| `0005-episode-count.patch` | 仅将 `stack_blocks` 的 `eval_nums` 从 25 改为 50 | 扩大单任务布局数量；不改每局语义，不属于加速 |
| `0006-unpaced-render.patch` | 可选的 render 墙钟限速控制，带固定时间步守卫 | 保持物理/相机 fence；重复对照无稳定整体收益，保持关闭 |
| `0007-physics-threads.patch` | 可选的固定 PhysX CPU 工作线程上限，启动参数与生效检查 | 不改物理参数/后端或 autotune；线程顺序可能影响非确定性，需逐局对照 |
| `0008-control-state.patch` | 同次双臂控制复用展开及 CPU 状态复制优化 | 不跨物理步缓存，默认关闭，完整成功率未验收 |

parallel 为可选 suica/evostudio-evaluation-kit#11 实验，完整命令见 [autotune 对照](../autotune.md)。必须先完成
原生/measurement 对照及上游双环境参考，再应用 parallel；退出全部进程后首先
回滚 parallel，才回滚 measurement。parallel 与 measurement 修改不同文件；
每次运行都检查两份 patch 的完整文件状态。主 `run_eval.py` 拒绝带 parallel patch 的运行，
避免将分片实验误记为原生基线。

## Isaac Sim 5.1 主路径

1. `prepare.py sources/assets/model/verify` 准备并验证原始输入。
2. 应用 `isaac51-deps`，再安装模拟器/模型环境。
3. `prepare_runtime_assets.py apply` 调用上游路径生成器。工具将原始机器人 YAML
   备份到外部 `ROOT/../evidence/runtime-assets/`，检查生成字节，支持回滚。
4. 应用 `episode-count` 声明 50 局范围；渲染预检通过后，用 measurement 的 **before** 状态跑原生入口。
5. 应用 measurement，以完全相同的设置跑测量入口并对照。

```bash
PYTHON=/path/to/python3.11
RECIPE=/path/to/roboscope-kit/examples/robodojo
ROOT=/autodl-fs/data/robodojo/RoboDojo

"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch isaac51-deps
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch isaac51-deps # after
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch episode-count
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch episode-count # after
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
# 此处按主 README 安装环境、路径生成、渲染预检并跑原生样例。
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # after
```

`episode-count` 与其他 patch 修改不同文件；在所有原生、测量和候选运行中
保持相同状态。上游 `EVAL_NUM` 使用 `min(请求值, 任务上限)`，不能单靠该
环境变量扩展到 50 局。当前 profile 明确要求此 patch，串行与已有 autotune
入口都在启动 GPU 前检查有效上限和所需布局。旧 profile 没有此项时也会检查
真实上限，避免再把 25 局正常退出误以为完成 50 局。

依赖 patch 变更后，须按主 recipe 重建对应私有环境的可编辑包元数据并通过 `pip check`。不提交或自动执行历史环境的包卸载修复。

## 计时口径

measurement 记录模型导入/加载、SimulationApp、场景初始化、每局 reset/执行及
模型服务调用。计时包含主机侧等待，不是 CUDA kernel profiler。首次推理 warmup
包含在执行阶段，可从第一条 action-call 计时查看。每局动作数只计真正执行动作，
不包含未消费的预测或物理子步。

每 500 ms 采样 GPU；总墙钟包括模型启动、仿真启动、评测和子进程关闭。正式窗口
从 evaluation_start 到 evaluation_end，包含 reset、逐步执行和该窗口内的结果
输出。子阶段可能嵌套，不能相加。JSONL 先写本地 `--metrics-root`，结束后归档到
共享证据目录，归档耗时单独列出。准备、下载、安装不混入评测墙钟。

## 回滚

CPU 工作线程候选的应用、显式对照和回滚见 [线程实验](../physics-threads.md)。
它与 measurement 修改不同文件；运行退出后先回滚候选，再回滚测量 patch。

另有默认关闭的 [视频写入实验](../async-video.md)，由
`0004-async-video.patch` 归属。只在独立候选 checkout 应用，
基线期间不应用；它与其他 patch 的文件范围分离，命令及逆序回滚见该文档。

评测进程全部退出后，按相反顺序回滚；不要删除外部备份。

```bash
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch episode-count
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch episode-count # before
"$PYTHON" "$RECIPE/prepare_runtime_assets.py" rollback "$ROOT"
"$PYTHON" "$RECIPE/prepare_runtime_assets.py" check "$ROOT" # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch isaac51-deps
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch isaac51-deps # before
"$PYTHON" "$RECIPE/prepare.py" verify "$ROOT"
```

回滚依赖 metadata patch 后，若继续使用原环境，需重建对应包元数据；原上游
Starlette 冲突会恢复。建议保留或删除整个私有环境，不能把已回滚源码与旧元数据
组合当作验收环境。

不要对未知改动使用强制 reset 或绕过内容检查。先保留现场，再在新目录从固定
源码重新准备。patch、必要运行配置和工具属于仓库交付；日志、JSONL、视频、安装
包和源码工作目录属于仓库外实验材料。

更新本工具前先用旧版本回滚已应用的 patch；更新后再按上述顺序应用。
2026-10-26 的清理将校验清单读取改为读取 patch，原先的实验结果仍对应报告中的
历史提交。多副本入口的变化及应用步骤见 [autotune 对照](../autotune.md)。

## 后续 runtime 候选

- `0008-control-state.patch`：归属名 `control-state`，同次双臂控制共享展开结果并避免 CPU 状态的重复复制；[应用与回滚](../control-state.md)。每个物理步仍读取新状态，默认关闭。
- `0003-parallel-autotune.patch` 还包含默认关闭的观察回执等待/FIFO；每连接保持单个在途 RPC，保留全局模型锁和 history 语义；[开关、检查与回滚](../observation-runtime.md)。不修改模型 forward 或随机流分配，不重写 SDK autotune。
- `0001-measurement.patch` 增加锁等待、执行器派发、方法体及返回循环的分项计时，以及显式启用的因果 trace。仅做诊断，不计为加速。

升级时必须先使用原版本工具回滚 parallel，再回滚 measurement，确认 before 后应用新版本。其他候选分别独立应用和逆序回滚；不叠加未经验证的组合。所有相关进程退出后操作。历史对照仍按其当时提交解释，当前 patch 不冒充历史原始字节。

必要输入配置为 TOML；patch 范围及摘要由工具从提交的 diff 与固定源码计算，不提交派生 JSON 清单。原始输出不属于 PR 内容。
