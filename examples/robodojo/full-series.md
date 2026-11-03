# RoboDojo 单任务 50 局评测

用户最终确认的基线为 **`stack_blocks` 单任务 50 局**：ARX-X5、官方
Simate-beta checkpoint、seed 1、固定 layouts 0–49，每个布局运行一次。
不再运行 50 轮。Meta-World 的 50 任务×50 局只是此前参考对话的范围，
不适用于本次 RoboDojo 实验。历史六局属于预实验。

`recipe.toml` 中 `full_episodes=50`、`pilot_episodes=6`。任务、机器人、
checkpoint、550 动作上限、FP32、输入处理、模型历史、相机和视频均不变。
模型 RNG 在这一轮内继续消耗，episode reset 不额外重设随机种子。

## 执行

先完成主 README 的安装与原生最小验证。`PYTHON` 使用已安装 SDK 的 Python
3.11+ 控制环境；本机 `/workspace/robodojo-envs/controller/bin/python`
使用固定 roboscope-kit `2792b6e160f06e24d18499b5ca5f50cca5f3fc31` 的资源测量函数（历史结果在其前身 EvoStudio SDK `3ee774a` 上测得）。
源代码由已提交版本归档，运行期间不更新其目录、模拟器源码或环境。

```bash
# 相同版本的 apply 为幂等操作；旧版本必须用旧工具先回滚。
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch episode-count
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch episode-count # after
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch measurement
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch measurement # after

# OUT 是尚不存在的、仓库外的结果目录，日志同样放在仓库外。
setsid nohup "$PYTHON" -u "$RECIPE/run_series.py" "$ROOT" --env-root "$ENV_ROOT" \
  --profile "$RECIPE/recipe.toml" --seed 1 --episodes-per-round 50 --rounds 1 \
  --num-envs 1 --label baseline50 --output "$OUT" \
  >"$BASE/evidence/baseline50-launch.log" 2>&1 < /dev/null &
echo $!
```

上游 `_task.yml` 对 `stack_blocks` 单独设置 `eval_nums: 25`，即使设置
`EVAL_NUM=50` 也只执行 25 局。`0005-episode-count.patch` 仅将这一配置
改为 50；其他任务保持原值。两组对照都应用此 patch，结果明确标为扩展范围。
启动器在加载 GPU 前验证实际任务上限和 layouts 0–49。首次 b92 运行因这一
上限只完成 25 局，保留为不完整记录；不把它与新进程的 25 局拼接成基线。

`setsid` 分离 SSH 会话的进程组，`nohup` 忽略挂断信号。执行器支持显式重复
轮数，但本次必须 `--rounds 1`。记录 requested=50、unique_layouts=50；
不能把历史被取消的重复任务算入此次完成数。默认 14400 秒超时是防止挂死，
不改变每局动作上限。错误会停止并留下 `complete=false`，不跳过缺失或自动
重试。重新启动使用新的 OUT，保留原中断记录。

`series.json` 在这一轮结束后原子汇总；运行中读取 `round-000.log` 打印的
子运行目录及其原生 `_result.json`。无需持续轮询。取消时向 **series 主 PID**
发送 SIGTERM，执行器给 launcher 最多 120 秒清理各自拥有的模型/仿真进程组，
避免提前杀死 launcher 遗留子进程。不要按进程名批量停止其他作业。

## 计时和完整性

- 正式耗时从 evaluation_start 到 evaluation_end，包含 reset、实际执行、
  视频写入/结束及窗口内结果输出，不含模型加载和仿真初始化。
- 全程耗时包含模型加载、初始化、子进程 preflight、结果归档和清理。
  开始前的布局摘要检查、下载及环境安装单列。未启用 autotune；后续开关
  对照复用已有实现，并单列搜索耗时。
- GPU 每 500 ms、cgroup CPU/RAM 每秒采样；RAM 包含容器文件缓存。
- 必须有 layouts 0–49 各一次、完整输入/动作摘要和正确步数上限。失败局
  保留在分母；不稳定、缺失、重复及中断不得默认为完整。
- measurement 记录初始 state/vision/instruction/layout、模型输入动作及
  映射后的控制序列摘要，reset 稳定性/settling、视频结束耗时。计时有嵌套，
  不能相加为互斥阶段。候选与基线使用同一测量版本，不能混用旧六局计时。
- 完成后在计时外运行 `verify_videos.py "$OUT"`，校验全部 150 个视频及
  `executed_steps + 1` 解码帧数；上游可能记录 finalize 错误后继续，因此
  退出 0 本身不足以验收。

优化在独立 checkout 准备，不改变正在运行的基线。后续候选仍使用同一个
50 局范围；需要重复验证时单独报告每组结果，不把多组相加成扩大套件。
