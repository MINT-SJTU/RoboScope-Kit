# 可选视频写入重叠实验

状态：代码与本机 CPU/真实 ffmpeg 验证通过，**尚无 GPU 评测加速结论**。
完整基线运行在独立冻结版本；该补丁不应用到正在运行的基线目录。

上游已经逐帧流式编码，没有缓存整局。此次仅把阻塞的 ffmpeg stdin 写入移到
独立线程，尝试与后续仿真重叠；不改变相机渲染、控制循环、模型输入或编码质量。
每相机最多排队两帧，加上正在写的一帧；生产者先生成独立 `bytes`，因此不引用
下一次 capture 可能覆盖的数组。队列满时等待，不丢帧、不覆盖、不跳过失败。
640×480 RGB 三路的排队+写入载荷最多约 7.91 MiB，另有调用栈、pipe 和编码器
自身缓冲；不能把这个数当作总进程内存。

FIFO 写入线程在 close 前完全排空，仍使用原 ffmpeg 参数（libx264、CRF 23、
yuv420p、25 FPS）。队列无进展或排空超过 120 秒报错；abort 先终止编码器，
再等待写入线程，避免关闭正在使用的 pipe 导致死锁。上游 finalize 的异常处理
会继续评测，因此完成后必须运行下文的视频检查；仅退出 0 不足以接受结果。

## 应用、检查、回滚

先创建独立源码 checkout，按 recipe 完成固定源码/资产/依赖准备及测量 patch，
使用已提交版本的工具。`async-video` 只修改 `utils/save_file.py` 并新增
`utils/recipe_video_pipe.py`，与 measurement、parallel、依赖 patch 无文件重叠。
它不会修改 `XPolicyLab/policy/Simate_beta` 或 SDK。

```bash
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch async-video # before
"$PYTHON" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch async-video
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch async-video # after

# 先做匹配的小范围验证，再完整复测；仅显式 flag 才启用后台写入。
"$PYTHON" "$RECIPE/run_eval.py" "$ROOT" --env-root "$ENV_ROOT" \
  --mode measured --seed 1 --episodes 50 --label video-on --async-video

# 在评测结束后执行，开销单列在正式评测之外；替换为打印的 evidence 目录。
"$PYTHON" "$RECIPE/verify_videos.py" /path/to/completed/run
"$PYTHON" "$RECIPE/compare_runs.py" /path/to/baseline/run /path/to/video-on/run

# 所有评测进程退出之后回滚，重复执行也安全。
"$PYTHON" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch async-video
"$PYTHON" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch async-video # before
```

原生对照必须回滚此补丁；launcher 拒绝未知 runtime 改动。`run_eval.py` 会清理
继承的 `ROBODOJO_ASYNC_VIDEO`，只由 `--async-video` 设置为 1。
`run_series.py` 同样支持该参数，且任务/布局/轮次/并发须与完整基线匹配。
不能在同一 GPU 的基线运行期间启动候选测量或切换其代码。

## 验证与待测

本机 Python 3.13.9 / ffmpeg 7.1，使用同步和异步两种路径分别编码 30 个
640×480 RGB 帧、25 FPS。输入使用非连续视图且 append 后立即覆写，解码后
两路 RGB 字节摘要完全一致。另验证 200 帧 FIFO、两帧队列上限、超时、broken
pipe、short write、阻塞中 abort、重复 close/abort，以及补丁重复应用/回滚。
这些是输出与清理正确性检查，不是 RoboDojo GPU 性能结果。

待完整基线完成后，先做 off/on/on/off 小范围验证，再进行相同单任务 50 局对照。
每局必须有三个成功/失败标签正确的视频，640×480、25 FPS，实际解码帧数为
`executed_steps + 1`，不得有临时/额外文件。检查工具也可接收完整 series 目录。
分别报告视频等待/结束、正式吞吐、实际 steps/s、成功率变化和包含加载的总耗时；
初始图像差异、动作轨迹差异单列。编码已经与仿真有部分重叠，且复制和线程也有
成本，收益可能很小甚至为负；不预先宣称达到 suica/evostudio-evaluation-kit#11 的 2× 目标。
