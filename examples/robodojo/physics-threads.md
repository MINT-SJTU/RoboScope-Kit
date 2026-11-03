# PhysX CPU 工作线程对照

本候选只改变单个仿真进程内部 PhysX CPU dispatcher 的工作线程上限，不改变
仿真进程数量、模型服务副本或现有 autotune。默认关闭；不能把配置的线程数
当成实际活跃核数。物理 dt/后端/solver、相机同步、视频及模型代码均保留。

归属 `0007-physics-threads.patch`，在原有 AppLauncher 前添加显式 Kit 参数，
首次相机采集时检查生效值及 PhysX dispatcher。冲突参数或未生效时报错。
相机等待循环和帧数逻辑不变。当前只支持 recipe 固定的 Isaac Sim 5.1 CPU
环境；`--physics-threads` 可取 1、4、8，为手动固定值，不做搜索或自动选择。

[Kit 配置文档](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.0.3/guide/configuring.html)
说明 `/persistent` 设置可每帧写回用户配置。因此显式启用线程参数时同时使用
`--/app/settings/persistent=false`，禁止本轮覆盖持久化到后续运行；不改变
loadUserConfig。受控线程对照的两组都显式指定线程数，保持此项一致，基准
线程值必须来自先前真实诊断，不能猜测默认值。改变线程执行顺序可能改变
浮点/非确定性轨迹，必须检查逐布局输入、动作、步数及终止条件。

2026-10-27 本机请求 1 线程时首次 reset 挂起，已保存失败并回滚；**不要将 1
作为可用或已验收配置**。8/4 线程单局均完整完成；物理每子步减少 7.27%，execution
每动作仅减少 0.60%，尚不能认定稳定整体收益。后续 8/4/4/8 各六局已按最新要求
取消并回滚，当前不继续线程参数试跑；详见
[runtime 冗余检查](PERFORMANCE_REPORT_2026-10-28.md)。

## 应用、检查和回滚

先确认本轮进程全部退出，GPU 空闲。保留 measurement、isaac51-deps、
episode-count；parallel、async-video、unpaced-render 均保持 before。
不同版本 measurement 先用旧工具 rollback，再用新工具 apply/check。

```bash
BASE=/autodl-fs/data/robodojo
ROOT="$BASE/RoboDojo"
RECIPE=/path/to/committed/examples/robodojo
PY=/workspace/robodojo-envs/policy/bin/python

"$PY" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch physics-threads # before
"$PY" "$RECIPE/apply_runtime_patch.py" apply "$ROOT" --patch physics-threads
"$PY" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch physics-threads # after
"$PY" "$RECIPE/run_eval.py" "$ROOT" --mode measured --seed 1 --num-envs 1 \
  --episodes 1 --label physics-thread-diagnostic --physics-threads 4 --timeout-s 600 \
  --execution-profile --python-profile
"$PY" "$RECIPE/verify_videos.py" /path/to/completed/diagnostic

"$PY" "$RECIPE/apply_runtime_patch.py" rollback "$ROOT" --patch physics-threads
"$PY" "$RECIPE/apply_runtime_patch.py" check "$ROOT" --patch physics-threads # before
```

所有基准、候选均验证完整视频；重复布局不视为独立新场景。用现有 compare_runs.py
检查结果完整性及逐布局差异，报告正式与全程吞吐、资源、成功次数/率及限制。
有稳定收益才进入 50 局验收，未完成验收前不作为默认优化。

patch 工具验证 pinned 源码和完整 before/after，拒绝未知修改和部分应用。
本 patch 修改 render_sync.py 的启动/检查入口并新增 helper；与已有 measurement、
episode-count 不重叠。所有 raw JSON、日志、视频留仓库外；性能阶段记录见
[CPU 诊断报告](PERFORMANCE_REPORT_2026-10-28.md)。
