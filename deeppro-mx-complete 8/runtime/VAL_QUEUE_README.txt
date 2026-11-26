val 六组后台队列（仅 v8，未包含 official/test 阶段）

启动或恢复（在本项目目录执行）：
  python -S runtime/val_scheduler.py start
只生成可审阅命令、不启动：
  python -S runtime/val_scheduler.py plan
单独刷新最终结果汇总（不训练）：
  python -S runtime/summarize_val.py

主容器 dp-mta-v8-val-main-20261126 由 root 启动。调度器只接管监控，
绝不创建/重启主训练。主 run 必须至少完成一次全部 val clip 的正式选权重，
且有 best_model.pth 与 best_selection.json，才放行五个 controls。

控制组：mta_val_s46_{noex,self,nogain,mx,noapp}。
统一 train / seed46 / 32 epochs / batch4 / val_iou / paper_k6_floor。
noex 三阶段 none；self 三阶段 self；nogain gain1；mx_lv gain2；
noapp=mxc_lv；其余 gain 默认2。每组完成训练后对 best_model.pth 进行
val infer_mx --save-arms，再用官方评估实现 score --author：
Pd 按合并后的连通组件计，Fa 与 IoU 按像素计。
完整参数见 runtime/val_queue/recipe.json、planned_commands/*.json。

只用物理 GPU 0/1/2；无计算进程且 memory.used <500 MiB 才可启动。
运行中/初始化中的本队列容器也会占用该 GPU 的调度槽。
monitor 采用 --pid host 让 nvidia-smi 看得到宿主计算 PID；
NVIDIA_DRIVER_CAPABILITIES=utility 且 Python -S，不导入 torch、不计算模型。
monitor 使用 Docker socket 创建授权的 worker；复用 M0 镜像与只读数据挂载。
worker 通过 PYTHONPATH=/experiment/runtime 加载既有 torch 兼容补丁。
不终止其他任务，不调整 batch/epochs/配方，不自动重试失败 worker。

状态/来源：
  runtime/val_queue/state.json                 原子状态，支持重启后接续
  runtime/val_queue/events.jsonl、scheduler.log 调度事件与错误
  runtime/val_queue/scheduler.command.json     后台 monitor 的启动命令
  runtime/val_queue/commands/<run>.json        实际 GPU 分配后的 worker 命令
  runtime/val_queue/source_manifest.json       monitor 与 main 源文件来源
  runtime/val_queue/provenance/<run>.json      每组启动时源码 SHA 核验
  runs/<run>/{m0c,m1}.json                    M0/M1 决策、SHA、用户取消门槛
  logs/{train,infer,score}.<run>.log            各 worker 持久日志
主 run 已有日志沿用 logs/train.mta_val_s46.log、logs/eval.mta_val_s46.log。

运行观察：
  docker logs --tail 30 dp-mta-v8-val-scheduler-20261126
  cat runtime/val_queue/state.json
  cat results/comparison/val.txt

调度器用持久 Docker 容器（restart=unless-stopped）并加文件锁。
任务失败后 state 保留 failed/error，不重复创建；需先人工确认原因。
主 run 失败会暂停新的 control 启动，已经运行的 controls 不会被停止。
所有任务终止后 monitor 保持闲置监控，既不重复训练也不启动 official。

最终表 results/comparison/val.csv 与 val.txt 只公开已经训练32轮、完成最佳权重
推论/计分、且全体 val clip 权重 SHA/缓存指纹一致的直接 IoU/Pd/Fa@0.5 + AUC。
选权重的窗口 IoU 独立保存在 selection_IoU 列，不冒充最终融合后的检测 IoU。
pending/error 项留空。val 同时用于选择权重，这仍是验证集比较。

验证证据：
  runtime/val_scheduler_validation.json  13项假 Docker/GPU 队列验证
  runtime/val_summary_validation.json    纯文件指标/权重/缓存验证
  runtime/test_fixtures/                 合成 fixture，绝非实验数值
  runtime/val_queue/scheduler_smoke2.json 短监控容器实测，exit0，torch未导入
smoke 只查询 Docker 状态、GPU显存与计算PID；未启动调度循环或训练。
