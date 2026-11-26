DeepPro-Plus / RGB-T 本地复现

目录：/home/md0/ly/DP/reproduction/rgbt_deepproplus_20261126
数据：/home/md0/ly/DP/datasets/RGB-T（只读挂载）
来源：https://github.com/TinaLRJ/DeepPro
论文：https://arxiv.org/html/2506.12766v5

配置
  原 README 未显式设置 epoch；原 train.py 默认 32。本次采用 32。
  DeepPro-Plus；仅红外01；40帧；128×128裁剪；batch4；FP32。
  Adam lr=.001, weight_decay=.0001；每10轮lr乘.7；SoftIoU loss。
  sample_rate=.05：采用 train.py 对 RGB-T 的参数注释建议。
  此项不同于不带该参数运行 README 命令时的默认 .1。
  每轮947个样本，drop_last=True 后236次更新；共32轮。
  原始训练未调用seed_everything，worker种子依赖时间，本次保留这一行为。
  原程序每轮用测试集IoU选择best_model；这是原实现协议，不是独立验证集协议。

改动范围
  原仓库源码与数据未改动；source/ 是独立运行副本。
  唯一算法入口改动：train.py --sample_rate 参数类型 int 改为 float。
  结构、损失、采样、训练循环和评估计算均为原实现。
  source_manifest.json 记录原文件和运行文件SHA256；source_changes.diff列出改动。
  pipeline.py 调用 train.main 后显式调用并检查 test.py，修复原自动测试
  忽略自定义输出路径和子进程失败的问题；不改动训练/测试计算。
  runtime/ 是本机Docker兼容环境，保留基础镜像torch，固定NumPy1.23.5。
  它不是作者未公开的原始运行环境。

数据限制与参照值
  45训练序列/19,135帧；15测试序列/5,976帧；全部灰度640×512。
  mask是已有的k=6/floor Gaussian重建标签，不能视为作者原始mask。
  当前测试目标11,300，作者日志11,303；详见data_audit.json。
  训练与测试clip互斥，但4个源视频前缀跨组；保留提供的划分。
  test.py对每个序列40帧滑窗、重叠4帧、重叠处取最大概率，共172窗口。
  保留27阈值ROC和原ShootingRules，IoU阈值=.5。
  论文表格：Pd=77.83%, Fa=4.47e-5, AUC=.9730。
  作者存档在阈值.99时Pd/Fa与论文四舍五入值一致，据此推断表格工作点为.99。
  作者存档曲线按原始计数积分AUC=.96912177，与论文.9730本身已有差异。
  两个阈值.5/.99都应报告；不得以测试集调参后替代原表格工作点。

运行（从任意目录执行；镜像ID固定在runtime/image-id.txt）
  python3 /home/md0/ly/DP/reproduction/rgbt_deepproplus_20261126/reproduce.py preflight --gpu 1
  python3 /home/md0/ly/DP/reproduction/rgbt_deepproplus_20261126/reproduce.py released --gpu 2
  python3 /home/md0/ly/DP/reproduction/rgbt_deepproplus_20261126/reproduce.py train --gpu 1

  released 和 train 使用独立的后台容器，关闭终端后继续执行。
  命名容器与已有训练checkpoint会阻止误启动重复训练，不会静默覆盖已有实验。
  本次训练从随机初始化开始，不加载released权重。

查看进度
  python3 /home/md0/ly/DP/reproduction/rgbt_deepproplus_20261126/reproduce.py status
  docker logs --tail 30 deeppro-rgbt-train-20261126
  docker logs --tail 30 deeppro-rgbt-released-20261126
  python3 /home/md0/ly/DP/reproduction/rgbt_deepproplus_20261126/report.py

结果
  preflight.json：真实训练batch前向/反向/Adam与全分辨率权重推理预检。
  released_status.json / train_status.json：starting/training/evaluating/completed/failed。
  released_metrics.json / train_metrics.json：成功完成后写入的27阈值指标。
  author_metrics.json：作者已有日志独立提取的参照值。
  REPORT.txt：状态和指标摘要，阶段完成后自动刷新，也可手动运行report.py刷新。
  STATUS.json：后台monitor.py每30秒记录真实容器状态；任务退出后归档console日志。
  log/sem_seg/released_rgbt_deepproplus/eval.txt：发布权重重新评估日志。
  log/sem_seg/retrained_rgbt_deepproplus/logs/DeepPro-Plus.txt：每轮训练日志。
  log/sem_seg/retrained_rgbt_deepproplus/checkpoints/：best及每5轮/末轮权重。
  log/sem_seg/retrained_rgbt_deepproplus/eval.txt：训练完成后的完整测试日志。

状态文件只表示程序阶段；实际进程状态以reproduce.py status的container为准。
存在checkpoint不代表32轮训练或最终测试已完成，应同时检查completed状态与metrics文件。
