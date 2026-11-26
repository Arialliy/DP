RGB-T：DeepPro-LV、DeepPro-LVH、MRTP、LVH-dec训练

目录：/home/md0/ly/DP/reproduction/rgbt_research_models_20261126
数据：/home/md0/ly/DP/datasets/RGB-T（只读挂载）
代码快照：/home/md0/ly/DP/deeppro-mx-complete 5 的 code/、configs/
source_manifest.json记录源文件哈希。模型/训练代码没有改动；独立configs/splits.json
仅新增official_train、official_test两个别名，分别等于数据目录中的train.txt/test.txt。
原目录及其他正在执行的验证任务保持独立。
本机torch2.2.0a缺少convert_conv3d_weight_memory_format工具；torch_compat.py提供
官方语义的纯Python兼容实现，train_entry.py安装后调用未修改的code/train.py。
它只调整Conv3d权重的内存布局，不改变权重数值、网络结构或损失；CPU等价性检查
及真实GPU更新检查记录在独立预检文件中。

共同配置
  官方45训练序列、19,135帧；15测试序列、5,976帧。
  本包默认train只有32序列，本次明确传--split official_train以与正在训练的Plus用相同数据。
  epochs=32；seed=46；batch=4；sample_rate=.05；40帧；128×128裁剪。
  每轮947个样本，drop_last后236次更新，总共7,552次更新/模型。
  Adam lr=.001、wd=.0001；每10轮lr乘.7；FP32张量，允许镜像默认TF32；channels_last_3d。
  仅用红外01通道；RGB-T均值85.0799、标准差47.4845。
  标签paper_k6_floor，对应已有Gaussian掩码；不是作者公开的原始标签。
  每个模型从随机初始化开始，不加载DeepPro-Plus或旧模型权重。

模型配置
  DeepPro-LV: lv_full；level_focal=0；persist=hard；hard_neg=0。
  DeepPro-LVH: lvh_full；level_focal=1；persist=ramp；hard_neg=.15；motion_gate=0。
  MRTP: mr_lv；level_focal=1；persist=hard；rates=1,2,4；rate_probs=.5,.25,.25；coords=auto。
  LVH-dec: lvh_dec；level_focal=1；persist=ramp；hard_neg=0；motion_gate=1；arrival_target=history。
  aux=.5、main_focal=0；LVH历史窗口hist_len=80、hist_step=4、hist_mode=both。
  LVH通过专用数据集自动配准，不向训练命令传--coords warp。

数据准备
  预先构建全部60序列uint8缓存，避免多模型首次同时生成缓存造成临时文件竞争。
  每个缓存验证尺寸/类型及首、中、末帧与原图完全一致，并记录SHA256。
  配准矩阵单独复制自deeppro-mx-complete/results/profile/homographies，60序列均经核验。
  原配准估计函数和图像/框几何处理与v5一致；矩阵不受旧轨迹编号冲突影响。
  不复制旧targets.csv。四个模型不需要MX的速度表、exemplar或M0流程。
  难负样本由本次快照中原hotspots.py为45训练序列生成，供LVH使用。
  preprocessing_provenance.json记录数据、缓存和配准矩阵来源与哈希。

GPU安排
  GPU 2：DeepPro-LV -> MRTP。
  GPU 0：DeepPro-LVH -> LVH-dec。
  GPU 1：保留给已经运行的DeepPro-Plus，不被此调度器使用。
  每个GPU同时只运行本批次的一个模型；等待中的模型会在前一个容器退出后启动。
  启动前需通过所有4个模型的真实batch4、40帧、128裁剪的一次参数更新检查。
  smoke输出位于runs/preflight_<arch>/，与正式32轮训练完全独立。
  后台容器和调度器不依赖当前终端持续打开。

检查进度
  cat /home/md0/ly/DP/reproduction/rgbt_research_models_20261126/REPORT.txt
  cat /home/md0/ly/DP/reproduction/rgbt_research_models_20261126/STATUS.json
  docker logs --tail 20 dp-rgbt-lv-full-20261126
  docker logs --tail 20 dp-rgbt-lvh-full-20261126
  docker logs --tail 20 dp-rgbt-mr-lv-20261126
  docker logs --tail 20 dp-rgbt-lvh-dec-20261126

结果
  runs/deeppro_lv_rgbt_s46/
  runs/deeppro_lvh_rgbt_s46/
  runs/mrtp_rgbt_s46/
  runs/lvh_dec_rgbt_s46/
  每个目录含config.json、train_log.txt、last.pth、epoch_8/16/24.pth、final.pth。
  job.py在训练结束后验证32轮/7,552更新、loss有限、最终权重严格加载及参数有限，
  然后才将jobs/<arch>.json状态标为completed；失败则记录traceback。
  STATUS.json/REPORT.txt每20秒更新；任务退出后logs/<arch>_console.log归档控制台。

协议边界
  本次用户请求为训练4个模型；没有展开3种子×5折或额外消融，也没有自动进行全测试评估。
  这些模型按原研究训练器使用最后一轮final.pth；正在运行的官方Plus按测试集IoU选best。
  数据/轮数相同不等于checkpoint选择协议完全相同，后续比较必须标明这一差异。
  若后续评估：LV/LVH/dec用code/infer.py，MRTP用code/infer_mr.py
  --rates 1,2,4 --coords auto --fusion max,cal。不要用infer.py替代MRTP推理。

调度器入口
  python3 /home/md0/ly/DP/reproduction/rgbt_research_models_20261126/scheduler.py
  文件锁防止重复调度；已有容器会被接管监控，已有checkpoint不会被静默覆盖或续训。
  当前批次已经启动调度器，正常情况下无需再次运行。
