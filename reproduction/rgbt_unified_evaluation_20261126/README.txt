RGB-T 统一 ROC 评测汇总与真实数据绘图

统一评测入口
scheduler.py 在宿主机运行，使用 config.json 中固定的 Docker 镜像；只向物理 GPU 2
提交一个评测容器。容器内 cuda:0 是映射后的 GPU 2，不是宿主机 GPU 0。
STATUS.json、jobs/*.status.json 和 logs/scheduler.log 保存队列与逐视频进度。
scheduler.pid 和 scheduler.lock 用于定位调度器、防止重复启动。
评测容器退出即释放 GPU；调度器等待训练完成时不占 GPU。

需要恢复调度器时，在本目录执行：
  nohup python -u scheduler.py >> logs/scheduler.log 2>&1 &
已有运行中评测容器会被识别，完整结果会被复用，不会再次提交同一作业。
失败任务不会无限重试；先检查 jobs/<job_id>.status.json 与 logs/<job_id>.log。

权重与协议
DeepPro-Plus、DeepPro-LV、DeepPro-LVH、MRTP、LVH-dec 均读取各自原训练任务选出的
best_model.pth。权重按测试集 IoU 的原版规则选择；评测不更改训练或重新选权重。
首次先冻结当前 best；32 个 epoch 完成后自动接入最终 best。相同 SHA256 复用
已完成评测并改为 final，不同 SHA256 则新建评测作业，完整结束后替换报告中的中期值。
checkpoints/ 内是独立权重快照，jobs/ 记录来源、SHA256 和快照时的训练轮数。
source_manifest.json 固定实际推理和指标代码；每次评测均核验哈希。

统一使用原 15 个测试视频、5,976 帧、现有 PNG 标签（11,300 个连通目标），
FP32 推理，40 帧窗口，重叠位置取最大值后每个源帧只统计一次。
Pd、Fa 使用原 ShootingRules 的 27 个阈值、8 连通目标、1 像素命中邻域、
4 像素虚警排除邻域，并保留原边界切片行为。AUC 为这些实测点的梯形积分。
Pd 是命中的连通目标比例；Fa 是排除目标邻域后的虚警像素数 / 全部像素数。
CSV 同时保留 0.5、0.99 的 Pd/Fa，roc_points.csv 保留全部 27 个操作点及整数计数。
这里的融合后 IoU 与每个 epoch 按窗口累计的选权重 IoU 不完全相同。

模型推理沿用各自已声明的设置：LVH/dec 保留历史和稳定化，MRTP 保留 rates=1,2,4
与 auto 坐标。MRTP 的 rate1_auto、multirate_max、multirate_cfar_score 分别报告，
事先指定 multirate_max 为主结果。CFAR 项使用固定 sigmoid 转换后的排序分数，
不是校准概率；有限的 27 点 AUC 不能视为原生 CFAR 阈值积分。
作者发布权重一行来自此前已完成的原版 test.py 计数，作为独立参考，不冒充本次重训。

验证记录
preflight.json 和 logs/unit_tests.log 记录 16 项 CPU 验证、5 种模型的真实尺寸
GPU 前向验证，以及一个真实视频的完整指标流程验证。
原指标与优化指标逐项计数核对；完整评测额外核对 15 clips / 5,976 frames /
1,958,215,680 pixels / 11,300 targets。部分视频的预检结果不会进入正式汇总。
figure_qa.json 只适用于其中记录哈希的图像；后续自动更新图像需要重新视觉检查。

职责
report.py 仅读取已完成的评测结果，生成汇总表、状态报告和 Python/matplotlib 图。
不运行推理、不使用GPU、不下载内容、不生成模拟数据。
本阶段统一评测由root使用用户临时授权的GPU 2执行，评测结束后释放。
既有训练仍由GPU 0/1承担；report.py本身只需要CPU。

运行
  python report.py
默认读取脚本所在目录的 config.json 与 results/，并将导出写回该目录。
  python report.py --root /evaluation --output /evaluation
  python report.py --no-plot
--no-plot仅生成CSV与文字报告；无需导入绘图库。
完整绘图需要Python 3.10+、matplotlib，可复用image-id.txt指定的既有兼容环境。
若缺少Python绘图库，脚本明确报错，不切换绘图语言。

结果文件契约
每个job/variant写入独立 results/<job_id>[/<variant>]/metrics.json：
  model: 稳定模型标识，同模型不同variant使用同一标识以保持颜色一致
  label: 图例显示名称
  arch: 网络架构标识
  checkpoint_sha256: 实际评测权重的64位十六进制SHA256
  best_epoch: 最佳权重轮次；未知可为null
  training_epochs_completed: 已完成训练轮数；作者发布权重未知时可为null
  final_checkpoint: 完整计划训练结束后的选定权重为true，中期权重为false
  checkpoint_status: 建议明确填 released / final / interim
  variant: 推理或评测变体名称；默认可填default或native
  auc: 评测器计算的真实AUC，使用0..1尺度
  iou_at_0_5: 原始0.5阈值IoU；无法核实则null
  complete: 完整评测为true；false的结果不进入曲线或数值汇总
  source: 来源描述字符串或对象；作者权重可用 {"kind":"released"}
  protocol: 保留数据、标签、阈值、匹配规则、融合等评测说明
  thresholds: 操作点数组，每项为以下对象：
    threshold, pd, fa, true_targets, targets, false_pixels
  thresholds也兼容按上述顺序排列的六列数组。
Pd/Fa均为0..1比例，计数为非负整数；同一曲线的targets必须一致。
Pd与true_targets/targets需在1e-5内一致，允许原作者文字日志的小数舍入。

权重来源分类
优先采用显式checkpoint_status。缺省时，source.kind/type为released、
author_released、released_checkpoint或released=true视作released；否则由
final_checkpoint区分final/interim。不根据文件名推测作者发布权重。
这里final表示完整训练结束后的最佳权重，不要求best_epoch等于最后一轮。

结果筛选
config.json存在active_results时严格使用它；否则尝试manifest.json中的同名字段；
两者都未定义时递归读取results/**/metrics.json。
推荐active_results为列表，元素可为：
  相对metrics.json路径字符串；
  或 {"path":"results/<job>/metrics.json","checkpoint_sha256":"<已选权重哈希>"}。
也支持job_id（解析到results/<job_id>/metrics.json）或以job_id为键的对象。
显式空列表表示不纳入任何结果。选定文件缺失、未完整评测、哈希不符或数据非法时，
会列入REPORT.txt，不回退到旧权重结果。
pending_results支持字符串列表，或包含label/model/job_id和reason/status的对象列表；
报告逐项展示该列表。pending、pending_models是兼容别名。
config中的pending列表由root更新，脚本不会自行猜测缺少的训练结果。

导出
metrics.csv：每条完整有效曲线一行，含权重状态、hash、轮次、AUC、IoU及source/protocol。
roc_points.csv：所有原始操作点，保留输入顺序、计数、Fa=0与完整数值。
REPORT.txt：实际纳入结果、pending、被排除结果、协议差异和绘图状态。
ROC.png：600dpi预览。
ROC.pdf：TrueType可编辑文字。
ROC.svg：可编辑SVG文字。

图形约定
双面板：全Fa范围0..1；重点低Fa范围0..1e-3。两个面板均为线性轴，保留Fa=0。
仅连接现有阈值点，不插入操作点、不平滑、不拟合、不补端点，不重新归一化AUC。
模型颜色跨面板一致；final实线、interim虚线、released点线；状态与variant还通过
图例和标记区分。共享图例包含训练/权重状态、best_epoch、variant与真实AUC。
无结果时只显示等待提示，不创建任何假曲线。
不同结果的targets总数不一致时，REPORT明确提示其标注协议可能不同，不视作完全
相同ground truth下的直接比较。单次checkpoint曲线不提供虚构置信区间或显著性结论。

交付前检查
核验active_results已指向当前选定权重，pending列表与实际状态一致。
打开真实输出的PNG/PDF/SVG检查图例与标签、不重叠/裁切、文字可编辑以及最终尺寸可读。
figure_contract.json记录图形论点、数据证据链、输出与QA约束；当前脚本不声称已完成
尚未生成的图像的人工视觉审核。
