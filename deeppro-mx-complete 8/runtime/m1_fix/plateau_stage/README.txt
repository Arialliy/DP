M1 plateau 候选最小修复 stage（主代码未修改）

可审阅补丁：plateau_candidates.patch
对应文件：fixed/code/mx_data.py、fixed/code/test_mx_data.py
原始副本：original/code/；共58个源文件快照，fixed中的其余56个原样保留，diff仅上述2个文件。
主目录 code/ 在stage结束时全体SHA与 original_source_manifest.json 一致。
root 在旧源量化结束后决定应用补丁与真实M1重跑；本stage未应用补丁。

规则：
- 保留原local_contrast、绝对值、严格floor>3、9×9 maximum_filter、M1每帧top100及其他全部参数。
- peak掩膜先8连通平台分组；各平台在真实最大值像素中选择距离该平台质心最近的点。
- 距离同分按(y,x)稳定选点；不同代表得分相同时按代表(y,x)行列序稳定排序，然后截取topk。
- 无峰或k<=0返回float64形状(0,4)。不使用GT，不新增可调参数，不改teacher。
- find_objects局部切片避免逐平台反复扫描整帧。

公共范围：frame_peaks还供SelfPeaks/self_bank及*_self对照使用。因此未来self候选也会采用
每平台一点的语义；这不是仅改一个M1结果文件，也不授权恢复任何模型训练。
旧源量化若用“平台首个row-major成员”做代表，其几何覆盖数仅是那个规则的假设，
不能当作本修复“质心最近实际成员”的最终恢复结果。

测试：
固定镜像 sha256:ea2c24a3a72cad22047223725869da1044de36feb1c5d0a19f23544a9cc3e594，CPU，无GPU，torch未导入。
test_mx_data：原10项+新增8项=18项通过。
新增覆盖实际单亮点9平台像素→1、12强点+1弱点预算、非凸实际代表、同分离散平台、
8连通对角、孤立峰排序、代表行列排序与严格floor/空结果/k0。
未改动的test_miss_recovery：11项全部通过，明确绑定stage的mx_data；其中CLI仅运行合成小数据。
总计29项，测试用时约11.3秒；没有真实M1、模型forward或训练。

首次测试因stage未带configs/splits.json而在M1合成CLI测试中断；候选测试均已通过。
复制未改动的配置后完整重跑29项通过。首次日志和返回码保留，未隐藏该次中断。

证据：tests.log、tests.attempt1.log、test.command.json、test_work/results.json、validation.json、
original_source_manifest.json、patch_dry_run.log。patch --dry-run -p1 已通过；没有真实应用。
