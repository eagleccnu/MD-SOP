# MD-SOP v2.2：冻结版发布准备

本包对应实验`v2_2_20261107`，2026-11-08整理。六个模型、数据划分、特征规范和训练协议已冻结。本次只准备文件，没有创建远端GitHub仓库、提交代码或上传。

MD-SOP直接使用ESM-C 600M的1152维表示与HMM-v1-legacy30的30维表示，预测逐残基序参数S²。与原始SOPPCL不同，本模型没有VAE/对比学习环节；模型为128维投影＋两层BiLSTM＋w1或w15回归头。

v2.2只替换MD-iRED标签：先平均每个模拟重复中全部完整、互不重叠的5τc窗口S²，再对R1/R2/R3三个均值等权平均。不只取第一窗口、不拼接轨迹、不按所有窗口数重新加权、不用full100ns回退；S-OPPE和实验标签不变。

## 六个模型及使用入口

- `A_MD_w1`、`A_MD_w15`：MD标签训练。
- `B_SOPPE_w1`、`B_SOPPE_w15`：S-OPPE标签训练。
- `C_MIX_w1`、`C_MIX_w15`：MD与S-OPPE混合，各来源loss权重0.5。

不能把本轮实验RMSE最低的A_MD_w1当成事后选出的唯一正式模型；保留全部六组及负结果。

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
python code/predict_features.py --model A_MD_w1 \
  --features examples/synthetic_features.npy --device cpu \
  --output outputs/synthetic_A_MD_w1.tsv
```

示例特征为固定seed生成的随机数，只验证软件接口，不是蛋白质的真实ESM-C/HMM表示，也不是预测效果演示。真实预测可用`--esmc <L×1152.npy> --hmm <L×30.npy>`替代`--features`，并用`--fasta`检查长度。仅长度匹配并不能证明残基映射正确。

本包提供的是**已制备特征→预测**入口，不自动从FASTA提取全部特征。需要按[特征规范](docs/FEATURES.md)另备ESM-C和HMM。输出是完整序列的1-based位置及未经裁剪/平滑的S²预测。

## 冻结与复现边界

`checkpoints/`内六份推理权重来自原best，逐张量相同；不含优化器、调度器或随机状态，不用于继续训练。文件SHA与原best不同是正常的，模型参数SHA必须相同。

`frozen_training/`及`configs/`保存原科学脚本/配置，未将路径硬改成新机器，也没有绕过哈希验收。原训练入口依赖原完整绑定树，并非任意目录下立即可运行的训练工具。完整重训需要另备数据、建新运行标识并重新验收，不能覆盖本轮冻结结果。

原始MD轨迹、标签数组、ESM-C缓存、HH-suite数据库、第三方预训练模型、旧版本权重、last/优化器checkpoint、服务器日志和私人审计不放入待上传仓库。提供的数据集ID/划分及摘要不是完整训练数据发布，参见[数据可用性](docs/DATA_AVAILABILITY.md)。

## 结果与限制

六模型×MD/S/实验三个主端点共18项评分，所有训练曲线和MD的2×2目标对照均保留，见[结果表](results/RESULTS.md)。MD-only的MD与实验RMSE点估计较好，S-only的S域误差较低；混合训练及w15不一致占优。

仅一个训练seed，实验集10蛋白/8同源组，并且是历史复用基准。标签变化本身会改变评分，不能用“旧模型/旧标签”到“新模型/新标签”的对角线差宣称训练提升。实验端24项Holm检验缺少0.05水平拒绝分辨率，不显著不代表等效。历史SOPPCL只作背景比较，不能忽略开发池/同源重叠。

准备工作已完成不等于已发布。上传前仍需确认仓库名称/可见性、软件与权重许可、数据披露范围和稿件引用信息；不需要提供GitHub登录密码。
