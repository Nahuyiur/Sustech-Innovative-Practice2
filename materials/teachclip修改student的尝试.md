# Openclip

使用卷积模型（Convnextv）替换CLIP中的视觉encoder（ViT）来完成clip任务

## Part A：ConvNeXtV2-Tiny

这三组的共同点（便于看“策略差异”）：
- 任务/数据一致：COCO train2017/val2017。
- 大多数超参一致：`epochs=26`、`lr=7e-4`、`precision=amp`、`lr_scheduler=cosine`。
- 有效 global batch 基本一致：`batch_size=192`、`world_size=2`、`accum_freq=1`，即 `192×2×1=384`。

### A1) Tiny 三组总览（最终效果）

| 实验目录 | 模型 | 文本塔 | 冻结/解冻核心策略 | warmup | I2T R@1/5/10 | T2I R@1/5/10 |
| --- | --- | --- | --- | --- | --- | --- |
| coco2017_convnextv2_tiny_gpt2_amp_cosine_bs192x2_nofreeze_lr7.0e-4_wu2.2k | convnextv2_tiny_gpt2-512 | GPT2（pretrained_text_path） | 不冻结（`lock_image=False, lock_text=False`） | 2200 | 8.12%/28.27%/40.50% | 5.93%/29.39%/41.57% |
| coco2017_convnextv2_tiny_gpt2_amp_cosine_bs192x2_unf35_lr7e-4_wu2.4k | convnextv2_tiny_gpt2-512 | GPT2（pretrained_text_path） | 先冻结，后解冻（text epoch=3 解冻 6 层；image epoch=5 解冻 4 组） | 2400 | 9.55%/34.26%/47.52% | 6.90%/34.52%/47.62% |
| coco2017_convnextv2_tiny_cliptext_amp_cosine_bs192x2_unfimg5_lr7e-4_wu2.4k | convnextv2_tiny_cliptext-512 | CLIP text（无 text safetensors） | image 先冻结，epoch=5 解冻 4 组；text 在 params 中 `lock_text=False` | 2400 | 9.07%/32.49%/45.62% | 6.75%/32.97%/45.96% |

### A2) Tiny 的“策略变化”怎么理解（不是简单列参数）

这里最有信息量的对比是：**同样是 GPT2 文本塔时**，“全量训练（nofreeze）” vs “冻结后逐步解冻（unf35）”。

1) `...gpt2...nofreeze...`（全量训练）
- 改动本质：把 image tower + GPT2 text tower 全部当成可训练参数一起端到端 finetune（`lock_image=False, lock_text=False`）。
- 代价：可训练参数非常大（`out.log` 显示 trainable total ≈ 155M），在 COCO 这种相对小数据上更容易出现“训练不稳定/难以泛化”的情况。
- 结果：三组里最终 R@ 反而最低（I2T R@1=8.12%，T2I R@1=5.93%）。

2) `...gpt2...unf35...`（冻结→按计划解冻）
- 改动本质：先把两塔大部分参数冻结，只训练较小子集；再按 epoch 解冻部分层/组，让模型逐步适配数据。
  - `unfreeze_text_at_epoch=3` 且 `unlocked_text_layers=6`：先放开文本塔一小部分。
  - `unfreeze_image_at_epoch=5` 且 `unlocked_image_groups=4`：再放开更多视觉侧。
- 直观效果：可训练参数从“全量训练的 1e8 级”压缩到“百万级”（`out.log` 显示 total ≈ 3.0M），训练更像“有约束的微调”。
- 结果：相对 nofreeze 明显提升（I2T R@1 8.12%→9.55%，T2I R@1 5.93%→6.90%）。这说明在该设置下，“控制可训练子集 + 逐步解冻”比“一上来全量 finetune”更有效。

3) `...cliptext...unfimg5...`（换文本塔实现 + 视觉侧解冻）
- 改动本质：文本侧从 GPT2（外部 `model.safetensors`）换成 `cliptext` 版本；视觉侧仍是“先锁再解冻”（`unfreeze_image_at_epoch=5`、`unlocked_image_groups=4`）。
- 与 `unf35` 的关键不同：这组没有 `pretrained_text_path`，因此“文本塔是否强预训练/是否参与训练”的实际情况更依赖实现细节；同时该组 `out.log` 的 trainable 统计口径与另外两组不一致（见本文开头说明）。
- 结果：指标介于 nofreeze 与 unf35 之间（I2T R@1=9.07%，T2I R@1=6.75%），整体略低于“GPT2 + 逐步解冻”的 `unf35`。

---

## Part B：ConvNeXt-Base（将学生模型替换为ConvNeXt-Base，原始模型投影头为线性的，不是注意力聚合的absattn，我训的是换成注意力聚合的absattn）

这四组围绕 `convnext_base_w_absattn` 做迭代，主线是：
1) 先做分阶段解冻的预训练/微调（3stage）。
2) 引入 distillation（teacher 也是 `convnext_base_w` 权重）。
3) 在 distill 下尝试“是否/何时解冻”。
4) 去掉 distill，加入 headalign并调整解冻节奏。

共同点（便于聚焦改动）：
- 数据/评估一致（COCO）。
- global batch 也基本固定在 384（`192×2` 或 `96×4`）。
- 视觉/文本大多初始锁定（`lock_image=True, lock_text=True`），训练更偏向“只训少量参数/逐步放开”。

### B1) CNBW 四组总览（最终效果）

| 实验目录 | 主要策略变化（相对上一阶段） | lr / epochs | 解冻策略摘要 | I2T R@1/5/10 | T2I R@1/5/10 |
| --- | --- | --- | --- | --- | --- |
| coco2017_cnbw_absattn_3stage_preval_lr2e-5_bs192x2_10e | baseline：不蒸馏 + 三阶段逐步解冻 | 2e-5 / 10 | image epoch=3 解冻 2 组；text epoch=8 解冻 | 12.34%/44.34%/58.47% | 8.85%/45.43%/58.35% |
| coco2017_cnbw_absattn_distill_lr5e-4_frozen_bs96x4_20e | 加 distill + 全程冻结（不解冻） | 5e-4 / 20 | `unfreeze_* = None` | 13.10%/46.81%/60.30% | 9.74%/48.61%/60.84% |
| coco2017_cnbw_absattn_distill_lr5e-4_imgunf5g2_text10_bs96x4_20e | distill 仍开 + 改为后期解冻 | 5e-4 / 20 | image epoch=8 解冻 2 组；text epoch=18 解冻 | 12.98%/46.12%/59.37% | 9.08%/47.44%/59.81% |
| coco2017_cnbw_absattn_headalign_lr5e-4_imgunf6g1_text10_bs96x4_acc1_10e | 关闭 distill + 调整为更早/更小幅解冻（并引入 headalign） | 5e-4 / 10 | image epoch=6 解冻 1 组；text epoch=10 解冻 | 13.11%/46.95%/60.96% | 9.86%/48.61%/61.30% |

### B2) CNBW 的“策略变化”怎么理解（不是简单列参数）

1) `3stage_preval`：逐步解冻适配（小 lr、短训练）
- 改动本质：把大模型当成“预训练权重 + 少量可训练参数”的起点，通过 `unfreeze_image_at_epoch=3`、`unfreeze_text_at_epoch=8` 逐步放开，让模型先学对齐头/投影，再让 backbone 适配。
- 结果：作为 baseline，达到 I2T R@1=12.34%。

2) `distill...frozen`：用 distillation 约束学习，但保持冻结（训练更久、lr 更大）
- 改动本质：引入 teacher（`distill=True`）提供额外监督，同时不做任何解冻（`unfreeze_* = None`），把学习限制在“少量可训练参数”上。
- 直观解释：distill 在这里更像“强正则/对齐约束”，避免在小数据上乱跑；冻结则进一步减少可训练自由度。
- 结果：相对 3stage baseline 有提升（I2T R@1=13.10%，T2I R@1=9.74%）。

3) `distill...imgunf...text...`：同样 distill，但尝试在后期放开更多参数
- 改动本质：保持 distill，增加后期解冻（image epoch=8、text epoch=18），并把 `unfreeze_new_lr_scale` 调小到 0.05。
- 可能原因：希望在 teacher 约束下，最后几轮能“微调 backbone/文本”拿到额外增益。
- 结果：最终指标没有超过“distill+全程冻结”（I2T R@1 12.98% < 13.10%）。这暗示在该训练时长/数据规模下，后期解冻带来的自由度可能大于收益。

4) `headalign...`：关闭 distill，改成更早更小幅度解冻（并加入 headalign）
- 改动本质：不再依赖 teacher，而是通过（从命名推断的）head-level 对齐损失/策略来替代 distill 的约束；同时解冻更早（image epoch=6、text epoch=10）但幅度更小（image 仅 1 组）。
- 结果：这组在 base 系列里指标最好（I2T R@1=13.11%，T2I R@1=9.86%，T2I R@10=61.30%）。

---

## 附：逐实验关键配置（便于追溯）

下面保留每个实验的关键字段，用于回查 `params.txt`。

### coco2017_convnextv2_tiny_gpt2_amp_cosine_bs192x2_nofreeze_lr7.0e-4_wu2.2k

- 训练策略（来自 params.txt）
  - model: convnextv2_tiny_gpt2-512
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0007 / cosine / 2200 / 0.2
  - epochs: 26
  - batch/world/accum/global: 192 / 2 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=False, lock_text=False; unfreeze_image_at_epoch=None (unlocked_image_groups=None); unfreeze_text_at_epoch=None (unlocked_text_layers=None); unfreeze_new_lr_scale=0.1
  - pretrained_image: True (/lab/haoq_lab/cse12412832/OpenClip/data/weights/convnextv2_tiny/model.safetensors)
  - pretrained_text_path: /lab/haoq_lab/cse12412832/OpenClip/data/weights/gpt2/model.safetensors
- out.log 可训练参数（INIT）：visual=30070400, text=125259008, total=155329409
- 最终效果（results.jsonl 最后一条）
  - epoch: 26
  - I2T R@1/5/10: 8.12% / 28.27% / 40.50%
  - T2I R@1/5/10: 5.93% / 29.39% / 41.57%

### coco2017_convnextv2_tiny_gpt2_amp_cosine_bs192x2_unf35_lr7e-4_wu2.4k

- 注意：`params.txt` 里的 name 与目录名不同：name=coco2017_convnextv2_tiny_gpt2_amp_cosine_bs192x2_unf35_lr7e-4_wu2.4k_debug
- 训练策略（来自 params.txt）
  - model: convnextv2_tiny_gpt2-512
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0007 / cosine / 2400 / 0.2
  - epochs: 26
  - batch/world/accum/global: 192 / 2 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=True, lock_text=True; unfreeze_image_at_epoch=5 (unlocked_image_groups=4); unfreeze_text_at_epoch=3 (unlocked_text_layers=6); unfreeze_new_lr_scale=0.1
  - pretrained_image: True (/lab/haoq_lab/cse12412832/OpenClip/data/weights/convnextv2_tiny/model.safetensors)
  - pretrained_text_path: /lab/haoq_lab/cse12412832/OpenClip/data/weights/gpt2/model.safetensors
- out.log 可训练参数（INIT）：visual=2203904, text=819200, total=3023105
- 最终效果（results.jsonl 最后一条）
  - epoch: 26
  - I2T R@1/5/10: 9.55% / 34.26% / 47.52%
  - T2I R@1/5/10: 6.90% / 34.52% / 47.62%

### coco2017_convnextv2_tiny_cliptext_amp_cosine_bs192x2_unfimg5_lr7e-4_wu2.4k

- 训练策略（来自 params.txt）
  - model: convnextv2_tiny_cliptext-512
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0007 / cosine / 2400 / 0.2
  - epochs: 26
  - batch/world/accum/global: 192 / 2 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=True, lock_text=False; unfreeze_image_at_epoch=5 (unlocked_image_groups=4); unfreeze_text_at_epoch=None (unlocked_text_layers=None); unfreeze_new_lr_scale=0.1
  - pretrained_image: True (/lab/haoq_lab/cse12412832/OpenClip/data/weights/convnextv2_tiny/model.safetensors)
- out.log 可训练参数（INIT）：visual=2203904, text=0, total=65632001
- 最终效果（results.jsonl 最后一条）
  - epoch: 26
  - I2T R@1/5/10: 9.07% / 32.49% / 45.62%
  - T2I R@1/5/10: 6.75% / 32.97% / 45.96%

### coco2017_cnbw_absattn_3stage_preval_lr2e-5_bs192x2_10e

- 训练策略（来自 params.txt）
  - model: convnext_base_w_absattn
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 2e-05 / cosine / 1000 / 0.2
  - epochs: 10
  - batch/world/accum/global: 192 / 2 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=True, lock_text=True; unfreeze_image_at_epoch=3 (unlocked_image_groups=2); unfreeze_text_at_epoch=8 (unlocked_text_layers=None); unfreeze_new_lr_scale=0.2
  - pretrained: /lab/haoq_lab/cse12412832/TeachClip/data/models/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin
- out.log 可训练参数（INIT）：visual=4265600, text=0, total=4265601
- 最终效果（results.jsonl 最后一条）
  - epoch: 10
  - I2T R@1/5/10: 12.34% / 44.34% / 58.47%
  - T2I R@1/5/10: 8.85% / 45.43% / 58.35%

### coco2017_cnbw_absattn_distill_lr5e-4_frozen_bs96x4_20e

- 训练策略（来自 params.txt）
  - model: convnext_base_w_absattn
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0005 / cosine / 1200 / 0.2
  - epochs: 20
  - batch/world/accum/global: 96 / 4 / 1 / 384
  - distill: True (teacher_model=convnext_base_w)
    - teacher_pretrained: /lab/haoq_lab/cse12412832/TeachClip/data/models/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin
  - 冻结/解冻: lock_image=True, lock_text=True; unfreeze_image_at_epoch=None (unlocked_image_groups=None); unfreeze_text_at_epoch=None (unlocked_text_layers=None); unfreeze_new_lr_scale=0.1
  - pretrained: /lab/haoq_lab/cse12412832/TeachClip/data/models/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin
- out.log 可训练参数（INIT）：visual=3856000, text=0, total=3856001
- 最终效果（results.jsonl 最后一条）
  - epoch: 20
  - I2T R@1/5/10: 13.10% / 46.81% / 60.30%
  - T2I R@1/5/10: 9.74% / 48.61% / 60.84%

### coco2017_cnbw_absattn_distill_lr5e-4_imgunf5g2_text10_bs96x4_20e

- 训练策略（来自 params.txt）
  - model: convnext_base_w_absattn
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0005 / cosine / 1200 / 0.2
  - epochs: 20
  - batch/world/accum/global: 96 / 4 / 1 / 384
  - distill: True (teacher_model=convnext_base_w)
    - teacher_pretrained: /lab/haoq_lab/cse12412832/TeachClip/data/models/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin
  - 冻结/解冻: lock_image=True, lock_text=True; unfreeze_image_at_epoch=8 (unlocked_image_groups=2); unfreeze_text_at_epoch=18 (unlocked_text_layers=None); unfreeze_new_lr_scale=0.05
  - pretrained: /lab/haoq_lab/cse12412832/TeachClip/data/models/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin
- out.log 可训练参数（INIT）：visual=3856000, text=0, total=3856001
- 最终效果（results.jsonl 最后一条）
  - epoch: 20
  - I2T R@1/5/10: 12.98% / 46.12% / 59.37%
  - T2I R@1/5/10: 9.08% / 47.44% / 59.81%

### coco2017_cnbw_absattn_headalign_lr5e-4_imgunf6g1_text10_bs96x4_acc1_10e

- 训练策略（来自 params.txt）
  - model: convnext_base_w_absattn
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0005 / cosine / 1000 / 0.2
  - epochs: 10
  - batch/world/accum/global: 96 / 4 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=True, lock_text=True; unfreeze_image_at_epoch=6 (unlocked_image_groups=1); unfreeze_text_at_epoch=10 (unlocked_text_layers=None); unfreeze_new_lr_scale=0.05
  - pretrained: /lab/haoq_lab/cse12412832/TeachClip/data/models/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin
- out.log 可训练参数（INIT）：visual=3856000, text=0, total=3856001
- 最终效果（results.jsonl 最后一条）
  - epoch: 10
  - I2T R@1/5/10: 13.11% / 46.95% / 60.96%
  - T2I R@1/5/10: 9.86% / 48.61% / 61.30%
  - clip_val_loss: 2.1771786212921143

### coco2017_cnbw_absattn_headalign_lr5e-4_imgunf6g1_text10_bs96x4_acc1_10e

- 训练策略（来自 params.txt）
  - model: convnext_base_w_absattn
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0005 / cosine / 1000 / 0.2
  - epochs: 10
  - batch/world/accum/global: 96 / 4 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=True, lock_text=True; unfreeze_image_at_epoch=6 (unlocked_image_groups=1); unfreeze_text_at_epoch=10 (unlocked_text_layers=None); unfreeze_new_lr_scale=0.05
  - pretrained: /lab/haoq_lab/cse12412832/TeachClip/data/models/CLIP-convnext_base_w-laion2B-s13B-b82K/open_clip_pytorch_model.bin
  - pretrained_image: False (None)
  - pretrained_text_path: None
- out.log 可训练参数（INIT）：visual=3856000, text=0, total=3856001
- 最终效果（results.jsonl 最后一条）
  - epoch: 10
  - I2T R@1/5/10: 13.11% / 46.95% / 60.96%
  - T2I R@1/5/10: 9.86% / 48.61% / 61.30%
  - clip_val_loss: 2.3481009006500244

### coco2017_convnextv2_tiny_cliptext_amp_cosine_bs192x2_unfimg5_lr7e-4_wu2.4k

- 训练策略（来自 params.txt）
  - model: convnextv2_tiny_cliptext-512
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0007 / cosine / 2400 / 0.2
  - epochs: 26
  - batch/world/accum/global: 192 / 2 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=True, lock_text=False; unfreeze_image_at_epoch=5 (unlocked_image_groups=4); unfreeze_text_at_epoch=None (unlocked_text_layers=None); unfreeze_new_lr_scale=0.1
  - pretrained_image: True (/lab/haoq_lab/cse12412832/OpenClip/data/weights/convnextv2_tiny/model.safetensors)
  - pretrained_text_path: None
- out.log 可训练参数（INIT）：visual=2203904, text=0, total=65632001
- 最终效果（results.jsonl 最后一条）
  - epoch: 26
  - I2T R@1/5/10: 9.07% / 32.49% / 45.62%
  - T2I R@1/5/10: 6.75% / 32.97% / 45.96%
  - clip_val_loss: 3.162996768951416

### coco2017_convnextv2_tiny_gpt2_amp_cosine_bs192x2_nofreeze_lr7.0e-4_wu2.2k

- 训练策略（来自 params.txt）
  - model: convnextv2_tiny_gpt2-512
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0007 / cosine / 2200 / 0.2
  - epochs: 26
  - batch/world/accum/global: 192 / 2 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=False, lock_text=False; unfreeze_image_at_epoch=None (unlocked_image_groups=None); unfreeze_text_at_epoch=None (unlocked_text_layers=None); unfreeze_new_lr_scale=0.1
  - pretrained_image: True (/lab/haoq_lab/cse12412832/OpenClip/data/weights/convnextv2_tiny/model.safetensors)
  - pretrained_text_path: /lab/haoq_lab/cse12412832/OpenClip/data/weights/gpt2/model.safetensors
- out.log 可训练参数（INIT）：visual=30070400, text=125259008, total=155329409
- 最终效果（results.jsonl 最后一条）
  - epoch: 26
  - I2T R@1/5/10: 8.12% / 28.27% / 40.50%
  - T2I R@1/5/10: 5.93% / 29.39% / 41.57%
  - clip_val_loss: 3.66436767578125

### coco2017_convnextv2_tiny_gpt2_amp_cosine_bs192x2_unf35_lr7e-4_wu2.4k

- 训练策略（来自 params.txt
  - model: convnextv2_tiny_gpt2-512
  - opt/precision: adamw / amp
  - lr/scheduler/warmup/wd: 0.0007 / cosine / 2400 / 0.2
  - epochs: 26
  - batch/world/accum/global: 192 / 2 / 1 / 384
  - distill: False
  - 冻结/解冻: lock_image=True, lock_text=True; unfreeze_image_at_epoch=5 (unlocked_image_groups=4); unfreeze_text_at_epoch=3 (unlocked_text_layers=6); unfreeze_new_lr_scale=0.1
  - pretrained_image: True (/lab/haoq_lab/cse12412832/OpenClip/data/weights/convnextv2_tiny/model.safetensors)
  - pretrained_text_path: /lab/haoq_lab/cse12412832/OpenClip/data/weights/gpt2/model.safetensors
- out.log 可训练参数（INIT）：visual=2203904, text=819200, total=3023105
- 最终效果（results.jsonl 最后一条）
  - epoch: 26
  - I2T R@1/5/10: 9.55% / 34.26% / 47.52%
  - T2I R@1/5/10: 6.90% / 34.52% / 47.62%
  - clip_val_loss: 2.912994623184204

# TeachClip

## 特征预提取

使用教师模型XCLIP对MSRVTT-7k/9k进行特征预提取，在蒸馏学生的时候可以使用预提取的特征减少计算量，速度加快约15%，并缓解OOM

## 使用Convnextv为视觉encoder的CLIP模型替换学生模型进行蒸馏

> 基于Openclip中预训练的convnext_base_w模型

### msrvtt9k（10epoch）

Text-to-Video:

R@1: 3.4 - R@5: 8.7 - R@10: 12.5 - Median R: 119.0 - Mean R: 145.5

Video-to-Text:

V2T$R@1: 3.4 - V2T$R@5: 9.6 - V2T$R@10: 13.7 - V2T$Median R: 118.0 - V2T$Mean R: 145.7

### msrvtt-7k（10epoch）

Text-to-Video:

R@1: 2.5 - R@5: 8.8 - R@10: 12.8 - Median R: 125.0 - Mean R: 164.0

Video-to-Text:

V2T$R@1: 2.5 - V2T$R@5: 13.6 - V2T$R@10: 19.9 - V2T$Median R: 71.0 - V2T$Mean R: 124.1

### msrvtt-7k（16epoch，锁定视觉和文本encoder，只训练中间对齐和帧间聚合部分）

Text-to-Video:

R@1: 2.6 - R@5: 8.4 - R@10: 12.6 - Median R: 127.0 - Mean R: 163.5

Video-to-Text:

V2T$R@1: 3.6 - V2T$R@5: 11.2 - V2T$R@10: 19.0 - V2T$Median R: 72.0 - V2T$Mean R: 126.6
