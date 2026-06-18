# AudioLDM2 潜在隐写错误分布分析与纠错码方案

## 1. 背景与目标

当前项目在 AudioLDM2 的 latent diffusion 过程中进行隐写。秘密信息不是直接嵌入 wav，而是先映射到最后阶段 DDIM 更新中的噪声变量 `z_s`，再由嵌入后的 `z_s` 构造 `stego_x1`，继续生成 `stego_x0`，最后通过 VAE decoder 和 vocoder 得到 stego wav。

接收端的目标流程为：

```text
stego wav
-> fbank / mel
-> VAE encoder
-> x0'
-> DDIM inversion 得到 x1'
-> 使用同步生成的 x2' 和 x1' 反推出 z_s'
-> z_s' 解码得到 bits'
```

在未做 repeat 的情况下，早期 VAE-only 提取准确率约为：

```text
bit accuracy ≈ 60.38%
BER ≈ 39.62%
```

随后通过 channel 和 time 筛选，只在更稳定的 latent 区域嵌入，当前无 repeat 的 bit accuracy 已提升到约：

```text
bit accuracy ≈ 91%
BER ≈ 9%
```

下一步目标是：在保持较高容量的前提下，通过纠错码将最终 payload 恢复准确率提升到 100%。

---

## 2. 现有错误分布分析结论

### 2.1 错误不是完全均匀随机分布

之前对 VAE-Channel Bit Extraction 的错误分布做过统计，整体 BER 约 0.396。错误在 latent 空间中并非完全均匀分散，而是存在明显的结构性差异：

```text
latent shape = [1, 8, 256, 16]
channel 维度 = 8
time 维度 = 256
freq 维度 = 16
```

主要观察包括：

1. 不同 channel 的错误率不同。
2. 不同 time 区域的错误率不同。
3. 某些 channel-time 区域明显更不稳定。
4. 错误并没有高度集中到极少数 block，但存在可筛选的低错误区域。

这说明：直接对全 latent 位置均匀嵌入并不理想，应当先做位置筛选。

### 2.2 Channel 维度的差异

之前的错误统计中，部分 channel 的 BER 长期偏高。尤其在热力图中可以看到某些 channel 行整体更亮，说明这些 channel 上 VAE roundtrip 后的 `z_s'` 更容易跨过解码边界。

因此当前实现支持：

```bash
--exclude_channels 1,2
```

含义是：

```text
channel 1 和 channel 2 不嵌入秘密信息
只在其余 channel 中选择 latent 位置
```

被排除的 channel 会保留原始 `sender_zs`，不计入 BER，也不参与 payload 容量统计。

### 2.3 Time 维度的差异

错误热力图显示，某些时间段明显更不稳定。例如在之前的结果中，靠后的 time index 区域出现更高错误率。这可能与音频内容、mel/VAE 编码稳定性、尾部能量、静音段或声学结构变化有关。

因此当前实现支持按秒筛选嵌入区域：

```bash
--embed_start_sec 2
--embed_end_sec 6
```

表示只在 `[2s, 6s)` 对应的 latent time bins 中嵌入。

映射关系为：

```text
latent_time_index = time_sec / duration * latent_T
```

其中：

```text
duration = 10s
latent_T = 256
```

所以 2s 到 6s 大约对应：

```text
time_start_idx = floor(2 / 10 * 256) = 51
time_end_idx   = ceil(6 / 10 * 256)  = 154
```

### 2.4 筛选后的意义

筛选 channel 和 time 后，实际嵌入位置变为：

```text
used_positions = selected channel ∩ selected time window ∩ all freq bins
```

后续所有统计都应只针对 `used_positions`：

```text
overall BER
by_channel.csv
by_time.csv
by_block.csv
bit_comparison.csv
heatmaps
```

否则未嵌入的 channel/time 会污染统计结果。

---

## 3. 当前脚本支持情况

### 3.1 主实验脚本

主脚本：

```text
tools/verify_ddim_reversibility_v8.py
```

已支持：

```bash
--exclude_channels 1,2
--embed_start_sec 2
--embed_end_sec 6
```

并会保存：

```text
selected_positions
used_positions
exclude_channels
position_metadata
```

其中 `position_metadata` 包括：

```text
embed_start_sec
embed_end_sec
time_start_idx
time_end_idx
```

### 3.2 VAE-only 提取脚本

脚本：

```text
tools/ablate_vae_roundtrip_full_v8.py
```

会读取主实验 `.pt` 中的 `used_positions`，只在实际嵌入位置上统计 bit accuracy。

### 3.3 错误分布分析脚本

脚本：

```text
tools/analyze_vae_bit_error_distribution_v2.py
```

会读取：

```text
used_positions
exclude_channels
position_metadata
```

并输出：

```text
vae_bit_error_distribution_report.json
by_channel.csv
by_time.csv
by_freq.csv
by_block.csv
bit_comparison.csv
bit_errors_only.csv
heatmap_channel_time.png
heatmap_channel_freq.png
heatmap_time_freq.png
heatmap_group_blocks.png
```

其中 `bit_comparison.csv` 可以直接查看：

```text
message_index
flat_index
channel
time
freq
time_sec_start
time_sec_end
embedded_bit
decoded_bit
error
recovered_z
embedded_z
abs_z_diff
```

---

## 4. 为什么需要纠错码

当前筛选后 bit accuracy 约为 91%，也就是：

```text
BER ≈ 0.09
```

这说明单个 bit 的可靠性已经较高，但如果 payload 很长，整体完全正确的概率仍然会很低。

例如 payload 长度为 1000 bits，如果每个 bit 独立正确概率为 0.91，则全部正确概率为：

```text
0.91^1000 ≈ 1.15e-41
```

所以即使 bit accuracy 很高，也必须引入纠错码，才能保证最终消息完整恢复。

纠错码的目标不是让每个嵌入 bit 都正确，而是允许一定数量的 bit 错误，然后在解码端恢复原始 payload：

```text
payload bits
-> ECC encode
-> coded bits
-> latent embedding
-> extraction
-> noisy coded bits
-> ECC decode
-> recovered payload bits
```

---

## 5. 推荐纠错码方案

### 5.1 首选：BCH 码

BCH 码适合当前阶段，原因是：

1. 输入输出都是二进制 bit，和当前 latent bit 嵌入完全匹配。
2. 可以硬判决解码，不需要 soft information。
3. 实现相对简单。
4. 对随机 bit error 有明确纠错能力。

BCH 码通常记作：

```text
BCH(n, k, t)
```

其中：

```text
n = 编码后 codeword 长度
k = 原始 payload 长度
t = 每个 codeword 最多可纠正的 bit 错误数
```

如果当前 BER 约为 9%，对长度 `n=255` 的 codeword，平均错误数为：

```text
255 * 0.09 = 22.95 bits
```

因此 `t` 至少应接近或大于 23，才有较高概率成功纠错。

可优先测试：

| BCH 方案 | 近似码率 | 适用情况 |
|---|---:|---|
| BCH(255, 131, t≈18) | 51.4% | BER 降到 6%-7% 后可尝试 |
| BCH(255, 87, t≈26) | 34.1% | 当前 BER≈9% 的推荐起点 |
| BCH(255, 63, t≈30+) | 24.7% | 更保守，适合冲 100% |

注意：不同 BCH 库暴露的参数形式可能不同。有的库通过 `m` 和 `t` 构造 BCH，其中：

```text
n = 2^m - 1
```

例如 `m=8` 时：

```text
n = 255
```

实际 `k` 由库根据 `m,t` 和生成多项式决定。

### 5.2 需要交织 interleaving

纠错码前建议加入随机交织。原因是当前错误有 channel/time/block 结构，不完全是独立随机错误。

如果不交织，某个不稳定时间块可能会让同一个 BCH codeword 中出现大量连续错误，超过纠错上限。

推荐流程：

```text
payload bits
-> BCH encode
-> coded bits
-> random interleaving
-> latent positions embedding
-> extraction
-> inverse interleaving
-> BCH decode
-> recovered payload bits
```

交织的 permutation 应由 `message_seed` 或单独的 `ecc_seed` 决定，发送端和接收端必须一致。

### 5.3 Reed-Solomon 码不作为第一选择

Reed-Solomon 适合 symbol 或 byte 级错误，例如：

```text
RS(255, 127)
```

可纠正：

```text
(255 - 127) / 2 = 64 byte errors
```

但当前错误是 bit-level。若 BER=9%，一个 byte 完全正确的概率为：

```text
(1 - 0.09)^8 ≈ 0.47
```

也就是说 byte error rate 约为：

```text
53%
```

这对 RS 来说非常重。除非先通过 bit-level 交织、筛选或更强的判决降低 byte error，否则 RS 直接用于当前 bit 流并不划算。

### 5.4 LDPC / Polar 作为后续优化

LDPC 和 Polar 码理论上更强，尤其当接收端能提供 soft information 时。

当前每个 bit 的 soft information 可以来自：

```text
margin = |z_s' - threshold|
```

对于 1 bit per latent，默认 threshold 通常是 0：

```text
bit = 0 if z_s' < 0
bit = 1 if z_s' >= 0
```

所以：

```text
margin = |z_s'|
```

margin 越大，说明该 bit 越远离判决边界，可信度越高。

如果未来使用 LDPC soft decoding，可以把 `z_s'` 或 `margin` 转成 log-likelihood ratio：

```text
LLR ≈ scale * z_s'
```

不过 LDPC / Polar 的实现、调参和码率选择更复杂，不建议作为当前第一步。

---

## 6. 推荐实现路线

### 6.1 第一阶段：BCH + 随机交织

新增参数建议：

```bash
--ecc none|bch
--bch_m 8
--bch_t 26
--ecc_seed 1234
```

发送端流程：

```text
1. 随机生成 payload_bits
2. BCH encode 得到 coded_bits
3. 用 ecc_seed 生成 permutation
4. coded_bits 按 permutation 写入 selected latent positions
5. 生成 stego_x1, stego_x0, stego wav
6. 保存 payload_bits、coded_bits、permutation、BCH 参数
```

接收端流程：

```text
1. 从 VAE 得到 x0'
2. 得到 x1'
3. x2' + x1' 得到 z_s'
4. 解码得到 noisy_coded_bits
5. inverse permutation
6. BCH decode
7. 比较 recovered_payload_bits 和 payload_bits
```

需要统计：

```text
raw coded bit accuracy
raw coded BER
BCH decode success
payload bit accuracy
payload BER
failed codewords
corrected bit count
uncorrectable codeword count
effective capacity bits
effective capacity bytes
code rate
```

### 6.2 第二阶段：筛选强度与 BCH 强度联合扫描

建议扫描：

```text
channel selection:
  exclude none
  exclude 1,2
  exclude high-BER channels

time window:
  full duration
  stable middle segment
  top-k low-BER time blocks

BCH strength:
  t = 18
  t = 21
  t = 26
  t = 30+
```

目标是找到容量和准确率的折中：

```text
最大 payload capacity
同时 payload recovery accuracy = 100%
```

### 6.3 第三阶段：基于 margin 的位置选择

除了 channel/time 筛选，还可以根据接收端或校准集统计得到的位置稳定性选择嵌入区域。

对每个 latent 位置或 block 统计：

```text
BER_g
RMSE_g
mean_abs_g
margin_g
```

然后打分：

```text
S_g =
lambda_1 * (1 - BER_g)
+ lambda_2 * 1 / (RMSE_g + epsilon)
+ lambda_3 * margin_g
```

选择分数最高的 group 嵌入。

这种方法可以进一步降低 raw BER，减少 BCH 冗余，提高最终容量。

---

## 7. 容量估算

当前 latent 形状：

```text
[1, 8, 256, 16]
```

总 latent positions：

```text
8 * 256 * 16 = 32768
```

如果 `bits_per_z=1`，全量容量为：

```text
32768 bits = 4096 bytes
```

若排除 channel 1 和 2，只剩 6 个 channel：

```text
6 * 256 * 16 = 24576 bits = 3072 bytes
```

若只嵌入 2s 到 6s，duration=10s，对应约 40% 时间：

```text
24576 * 0.4 ≈ 9830 bits ≈ 1228 bytes
```

如果使用 BCH，最终 payload 容量还要乘以码率。

例如 BCH 码率约 34%：

```text
9830 * 0.34 ≈ 3342 bits ≈ 417 bytes
```

所以纠错后的真实容量大致为：

```text
有效容量 = 可用嵌入位置数 * bits_per_z * ECC_code_rate
```

---

## 8. 预期实验结果

如果 raw BER 约为 9%，合理预期为：

| 方案 | Raw BER | Payload 恢复 | 容量 |
|---|---:|---:|---:|
| 无 ECC | 9% | 长消息几乎不可能 100% | 最高 |
| BCH 较弱 | 9% | 可能仍有失败 codeword | 中高 |
| BCH(255, t≈26) + interleaving | 9% | 有希望接近或达到 100% | 中 |
| BCH 更强 + interleaving | 9% | 更稳 | 较低 |
| 位置筛选 + BCH + interleaving | 更低 | 最有希望稳定 100% | 取决于筛选强度 |

实际是否达到 100%，关键取决于：

```text
1. 错误是否被交织打散
2. 每个 codeword 的错误数是否小于 BCH 纠错上限 t
3. 是否存在系统性 bit 偏置，例如 bit1 明显更容易错
4. 筛选区域在不同 prompt/transcription 下是否稳定
```

---

## 9. 推荐下一步实验

建议按以下顺序进行：

### Step 1：固定当前最优筛选区域

例如：

```bash
--exclude_channels 1,2
--embed_start_sec 2
--embed_end_sec 6
--repeat_k 1
```

先确认多次 seed 下 raw BER 是否稳定在 9% 左右。

### Step 2：加入 BCH(255, t≈26)

先测试：

```text
BCH m=8, t=26
```

记录：

```text
raw coded BER
payload BER
decode success rate
failed codewords
```

### Step 3：加入 interleaving

比较：

```text
BCH without interleaving
BCH with interleaving
```

理论上 interleaving 后，burst/block 错误会被打散，BCH 成功率应提升。

### Step 4：降低码率冲 100%

如果 BCH t=26 仍不能 100%，尝试更强纠错：

```text
t=30+
```

或者进一步缩小嵌入区域，只选低 BER channel/time/block。

### Step 5：引入 margin-based selection

在 `bit_comparison.csv` 中分析：

```text
abs_z_diff
recovered_z
embedded_z
error
channel/time/freq
```

选择 margin 更大的位置嵌入，使 raw BER 从 9% 进一步降到 5% 以下。若 raw BER 能降到 5%，BCH 的冗余需求会明显下降。

---

## 10. 总结

当前实验已经证明：

```text
channel/time 筛选可以显著提升 VAE-only 提取准确率
```

从约 60% 提升到约 91%，说明错误具有可利用的空间结构。接下来要达到最终消息 100% 恢复，最合适的路线是：

```text
位置筛选
+ BCH 纠错码
+ 随机交织
+ payload-level accuracy 统计
```

推荐优先实现：

```text
BCH(255, t≈26) + interleaving
```

如果仍不能稳定 100%，再进一步：

```text
增强 BCH 纠错能力
或缩小嵌入位置到更稳定的 channel/time/block
或使用 margin-based position selection
```

最终实验评价应从单纯的 bit accuracy 转向：

```text
payload recovery success = 100%
effective payload capacity
code rate
failed codeword count
```

这更符合真实隐写系统对“秘密信息完整恢复”的要求。
