# AudioLDM2 Latent Steganography Experiment Summary

This document summarizes the full experimental path for testing a TASDF-Stega-like
latent steganography scheme on AudioLDM2.

The main model used in the experiments:

```text
model_name      = audioldm2-speech-ljspeech
task            = text-to-speech
text prompt     = A female reporter is speaking full of emotion
transcription   = Wish you have a good day
ddim_eta        = 0
guidance_scale  = 3.5
duration        = 10 s
latent shape    = [1, 8, 256, 16]
latent elements = 32768
```

The current AudioLDM2 generation chain is:

```text
DDIM diffusion latent x0
  -> VAE decoder / AutoencoderKL.decode()
  -> mel/fbank
  -> HiFi-GAN vocoder
  -> wav
```

The reverse audio-side chain tested later is:

```text
wav
  -> wav_to_fbank()
  -> mel/fbank
  -> VAE encoder
  -> x0'
  -> DDIM inversion
  -> x1'
  -> z_s'
  -> bits'
```

## 1. Initial Goal

The initial goal was to verify whether AudioLDM2's DDIM diffusion process can
support a TASDF-Stega-style latent embedding and extraction process.

The target protocol was:

```text
Sender:
  normal diffusion generation
  keep x2, x1, x0
  keep z_s for the x2 -> x1 transition

Receiver:
  replay normal diffusion to obtain x2'
  obtain x0 or x0'
  invert x0 / x0' to x1'
  combine x2' and x1' to recover z_s'
  compare x2/x2', x1/x1', z_s/z_s'
```

The core code under inspection was:

```text
audioldm2/latent_diffusion/models/ddim.py
```

The main experimental script developed for this was:

```text
tools/verify_ddim_reversibility.py
```

## 2. DDIM Determinism and the Role of `ddim_eta`

AudioLDM2's default generation path uses DDIM sampling. A crucial observation was:

```text
ddim_eta = 1.0
```

injects additional random noise at each DDIM step, so strict reversibility is not
expected.

For all reversibility and steganography experiments, we therefore used:

```text
ddim_eta = 0
```

With `ddim_eta=0`, DDIM becomes deterministic. There is no extra random Gaussian
noise term in the transition:

```text
x_{t-1}
  = sqrt(alpha_{t-1}) * pred_x0
  + sqrt(1 - alpha_{t-1}) * eps_theta
```

In this setting, the `z_s` term used in the experiment is not a newly sampled
noise variable. It is the denoising direction/noise prediction term for the
transition:

```text
z_s = eps_theta(x2, t, condition)
```

or, equivalently, the term recovered from:

```text
z_s =
  (x1 - sqrt(alpha_1) * pred_x0) / sqrt(1 - alpha_1)
```

## 3. Reversibility Experiment

### 3.1 First Test: 50 DDIM Steps

Command parameters:

```text
ddim_steps = 50
ddim_eta   = 0
```

The last three DDIM positions corresponded to:

```text
[21, 1, 0]
```

Results:

| Metric | max_abs | mean_abs | rmse | Interpretation |
|---|---:|---:|---:|---|
| replayed `x2'` vs sender `x2` | 0.0000 | 0.0000 | 0.0000 | deterministic replay is exact |
| inverted `x1'` vs sender `x1` | 0.0689 | 0.00368 | 0.00515 | one-step inversion has small error |
| inverted `x2'` vs sender `x2` | 0.8894 | 0.08995 | 0.1151 | two-step inversion error grows |
| `z_s'` from inverted `x2'` + `x1'` vs `z_s` | 46.96 | 0.8177 | 2.3585 | unusable |
| `z_s'` from replayed `x2'` + inverted `x1'` vs `z_s` | 1.2588 | 0.0671 | 0.0940 | much better, but not exact |

Key insight:

```text
The receiver should not use x2 obtained from x0 inversion.
It should use replayed x2' and combine it with x1' inverted from x0.
```

This matched the intended route:

```text
x2' = replayed deterministic diffusion state
x1' = inverted from x0
z_s' = function(x2', x1')
```

### 3.2 Second Test: 100 DDIM Steps

Command parameters:

```text
ddim_steps = 100
ddim_eta   = 0
```

The last three DDIM positions corresponded to:

```text
[11, 1, 0]
```

Results:

| Metric | max_abs | mean_abs | rmse | Interpretation |
|---|---:|---:|---:|---|
| replayed `x2'` vs sender `x2` | 0.0000 | 0.0000 | 0.0000 | exact deterministic replay |
| inverted `x1'` vs sender `x1` | 0.0384 | 0.00355 | 0.00456 | slightly better than 50-step |
| inverted `x2'` vs sender `x2` | 0.4141 | 0.0503 | 0.0638 | better than 50-step |
| `z_s'` from inverted `x2'` + `x1'` vs `z_s` | 23.56 | 0.4606 | 1.1473 | still unusable |
| `z_s'` from replayed `x2'` + inverted `x1'` vs `z_s` | 0.7001 | 0.0647 | 0.0833 | slightly better |

Conclusion:

```text
AudioLDM2 DDIM sampling is exactly reproducible under fixed seed/condition/params.
However, x0 -> x1 inversion is not exact.
The intended route using replayed x2' is much better than using inverted x2',
but z_s recovery is still approximate.
```

## 4. Real Bit Embedding at Latent Level

After verifying the numerical route, the next step was a true bit experiment:

```text
random bits
  -> Gaussian CDF midpoint z_s
  -> stego_x1
  -> stego_x0
  -> receiver uses x2' + inverted x1'
  -> z_s'
  -> decoded bits
```

### 4.1 Mapping Bits to `z_s`

For `bits_per_z=1`, the mapping was:

```text
bit 0 -> u = 0.25 -> z_s = Phi^{-1}(0.25) ~= -0.674
bit 1 -> u = 0.75 -> z_s = Phi^{-1}(0.75) ~=  0.674
```

This is a midpoint mapping inside the two standard-normal CDF intervals.

### 4.2 Latent-Level Result, 1 Bit per Latent

With:

```text
ddim_steps = 100
bits_per_z = 1
latent elements = 32768
```

Result:

| Metric | Value |
|---|---:|
| num_bits | 32768 |
| bit_accuracy | 0.9998169 |
| bit_error_rate | 0.0001831 |
| bit_errors | 6 |
| capacity | 4096 bytes / 10 s |
| bitrate | 3.28 kbps |

The recovered `z_s'` error against embedded `z_s` was:

| Metric | max_abs | mean_abs | rmse |
|---|---:|---:|---:|
| embedded `z_s` vs recovered `z_s'` | 0.8624 | 0.0363 | 0.0545 |

Conclusion:

```text
If the receiver has stego_x0 directly, 1 bit per latent is highly reliable.
```

## 5. Generating Stego Audio

The next step was to run the full generation path:

```text
bits
  -> z_s
  -> stego_x1
  -> stego_x0
  -> VAE decoder
  -> mel/fbank
  -> HiFi-GAN vocoder
  -> wav
```

Support was added to:

```text
tools/verify_ddim_reversibility.py
```

with:

```bash
--save_stego_wav
--save_cover_wav
```

Important clarification:

```text
stego_x0 -> VAE decoder -> mel -> HiFi-GAN -> wav
```

is not reversible. The VAE decoder is not strictly invertible, and HiFi-GAN is a
vocoder, not a lossless audio codec.

## 6. Full Audio Roundtrip Extraction

The first complete audio-channel extraction experiment was:

```text
Sender:
  bits -> z_s -> stego_x1 -> stego_x0 -> wav

Receiver:
  wav -> fbank/mel -> VAE encoder -> x0'
  x0' -> x1'
  replay x2'
  x2' + x1' -> z_s'
  z_s' -> bits'
```

Result for `bits_per_z=1`:

| Metric | Value |
|---|---:|
| num_bits | 32768 |
| bit_accuracy | 0.5099 |
| bit_error_rate | 0.4901 |
| bit_errors | 16059 |

This is essentially random guessing.

Intermediate errors:

| Metric | max_abs | mean_abs | rmse |
|---|---:|---:|---:|
| `x0'` from wav vs `stego_x0` | 1.5058 | 0.1387 | 0.1929 |
| `x1'` from wav vs `stego_x1` | 1.5253 | 0.1493 | 0.2037 |
| recovered `z_s'` from wav vs embedded `z_s` | 27.82 | 2.7239 | 3.7162 |

Conclusion:

```text
The full wav-channel extraction failed.
The recovered z_s' error is much larger than the 1-bit signal amplitude.
```

## 7. Locating the Bottleneck

To locate where the error enters, two ablations were created:

```text
tools/ablate_vae_roundtrip.py
tools/ablate_vocoder_fbank_roundtrip.py
```

### 7.1 VAE-Only Roundtrip

Experiment:

```text
stego_x0
  -> VAE decoder
  -> sender_mel
  -> VAE encoder
  -> x0'
```

Result:

| Metric | max_abs | mean_abs | rmse |
|---|---:|---:|---:|
| `x0'` from sender mel vs `stego_x0` | 0.1504 | 0.0286 | 0.0356 |
| redecoded mel vs sender mel | 0.2009 | 0.00945 | 0.01445 |

Interpretation:

```text
VAE-only reconstruction error is much smaller than full audio roundtrip error,
but it is still enough to damage the hidden z_s signal.
```

### 7.2 Vocoder/Fbank Roundtrip

Experiment:

```text
sender_mel
  -> HiFi-GAN vocoder
  -> wav
  -> wav_to_fbank()
  -> receiver_mel
```

Result:

| Metric | max_abs | mean_abs | rmse |
|---|---:|---:|---:|
| receiver fbank vs sender decoded mel | 3.2808 | 0.2101 | 0.3485 |

Interpretation:

```text
The mel -> wav -> fbank path is far from identity.
This explains why the full audio-channel extraction collapses to near-random.
```

## 8. VAE-Channel Bit Extraction

After measuring VAE-only reconstruction error, the full extraction was tested
without vocoder/wav:

```text
Sender:
  bits -> z_s -> stego_x1 -> stego_x0 -> VAE decoder -> sender_mel

Receiver:
  sender_mel -> VAE encoder -> x0'
  x0' -> x1'
  replay x2'
  x2' + x1' -> z_s'
  z_s' -> bits'
```

For `bits_per_z=1` without repetition:

| Metric | Value |
|---|---:|
| num_bits | 32768 |
| bit_accuracy | 0.6038 |
| bit_error_rate | 0.3962 |
| bit_errors | 12982 |

Error:

| Metric | max_abs | mean_abs | rmse |
|---|---:|---:|---:|
| `x1'` vs `stego_x1` | 0.1964 | 0.0378 | 0.0471 |
| recovered `z_s'` vs embedded `z_s` | 3.5797 | 0.6892 | 0.8598 |

Conclusion:

```text
Even without HiFi-GAN/wav, the VAE decoder -> encoder path severely weakens
the hidden latent signal.
```

## 9. Random Interleaved Repetition

To trade capacity for robustness, a repetition code with random interleaving was
implemented.

Procedure:

```text
1. Generate message bits.
2. Use message_seed to generate a random permutation over all latent positions.
3. Assign each message bit to repeat_k random latent positions.
4. Embed the same bit into those positions.
5. Decode every latent position.
6. Use majority vote over each group.
```

For `repeat_k=31`:

```text
effective capacity = floor(32768 / 31)
                   = 1057 bits
                   ~= 132 bytes / 10 s
```

### 9.1 Latent-Level Repetition Result

With:

```text
bits_per_z = 1
repeat_k = 31
interleave_repetition = true
```

Result:

| Metric | Value |
|---|---:|
| raw bit_accuracy | 0.9996948 |
| raw bit_errors | 10 / 32768 |
| voted_bit_accuracy | 1.0000 |
| voted_bit_errors | 0 / 1057 |
| effective capacity | 1057 bits ~= 132 bytes |

Interpretation:

```text
At latent level, random interleaved repetition easily corrects the few residual
errors.
```

### 9.2 VAE-Channel Repetition Result

Using the same `repeat_k=31` stego payload, but extracting through:

```text
stego_x0 -> VAE decoder -> mel -> VAE encoder -> x0'
```

Result:

| Metric | Value |
|---|---:|
| raw bit_accuracy | 0.5926 |
| raw bit_errors | 13349 / 32768 |
| voted_bit_accuracy | 0.7550 |
| voted_bit_errors | 259 / 1057 |
| effective capacity | 1057 bits ~= 132 bytes |

Interpretation:

```text
Repetition helps, but errors are strongly correlated/systematic.
The improvement is far below what an independent-error model would predict.
```

## 10. Threshold Calibration

The next hypothesis was that VAE extraction has a global threshold shift. For
`bits_per_z=1`, default decoding uses:

```text
z_s' >= 0 -> bit 1
z_s' <  0 -> bit 0
```

Two threshold calibrations were tested:

```text
median threshold:
  threshold = median(z_s')

oracle raw threshold:
  threshold chosen to minimize raw per-latent bit errors
```

For `repeat_k=31`, results were:

| Decoder | raw bit_accuracy | voted_bit_accuracy | voted errors |
|---|---:|---:|---:|
| fixed threshold 0 | 0.5925 | 0.7521 | 262 / 1057 |
| median threshold -0.153 | 0.5985 | 0.8675 | 140 / 1057 |
| oracle raw threshold -0.296 | 0.5996 | 0.7975 | 214 / 1057 |

Class statistics after VAE-channel extraction:

| Class | mean of `z_s'` | std of `z_s'` |
|---|---:|---:|
| bit 0 | -0.3291 | 0.6700 |
| bit 1 | 0.0044 | 0.6914 |

Interpretation:

```text
The two classes are heavily overlapped.
Median threshold significantly improves majority voting, but raw accuracy barely
changes.
The oracle threshold optimized raw bit accuracy, not voted message accuracy,
so it was not optimal for repetition voting.
```

## 11. Distribution of Repeated `z_s`

With `bits_per_z=1`, the embedded `z_s` values are only:

```text
-0.674 and +0.674
```

Thus the marginal distribution is not a continuous standard Gaussian. It is a
two-point discrete Gaussian-quantile distribution.

With repetition:

```text
each message bit appears in repeat_k latent positions
```

so the joint distribution also contains correlations. Random interleaving spreads
these repeated values over channel/time/frequency positions, but it does not make
them independent.

Trade-off:

```text
midpoint + repetition:
  stronger robustness
  weaker statistical naturalness

random sampling inside Gaussian intervals:
  better statistical naturalness
  worse robustness near decoding boundaries
```

## 12. HiFi-GAN vs WaveGlow Consideration

AudioLDM2 currently uses:

```text
mel/fbank -> HiFi-GAN -> wav
```

The code path is:

```text
audioldm2/utilities/model.py
  get_vocoder()
  name = "HiFi-GAN"
```

Replacing HiFi-GAN with WaveGlow is possible only for:

```text
mel/fbank -> wav
```

It cannot replace:

```text
x0 -> VAE decoder -> mel/fbank
```

because WaveGlow consumes mel spectrograms, not diffusion latents.

Important practical caveat:

```text
Most public WaveGlow checkpoints use 80-bin mel at 22.05 kHz.
AudioLDM2 speech-ljspeech uses 64-bin mel/fbank at 16 kHz.
```

Therefore a useful WaveGlow replacement would likely require training or
fine-tuning WaveGlow with AudioLDM2's exact mel configuration.

Even if WaveGlow improves:

```text
mel -> wav -> mel'
```

it will not solve the already observed VAE bottleneck:

```text
x0 -> VAE decoder -> mel -> VAE encoder -> x0'
```

## 13. Current Conclusions

### 13.1 What Works

At direct latent level:

```text
stego_x0 is available to receiver
```

the method works very well:

```text
1 bit/latent: 99.98% bit accuracy
repeat_k=31: 100% voted accuracy
```

### 13.2 What Fails

At VAE-channel level:

```text
stego_x0 -> VAE decoder -> mel -> VAE encoder -> x0'
```

the hidden signal is heavily degraded:

```text
raw bit accuracy ~= 59-60%
repeat_k=31 voted accuracy ~= 75%
median threshold + repeat_k=31 voted accuracy ~= 86.75%
```

At full audio-channel level:

```text
stego_x0 -> mel -> HiFi-GAN -> wav -> fbank -> VAE encoder -> x0'
```

the method collapses:

```text
bit accuracy ~= 51%
```

### 13.3 Main Bottlenecks

The bottlenecks are:

```text
1. VAE decoder -> encoder does not preserve the steganographic perturbation.
2. HiFi-GAN/wav/fbank roundtrip adds even larger mel-level distortion.
3. Recovered z_s' distributions for bit0 and bit1 overlap heavily after VAE.
```

### 13.4 Practical Interpretation

This AudioLDM2 latent embedding scheme is currently best described as:

```text
effective for latent-channel steganography
not yet effective for true wav-channel steganography
```

To make it practical for wav-channel steganography, one likely needs at least one
of:

```text
1. Embed closer to mel/fbank or waveform features.
2. Train a VAE/audio encoder with steganographic preservation loss.
3. Use stronger error-correcting codes with soft decisions.
4. Add pilots and threshold/linear calibration.
5. Improve or replace the vocoder only after solving the VAE bottleneck.
```

## 14. Suggested Next Experiments

1. Optimize threshold for voted accuracy, not raw bit accuracy.

```text
Current oracle threshold minimizes raw latent bit errors.
Need an oracle threshold that minimizes message-level voted errors.
```

2. Test larger repetition factors:

| repeat_k | capacity bits | capacity bytes / 10 s |
|---:|---:|---:|
| 31 | 1057 | 132 |
| 63 | 520 | 65 |
| 127 | 258 | 32 |
| 255 | 128 | 16 |

3. Add pilot bits for deployable threshold calibration.

4. Try soft-decision decoding:

```text
confidence = abs(z_s' - threshold)
```

instead of hard 0/1 votes.

5. Compare HiFi-GAN and WaveGlow only on:

```text
sender_mel -> vocoder -> wav -> wav_to_fbank -> receiver_mel
```

before integrating WaveGlow into the full pipeline.

## 15. repeat意义

从理论角度分析，这样repeat在多个随机交织分散位置latent嵌入的方式的意义

有意义，但它的意义更像 信道编码/纠错，不是从根本上解决 AudioLDM2 这条链路的不可逆问题。

可以把你的系统抽象成一个噪声信道：

message bit b
 -> 映射到多个 latent 位置 z_i
 -> VAE / inversion / extraction
 -> 每个位置得到 b_i'
 -> 多数投票恢复 b'
如果单个 latent 的判决正确率大于 50%，重复码理论上就能提升最终 bit 准确率。你现在 VAE 通道 raw accuracy 大约：

p ≈ 0.59 ~ 0.60
确实大于随机猜测，所以重复是有理论意义的。

但是效果取决于一个关键假设：

多个重复位置的错误要尽量独立。
如果独立，repeat_k=31 多数投票会非常强。
但你的实测：

raw accuracy ≈ 59.25%
repeat_k=31 voted accuracy ≈ 75.2%
median threshold voted accuracy ≈ 86.75%
说明重复有帮助，但远低于独立错误模型的理想提升。原因是错误存在相关性和系统性偏移。

随机交织的意义就在这里：

把同一个 bit 的 k 个副本分散到不同 channel / time / freq 位置，
避免它们落在同一个局部失真区域里一起错。
所以它比“相邻 latent 重复”更合理。相邻位置往往共享：

相似的 VAE 重建误差
相似的频带失真
相似的 DDIM inversion 误差
相似的局部结构偏移
随机交织可以降低局部相关错误，让多数投票更接近独立假设。

但它也有明显代价和限制：

1. 容量按 repeat_k 线性下降。
2. z_s 的联合分布不再像 i.i.d. 高斯。
3. 如果误差主要是全局偏移/模型性失真，随机交织也救不了。
4. 多数投票只能利用硬判决，没充分利用 zs' 的置信度。
从信息论角度看，重复码是最简单、最低效的纠错码。它适合做早期验证，因为简单直观：

如果 repeat 都救不回来，说明信道很差；
如果 repeat 明显提升，说明信道里仍有可利用信息。
你的实验正好证明了后者：

VAE 通道不是完全无信息；
raw 约 59%，median+repeat 可到 86.75%。
所以理论结论是：

随机交织 repeat 是有意义的，
它证明该 latent 信道存在弱可恢复信号；
但它不是容量效率高的最终方案，
也不能保证真实音频端鲁棒。
更进一步，应该从 repeat 过渡到真正的纠错码：

BCH / Reed-Solomon / LDPC / Polar code
并配合软判决：

用 zs' 离阈值的距离作为置信度
而不是只用 0/1 硬判决
这样理论上会比重复投票更高效。

一句话总结：

有意义，适合作为验证和基线；
但从理论和容量效率看，它只是低效纠错码，后续应换成交织 + 软判决纠错码。

## 16. repeat63结果

repeat_k=63 的结果比 31 明显好，尤其是 median threshold + vote：

repeat_k=31:
median_threshold_vote = 86.75%
errors = 140 / 1057
capacity = 1057 bits ≈ 132 bytes

repeat_k=63:
median_threshold_vote = 93.27%
errors = 35 / 520
capacity = 520 bits ≈ 65 bytes
对比表：

| repeat_k | raw bit acc | fixed vote acc | median vote acc | median vote errors | capacity |
|---:|---:|---:|---:|---:|---:|
| 31 | 59.25% | 75.21% | 86.75% | 140 / 1057 | 132 B / 10s |
| 63 | 58.93% | 76.15% | 93.27% | 35 / 520 | 65 B / 10s |

几个观察：

raw bit accuracy 没变，还是约 59%-60%

说明 VAE 通道本身的单点判别能力没有改善：

bit0 mean ≈ -0.330
bit1 mean ≈  0.004
std ≈ 0.67-0.69
两类还是高度重叠。

repeat_k 增大后，median threshold 投票明显提升

从 86.75% 到 93.27%，说明随机交织重复确实在累积弱信号。


当前结论可以更新为：

VAE 通道下，随机交织重复 + median 阈值是有效的；
repeat_k=63 可把 message-level accuracy 提升到 93.27%，
但容量降到约 65 bytes / 10s。