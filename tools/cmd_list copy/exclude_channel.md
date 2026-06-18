CUDA_VISIBLE_DEVICES=7 python verify_ddim_reversibility_v8.py \
  --model_name audioldm2-speech-ljspeech \
  -t "A female is speaking full of emotion" \
  --transcription "Success is not final, failure is not fatal. It is the courage to continue that truly counts. Keep moving forward." \
  --ddim_steps 100 \
  --ddim_eta 0 \
  --guidance_scale 3.5 \
  --seed 0 \
  --bits_per_z 1 \
  --repeat_k 1 \
  --exclude_channels 1,2 \
  --message_seed 1234 \
  --save_pt outputs_v8/ddim_bits_1b_no_ch1_2.pt \
  --save_cover_wav outputs_v8/cover.wav \
  --save_stego_wav outputs_v8/stego_bits_1b.wav


AudioLDM2 DDIM reversibility report
{
  "settings": {
    "model_name": "audioldm2-speech-ljspeech",
    "seed": 0,
    "ddim_steps": 100,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female is speaking full of emotion",
    "transcription": "Success is not final, failure is not fatal. It is the courage to continue that truly counts. Keep moving forward.",
    "bits_per_z": 1,
    "message_seed": 1234,
    "repeat_k": 1,
    "interleave_repetition": false,
    "exclude_channels": [
      1,
      2
    ],
    "last_three_ddpm_timesteps": [
      11,
      1,
      0
    ]
  },
  "normal_replay_receiver_x2_prime_vs_sender_x2": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "rmse": 0.0
  },
  "ddim_inversion_x1_prime_vs_sender_x1": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.035016655921936035,
    "mean_abs": 0.0015047836350277066,
    "rmse": 0.0022206942085176706
  },
  "ddim_inversion_x2_prime_vs_sender_x2": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.3596419095993042,
    "mean_abs": 0.038154758512973785,
    "rmse": 0.05077002942562103
  },
  "zs_prime_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 28.625818252563477,
    "mean_abs": 1.2399723529815674,
    "rmse": 1.8952401876449585
  },
  "zs_from_replayed_x2_and_inverted_x1_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.6387217044830322,
    "mean_abs": 0.02744802087545395,
    "rmse": 0.040506597608327866
  },
  "sender_zs_vs_sender_e_t": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 4.351139068603516e-06,
    "mean_abs": 4.07675315727829e-07,
    "rmse": 6.901849474161281e-07
  },
  "x0_self_check": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "rmse": 0.0
  },
  "bit_embedding_experiment": {
    "num_symbols": 24576,
    "bits_per_z": 1,
    "num_bits": 24576,
    "bit_accuracy": 0.9992269277572632,
    "bit_error_rate": 0.0007730722427368164,
    "symbol_accuracy": 0.9992269277572632,
    "symbol_error_rate": 0.0007730722427368164,
    "bit_errors": 19,
    "symbol_errors": 19,
    "embedded_zs_recovered_zs_diff": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 1.1877245903015137,
      "mean_abs": 0.07358860969543457,
      "rmse": 0.10936545580625534
    },
    "stego_x1_recovered_x1_diff": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 0.06511461734771729,
      "mean_abs": 0.004034346900880337,
      "rmse": 0.005995740182697773
    }
  }
}

然后跑 VAE-only 提取：

CUDA_VISIBLE_DEVICES=7 python ablate_vae_roundtrip_full_v8.py \
  --model_name audioldm2-speech-ljspeech \
  --input_pt outputs_v8/ddim_bits_1b_no_ch1_2.pt \
  --vae_encode_mode mode \
  --save_pt outputs_v8/ablate_vae_no_ch1_2.pt

VAE-only roundtrip ablation report
{
  "settings": {
    "input_pt": "outputs_v8/ddim_bits_1b_no_ch1_2.pt",
    "model_name": "audioldm2-speech-ljspeech",
    "vae_encode_mode": "mode",
    "seed": 0,
    "ddim_steps": 100,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female is speaking full of emotion",
    "transcription": "Success is not final, failure is not fatal. It is the courage to continue that truly counts. Keep moving forward.",
    "bits_per_z": 1
  },
  "x0_from_sender_mel_vs_stego_x0": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.27657270431518555,
    "mean_abs": 0.03586559370160103,
    "rmse": 0.05049408972263336
  },
  "mel_redecoded_vs_sender_mel": {
    "shape": [
      1,
      1,
      1024,
      64
    ],
    "max_abs": 0.26700496673583984,
    "mean_abs": 0.011138864792883396,
    "rmse": 0.018025444820523262
  },
  "vae_channel_extraction": {
    "num_symbols": 24576,
    "bits_per_z": 1,
    "num_bits": 24576,
    "bit_accuracy": 0.798583984375,
    "bit_error_rate": 0.201416015625,
    "symbol_accuracy": 0.798583984375,
    "symbol_error_rate": 0.201416015625,
    "bit_errors": 4950,
    "symbol_errors": 4950,
    "median_threshold": -0.26628825068473816,
    "median_threshold_decode": {
      "num_symbols": 24576,
      "bits_per_z": 1,
      "num_bits": 24576,
      "bit_accuracy": 0.8181966543197632,
      "bit_error_rate": 0.18180334568023682,
      "symbol_accuracy": 0.8181966543197632,
      "symbol_error_rate": 0.18180334568023682,
      "bit_errors": 4468,
      "symbol_errors": 4468
    },
    "oracle_best_threshold": -0.254738986492157,
    "oracle_threshold_decode": {
      "num_symbols": 24576,
      "bits_per_z": 1,
      "num_bits": 24576,
      "bit_accuracy": 0.8185628652572632,
      "bit_error_rate": 0.18143713474273682,
      "symbol_accuracy": 0.8185628652572632,
      "symbol_error_rate": 0.18143713474273682,
      "bit_errors": 4459,
      "symbol_errors": 4459
    },
    "class_stats": {
      "z_mean_for_bit0": -0.7451148629188538,
      "z_std_for_bit0": 0.6071600317955017,
      "z_mean_for_bit1": 0.19178073108196259,
      "z_std_for_bit1": 0.6632959842681885
    },
    "vae_x1_prime_vs_stego_x1": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 0.27489662170410156,
      "mean_abs": 0.03927459567785263,
      "rmse": 0.054085295647382736
    },
    "vae_recovered_zs_vs_embedded_zs": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 5.016977310180664,
      "mean_abs": 0.7164301872253418,
      "rmse": 0.9865139722824097
    }
  }
}

python analyze_vae_bit_error_distribution_v2.py \
  --vae_pt outputs_v8/ablate_vae_no_ch0_1_2.pt \
  --input_pt outputs_v8/ddim_bits_1b_no_ch0_1_2.pt \
  --out_dir outputs_v8/error_distribution_no_ch0_1_2 \
  --export_bits \
  --save_heatmaps