
CUDA_VISIBLE_DEVICES=7 python verify_ddim_reversibility_v9.py \
  --model_name audioldm2-speech-ljspeech \
  -t "A female is speaking full of emotion" \
  --transcription "Success is not final, failure is not fatal. It is the courage to continue that truly counts. Keep moving forward." \
  --ddim_steps 100 \
  --ddim_eta 0 \
  --guidance_scale 3.5 \
  --seed 0 \
  --bits_per_z 1 \
  --repeat_k 1 \
  --exclude_channels 0,1,2 \
  --embed_start_sec 0 \
  --embed_end_sec 7 \
  --message_seed 1234 \
  --save_pt outputs_v9/ddim_bits_1b_no_ch0_1_2_0s_7s.pt \
  --save_cover_wav outputs_v9/cover_no_ch0_1_2_0s_7s.wav \
  --save_stego_wav outputs_v9/stego_no_ch0_1_2_0s_7s.wav


然后跑 VAE-only 提取：

CUDA_VISIBLE_DEVICES=7 python ablate_vae_roundtrip_full_v9.py \
  --input_pt outputs_v9/ddim_bits_1b_no_ch0_1_2_0s_7s.pt \
  --model_name audioldm2-speech-ljspeech \
  --vae_encode_mode mode \
  --save_pt outputs_v9/ablate_vae_no_ch0_1_2_0s_7s.pt



VAE-only roundtrip ablation report
{
  "settings": {
    "input_pt": "outputs_v9/ddim_bits_1b_no_ch0_1_2_0s_7s.pt",
    "model_name": "audioldm2-speech-ljspeech",
    "vae_encode_mode": "mode",
    "seed": 0,
    "ddim_steps": 100,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female is speaking full of emotion",
    "transcription": "Success is not final, failure is not fatal. It is the courage to continue that truly counts. Keep moving forward.",
    "bits_per_z": 1,
    "exclude_channels": [
      0,
      1,
      2
    ],
    "selected_positions": 14400,
    "used_positions": 14400,
    "embed_start_sec": 0.0,
    "embed_end_sec": 7.0,
    "time_start_idx": 0,
    "time_end_idx": 180
  },
  "x0_from_sender_mel_vs_stego_x0": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.29581499099731445,
    "mean_abs": 0.03477591276168823,
    "rmse": 0.0501842126250267
  },
  "mel_redecoded_vs_sender_mel": {
    "shape": [
      1,
      1,
      1024,
      64
    ],
    "max_abs": 0.2996711730957031,
    "mean_abs": 0.010757171548902988,
    "rmse": 0.01768973469734192
  },
  "vae_channel_extraction": {
    "num_symbols": 14400,
    "bits_per_z": 1,
    "num_bits": 14400,
    "bit_accuracy": 0.9125694632530212,
    "bit_error_rate": 0.08743053674697876,
    "symbol_accuracy": 0.9125694632530212,
    "symbol_error_rate": 0.08743053674697876,
    "bit_errors": 1259,
    "symbol_errors": 1259,
    "median_threshold": -0.1963496059179306,
    "median_threshold_decode": {
      "num_symbols": 14400,
      "bits_per_z": 1,
      "num_bits": 14400,
      "bit_accuracy": 0.9276388883590698,
      "bit_error_rate": 0.07236111164093018,
      "symbol_accuracy": 0.9276388883590698,
      "symbol_error_rate": 0.07236111164093018,
      "bit_errors": 1042,
      "symbol_errors": 1042
    },
    "oracle_best_threshold": -0.1906210482120514,
    "oracle_threshold_decode": {
      "num_symbols": 14400,
      "bits_per_z": 1,
      "num_bits": 14400,
      "bit_accuracy": 0.9283333420753479,
      "bit_error_rate": 0.0716666579246521,
      "symbol_accuracy": 0.9283333420753479,
      "symbol_error_rate": 0.0716666579246521,
      "bit_errors": 1032,
      "symbol_errors": 1032
    },
    "class_stats": {
      "z_mean_for_bit0": -0.8191174268722534,
      "z_std_for_bit0": 0.46752142906188965,
      "z_mean_for_bit1": 0.4182995855808258,
      "z_std_for_bit1": 0.440893292427063
    },
    "vae_x1_prime_vs_stego_x1": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 0.2940082550048828,
      "mean_abs": 0.03621712327003479,
      "rmse": 0.05143251642584801
    },
    "vae_recovered_zs_vs_embedded_zs": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 5.360703468322754,
      "mean_abs": 0.6607637405395508,
      "rmse": 0.9384942054748535
    }
  }
}
Saved tensors to outputs_v9/ablate_vae_no_ch0_1_2_0s_7s.pt



python analyze_vae_bit_error_distribution_v3.py \
  --vae_pt outputs_v9/ablate_vae_no_ch0_1_2_0s_7s.pt \
  --input_pt outputs_v9/ddim_bits_1b_no_ch0_1_2_0s_7s.pt \
  --out_dir outputs_v9/error_distribution_no_ch0_1_2_0s_7s \
  --export_bits \
  --save_heatmaps
