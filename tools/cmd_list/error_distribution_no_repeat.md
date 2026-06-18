CUDA_VISIBLE_DEVICES=7 python verify_ddim_reversibility_v7.py --model_name audioldm2-speech-ljspeech -t "A female reporter is speaking full of emotion" --transcription "Success is not final, failure is not fatal. It is the courage to continue that truly counts. Keep moving forward." --ddim_steps 100 --ddim_eta 0 --guidance_scale 3.5 --seed 0 --bits_per_z 1 --repeat_k 1 --interleave_repetition --message_seed 1234 --save_pt outputs_v7_Success/ddim_bits_1b_10s.pt
CUDA_VISIBLE_DEVICES=7 python ablate_vae_roundtrip_full_v1.py --model_name audioldm2-speech-ljspeech --input_pt outputs_v7_Success/ddim_bits_1b_10s.pt --vae_encode_mode mode --save_pt outputs_v7_Success/ablate_vae_extract_bits_test.pt
python analyze_vae_bit_error_distribution.py \
  --vae_pt outputs_v7_Success/ablate_vae_extract_bits_test.pt \
  --input_pt outputs_v7_Success/ddim_bits_1b_10s.pt \
  --out_dir outputs_v7_Success/error_distribution_no_repeat \
  --export_bits \
  --save_heatmaps \
  --bit_export_limit 1000
