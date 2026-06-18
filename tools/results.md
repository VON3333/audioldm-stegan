AudioLDM2 DDIM reversibility report -50
{
  "settings": {
    "model_name": "audioldm2-speech-ljspeech",
    "seed": 0,
    "ddim_steps": 50,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female reporter is speaking full of emotion",
    "transcription": "Wish you have a good day",
    "last_three_ddpm_timesteps": [
      21,
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
    "max_abs": 0.06901095062494278,
    "mean_abs": 0.0036780002992600203,
    "rmse": 0.005155459977686405
  },
  "ddim_inversion_x2_prime_vs_sender_x2": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.8878087997436523,
    "mean_abs": 0.08994927257299423,
    "rmse": 0.11510290205478668
  },
  "zs_prime_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 46.940460205078125,
    "mean_abs": 0.8173935413360596,
    "rmse": 2.3577866554260254
  },
  "zs_from_replayed_x2_and_inverted_x1_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 1.2587952613830566,
    "mean_abs": 0.06708861142396927,
    "rmse": 0.094038225710392
  },
  "sender_zs_vs_sender_e_t": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 4.291534423828125e-06,
    "mean_abs": 5.194357299842522e-07,
    "rmse": 8.0040058492159e-07
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
  }
}
Saved traced tensors to outputsddim_reversibility.pt

AudioLDM2 DDIM reversibility report -100
{
  "settings": {
    "model_name": "audioldm2-speech-ljspeech",
    "seed": 0,
    "ddim_steps": 100,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female reporter is speaking full of emotion",
    "transcription": "Wish you have a good day",
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
    "max_abs": 0.03838180750608444,
    "mean_abs": 0.0035484852269291878,
    "rmse": 0.004564377013593912
  },
  "ddim_inversion_x2_prime_vs_sender_x2": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.414145290851593,
    "mean_abs": 0.050311993807554245,
    "rmse": 0.06380337476730347
  },
  "zs_prime_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 23.56169891357422,
    "mean_abs": 0.4605868458747864,
    "rmse": 1.1473493576049805
  },
  "zs_from_replayed_x2_and_inverted_x1_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.700103759765625,
    "mean_abs": 0.06472618132829666,
    "rmse": 0.08325657248497009
  },
  "sender_zs_vs_sender_e_t": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 4.410743713378906e-06,
    "mean_abs": 5.142720738149364e-07,
    "rmse": 7.921976816760434e-07
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
  }
}