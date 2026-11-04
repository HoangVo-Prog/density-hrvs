# Reproduce the larger-model experiment

The selected deployment configuration uses official Qwen3-VL-8B-Instruct revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, NF4 double quantization and bfloat16 compute.
The 2B default remains available. Actual IDs, revisions and quantization are saved.
Exact package versions used here are in `experiment_runtime_versions.json`.
The model cache on this machine is `/media/tinhanhnguyen/sub/model_cache`.

Download the official checkpoint to the secondary drive:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --cache-dir /media/tinhanhnguyen/sub/model_cache --exclude '*.bin'
```

The 12 GB GPU cannot keep 8B and both FP32 vision models resident. Use
`model_residency=staged`; each model executes on CUDA and then returns to RAM.
This adds transfer time while preserving its weights and input resolution.
Install `bitsandbytes` and `accelerate` in the experiment environment for NF4.

Run the requested full export from the repository with the verified experiment Python:

```bash
python -B -u -m algorithm.tools.run_experiment \
  --data-dir /home/tinhanhnguyen/Desktop/project/cvpr/data/data_official_official \
  --output-dir /media/tinhanhnguyen/sub/outputs/algo1_improve/official_official_qwen8_full \
  --config algorithm/configs/qwen8_nf4_staged.json
```

The runner saves the exact executable source and input hashes before starting one serial
GPU worker. Repeat the same command to resume saved successes using that frozen source.
Changing the dataset or requested configuration requires a new output directory.
Read `benchmark.log`, `run_state.json`, `summary.json`, and individual `completion.json`
files for progress. Existence of a state file alone does not prove a process is live.

Inference receives exactly image ID, question ID and question. Target boxes/answers are
read only by post-prediction inspection. The new export has 501 images and 1,060 questions;
134 questions on 67 images duplicate development. Report its 926 additional-image
questions separately. The 134-case frozen-box 8B answer comparison improves the exact
complete-answer diagnostic from 33 to 65 using legacy crops, but this is not an end-to-end
localization gain or a replacement for the inherited semantic grading rubric.
