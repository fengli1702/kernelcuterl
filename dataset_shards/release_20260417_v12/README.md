# release_20260417_v12

This folder contains the latest dataset package we are using now:

- base name: `kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl`
- rows: `20586`
- split into GitHub-safe parts (`part-000` ... `part-005`)

## Files

- `kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl.part-*`
- `kernel_dataset_final_v1_simplified_v12_clusterfix4.parts.sha256`
- `kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl.sha256`
- `release_manifest_v12_clusterfix4.json`
- `source_distribution.json`
- `release_manifest_simplified_v2.json` (upstream trace file)
- `full_static_interface_validation_20260417.json` (upstream trace file)
- `ref_smoke_check_simplified_v2_per_source3.json` (upstream trace file)
- `ref_smoke_summary_simplified_v2.md` (upstream trace file)

## Environment Requirements

### For dataset use only (merge/verify/filter)

- Linux (`x86_64`)
- `bash` + `coreutils` (`cat`, `sha256sum`, `wc`, `split`)
- Python `>=3.10`

### For reference runtime smoke/eval

- Docker `>=24`
- NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- NVIDIA driver with healthy NVML (`nvidia-smi` should respond)
- CUDA-capable GPU
- Python `>=3.10`
- PyTorch + Triton runtime matching your evaluation scripts

Note: if `docker run --gpus ...` hangs or returns `context canceled`, fix host GPU runtime first, then run smoke checks.

## Rebuild + Verify

```bash
cd dataset_shards/release_20260417_v12

# 1) verify each part
sha256sum -c kernel_dataset_final_v1_simplified_v12_clusterfix4.parts.sha256

# 2) rebuild full jsonl
cat kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl.part-* \
  > kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl

# 3) verify rebuilt file
echo "$(cat kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl.sha256)  kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl" | sha256sum -c -
```
