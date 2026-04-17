# Dataset Shards

This directory stores GitHub-safe raw shards for:

- `canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl`
- `release_20260412/canonical_20260412_decontam.tar.zst.part-*`
- `release_20260416_v6/kernelcuterl_20260416_v6_clean.tar.zst.part-*`
- `release_20260417_v12/kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl.part-*` (latest simplified dataset release)

## Files

- `canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.part-000` ... `part-011`
- `canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.sha256`

## Verify

```bash
cd dataset_shards
sha256sum -c canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.sha256
```

## Rebuild JSONL From Parts

```bash
cd dataset_shards
cat canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.part-* \
  > canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl
```

## Latest Release (2026-04-17, v12)

```bash
cd dataset_shards/release_20260417_v12
sha256sum -c kernel_dataset_final_v1_simplified_v12_clusterfix4.parts.sha256
cat kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl.part-* \
  > kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl
echo "$(cat kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl.sha256)  kernel_dataset_final_v1_simplified_v12_clusterfix4.jsonl" | sha256sum -c -
```

## Previous Release (2026-04-12)

```bash
cd dataset_shards/release_20260412
sha256sum -c canonical_20260412_decontam.parts.sha256
cat canonical_20260412_decontam.tar.zst.part-* > canonical_20260412_decontam.tar.zst
sha256sum -c canonical_20260412_decontam.tar.zst.sha256
zstd -d canonical_20260412_decontam.tar.zst -o canonical_20260412_decontam.tar
tar -xf canonical_20260412_decontam.tar
```
