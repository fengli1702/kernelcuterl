# Dataset Shards

This directory stores GitHub-safe raw shards for:

- `canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl`

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
