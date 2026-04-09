# Dataset Shards

This directory stores a GitHub-safe shard package for:

- `canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl`

## Files

- `canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.tar.gz.part-000`
- `canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.sha256`

Note: current compressed size is below 95MB, so split produced only one part.  
For repo size efficiency, only shard parts are committed, not the reconstructed archive.

## Verify

```bash
cd dataset_shards
sha256sum -c canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.sha256
```

## Rebuild Archive From Parts

```bash
cd dataset_shards
cat canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.tar.gz.part-* \
  > canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.tar.gz
```

## Extract JSONL

```bash
cd dataset_shards
tar -xzf canonical_all_plus_xpuoj_ast_near_dedup_exact_no_kernelbench_similar.jsonl.tar.gz
```
