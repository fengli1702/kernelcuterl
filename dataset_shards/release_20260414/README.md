# release_20260414

This release updates the canonical dataset by repairing `query` completeness:
- every sample `query` now contains the full prompt including `Reference implementation` code block
- 819 rows were backfilled (`computeeval=544`, `tritonbench=178`, `cudabench=97`)

Files:
- `canonical_20260414_decontam_queryfull.tar.zst.part-*`: compressed dataset archive shards
- `canonical_20260414_decontam_queryfull.tar.zst.sha256`: archive checksum
- `canonical_20260414_decontam_queryfull.parts.sha256`: shard checksums
- `manifest.txt`: release metadata
- `final_dataset_report_zh.md`: full report
- `eval_contamination_ir_embedding_report.json`: contamination scan report
- `query_full_backfill_report.json`: query completeness verification report
