# release_20260414 (updated 2026-04-15)

This release is replaced with `queryfull_en_v2`:
- full prompt query with embedded reference code (`query_full` repair)
- query text cleaned to English (`v2`), while keeping `reference_code` aligned in query

Files:
- `canonical_20260414_decontam_queryfull_en_v2.tar.zst.part-*`: compressed dataset archive shards
- `canonical_20260414_decontam_queryfull_en_v2.tar.zst.sha256`: archive checksum
- `canonical_20260414_decontam_queryfull_en_v2.parts.sha256`: shard checksums
- `manifest.txt`: release metadata
- `final_dataset_report_zh.md`: full report
- `eval_contamination_ir_embedding_report.json`: contamination scan report
- `query_full_backfill_report.json`: full-query repair verification
- `query_english_clean_report_20260415.json`: english clean pass-1 report
- `query_english_clean_report_20260415_v2.json`: english clean pass-2 report
