# release_20260412

Contents:
- `canonical_20260412_decontam.tar.zst.part-*`: split archive shards (<50MB)
- `canonical_20260412_decontam.tar.zst.sha256`: full archive checksum
- `canonical_20260412_decontam.parts.sha256`: per-part checksums
- `manifest.txt`: source/row metadata

Reconstruct:
```bash
cat canonical_20260412_decontam.tar.zst.part-* > canonical_20260412_decontam.tar.zst
sha256sum -c canonical_20260412_decontam.tar.zst.sha256
zstd -d canonical_20260412_decontam.tar.zst -o canonical_20260412_decontam.tar
tar -xf canonical_20260412_decontam.tar
```

Verify part files:
```bash
sha256sum -c canonical_20260412_decontam.parts.sha256
```
