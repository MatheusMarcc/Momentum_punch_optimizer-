# Patch application notes

This package is designed to replace/add files under the existing
`momentum_punch/` repository directory.

Recommended branch:

```bash
git checkout -b audit/pre-relatorio-2026
rsync -av momentum_punch_winner_patch/momentum_punch/ ./momentum_punch/
cd momentum_punch
python -m pip install -r requirements-dev.txt
pytest
git add .
git commit -m "fix: harden temporal integrity, costs and auditability"
```

Do not merge before reviewing real-data assumptions.
