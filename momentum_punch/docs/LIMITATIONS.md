# Limitations

- No real historical text/price dataset is bundled in this patch.
- RSS feeds do not provide a reliable long backfill.
- X/Twitter must not be a core historical source without licensed access and
  reproducible timestamps.
- The selected ETF universe has unequal listing histories; the real-data loader
  must enforce an explicit common-window or dynamic-universe policy.
- LLM scores remain model-dependent and require a human-audited validation set.
- Parameter values are hypotheses, not optimized or recommended settings.
