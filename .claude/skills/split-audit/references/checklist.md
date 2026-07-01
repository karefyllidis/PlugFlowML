# Split Audit Checklist

- [ ] Split method documented and matches the prediction task
- [ ] Manifest or seed recorded and versioned
- [ ] Primary entity key identified
- [ ] Zero exact duplicate keys across splits
- [ ] Near-duplicate policy checked (defined and applied)
- [ ] No temporal leakage (train only uses past if required)
- [ ] No group leakage (entire groups in one split)
- [ ] No label/feature leakage columns
- [ ] Holdout untouched during model selection
- [ ] Metrics labeled by split name
