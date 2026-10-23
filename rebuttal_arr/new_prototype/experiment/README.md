# Filtered 100-step watermark robustness experiment

This directory is a standalone adapter around the existing paper outputs and
the random-walk obfuscator. It does not modify either input tree.

The frozen experiment selects original watermarked configurations satisfying
both `AUROC > 0.8` and `Pass@1 > 0.8 * matched no-WM Pass@1`. Each selected
program follows one deterministic 100-step trajectory. Programs 0 through 100
are stored, while only program 100 is executed and scored. The AUROC negative
distribution is the paper's standard normal distribution.

Outputs are written under:

```text
new_prototype/data/watermark_attack/rw100-useful-v1/
```

Create the immutable manifest before submitting Slurm jobs:

```bash
cd rebuttal_arr/new_prototype
python -m experiment.run manifest
bash experiment/slurm/submit.sh
```
