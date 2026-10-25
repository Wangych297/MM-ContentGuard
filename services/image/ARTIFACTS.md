# Local Artifacts

Large training artifacts are intentionally not committed to GitHub.

Final first-layer model selected locally:

```text
checkpoints/image_model_dinov3_large/best.pt
```

Selection summary:

- model: DINOv3-Large
- selected epoch: 8
- validation accuracy: 0.9569230769230769
- validation macro F1: 0.9338556845608977

Local data manifests under `data/processed/` also stay untracked because they
contain machine-specific absolute paths. Rebuild them with:

```powershell
.\scripts\prepare_data.ps1
```
