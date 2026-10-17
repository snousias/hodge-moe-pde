# Repository progress journal

This journal preserves completed work, user preferences, and pending tasks for future sessions. Append dated entries when work changes; verify the working tree before continuing from an earlier entry.

## 2026-10-16 Documentation and dataset distribution

### Completed

- Reviewed repository structure, data configuration and loading, training and evaluation scripts, checkpoint provenance, and architecture and results documentation.
- Populated `dataset/.gitignore` to ignore the preprocessed mesh pool, generated geometry caches, optional raw simulations, preprocessing outputs, and downloaded archives. Preserved the raw manifest and cache directory placeholders.
- Updated `dataset/README.md` with dataset placement instructions and the command to stop tracking the mesh pool while retaining local files: `git rm -r --cached -- dataset/processed_augmented`.
- Explained that ignore rules do not untrack existing files or remove data from Git history. The assistant did not execute the untracking command.
- Rewrote the root `README.md` from scratch using the supplied paper, with its full title and authors, method description, main comparison, per-field results, selected ablations, installation, data setup, evaluation, training, and implementation limitations.
- Added original paper figures under `docs/paper/`: `compare_models.png`, `frequency_bands.png`, and `overview_landscape.pdf`. Rendered the architecture PDF as `overview_landscape.png` for Markdown display.
- Included an existing qualitative image from evaluation of the released checkpoint.
- Replaced both unsupported `\operatorname` macros with `\mathrm` after the user reported a rendering error.

### Sources and result distinctions

Paper source directory:

`/home/int/Dropbox/Workspace/01_3DMLCV_part_B/hodgeformer/experiments/regression_pdes_tetrahedral/paper/hodge_moe_pde_v8_NEURIPS_cleanup`

Use the manuscript's `main.tex` and `sections/` for the title, authors, experimental protocol, tables, and captions. Do not conflate these three runs:

- Paper headline vertex-only Hodge-routed MoE: 3.02M parameters, mean relative L2 0.0300, selected at epoch 380.
- Paper supplementary vertex, edge, and face routing experiment: 3.11M parameters, mean relative L2 0.0359, selected at epoch 560 after a continuation.
- Checkpoint packaged in this repository: `checkpoints/hodgemoe_vef_best_ep400.pth`, 3,106,490 parameters, fp32, epoch 400; mean relative L2 0.0329, R2 0.9981, Volume 0.00170, Surf 0.00202, mesh-native lift relative error 0.0313, lift Spearman correlation 0.9999.

The paper's 31% relative L2 and 47% interior MSE reductions against Transolver apply to the headline vertex-only model. Preserve the distinctions between paper figures and local checkpoint evaluation. Protocol differences, conditional spectral guarantees, and the wall-band lift convention are documented in the README and detailed release documentation.

### User preferences

- Use a professional paper-based title and explain terminology; avoid internal experiment labels such as v57 in the main presentation.
- Include meaningful paper figures and quantitative results in the README.
- Keep equations compatible with the user's Markdown renderer; it rejects `\operatorname`.
- Record progress in a persistent repository journal.

### Verification

- Checked that all local README links and image paths exist.
- Validated the three paper PNG files with Pillow and visually inspected the architecture and comparison images.
- Checked ignore behavior for sample dataset and cache paths, and confirmed documentation, the raw manifest, and cache placeholders remain eligible for tracking.
- `git diff --check` passed after README formatting and macro fixes.
- Training and model evaluation were not rerun for documentation-only changes. Final rendering in the user's application has not been independently verified.

### Pending

- Obtain the actual Dropbox shared download URL. No URL was present in the repository or supplied in the conversation; the READMEs explicitly mark it as pending. Replace that notice with the real link once supplied.
- If moving to download-only distribution, check whether the user has already untracked the mesh pool before taking further Git actions. No history rewrite was requested.
