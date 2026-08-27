# Historical scripts — not the publication entry point

These three scripts are preserved from the August 16 exploratory work:

- `evaluate_speed_gain.py`: hypothetical risk routes and interpolation to `(0, 0)`.
- `summarize_target_quality.py`: the legacy pixel-min oracle and optimistic low-ratio accounting.
- `make_report_assets.py`: figures for the original local geometry notes.

They originally ran at the geometry-baseline root next to its complete `results/` directory. They are not directly runnable in this archive folder and some require large local arrays omitted from this release. They are retained for provenance, **not as recommended code for new conclusions**.

Use root `summarize_results.py`, `results/audited/`, and `docs/AUDIT.md` for corrected publication language. Historical JSON under `results/recorded/speed_gain` and `results/recorded/target_quality_summary` is preserved without numeric edits; it inherits the limitations above. No real Matrix-Game end-to-end speedup was measured.
