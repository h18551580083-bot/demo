# Phase 0 gap register

Audit baseline: Git `693b7c3`, 2026-08-03. Final disposition is based on the current
working tree and the explicit autonomous closure authorization in the attached
task, not on deleted historical code.

| ID | Problem | Initial state | Normative source | Code/evidence source | Phase 0 blocker | Decision and files | Verification | Final result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0-01 | No public experiment pipeline | missing | Spec 3, 4.1 | `cg_pipeline.pipeline` | yes | One preflight/dry/train seam | entrypoint tests and dry report | resolved |
| P0-02 | No strict experiment config | missing | Spec 4.3 | `config.py`, both TOML contracts | yes | no defaults; every formal scientific value is exact; unknown/missing/TBD/float/mutation fails | `test_pipeline_config.py` | resolved |
| P0-03 | Existing-patch adapter absent | missing | ADR 0009, Spec 4.4 | `data.py` | yes | exact CSV/PNG/path/label contract | data tests; real preflight | resolved |
| P0-04 | Split isolation executable check absent | missing | Spec 4.5, Section 8 | `validate_manifest` | yes | exact declared `slide_id` claim only | negative cross-split test | resolved at slide-ID level |
| P0-05 | Reliable patient mapping absent | external missing input | Spec 5, Section 8 | `validate_patient_mapping` plus release approval binding | yes | do not infer; require complete mapping plus attributable provenance-reliability approval bound to mapping and source-manifest hashes | preflight patient gate and approval negative tests | **external blocker** |
| P0-06 | Morlet generator/frontend absent | missing | Spec 3.1, Decisions 1-18 | `morlet.py`, `frontend.py` | yes | explicit 32-kernel immutable bundle and FFT/spatial paths | frontend/spectral tests | resolved |
| P0-07 | H/E interaction/backend absent | missing | Spec 3.2-3.12 | `interaction.py`, `pooling.py`, `model.py` | yes | exact seven features and 9473-scalar backend | model tests | resolved |
| P0-08 | Fixed frontend not automatically proven | missing | Spec 4.6 | fixed-state and optimizer audits | yes | no optical Parameters; pre/post byte identity | training/dry tests | resolved |
| P0-09 | Loss/optimizer precision unresolved | TBD | Spec 6 | config and training protocol | yes | float32 BCE-with-logits; float32-state AdamW | config/training tests | frozen |
| P0-10 | Training seeds/budget/checkpoint/CI unresolved | TBD | Spec 6 | baseline TOML and protocols | yes | conservative three-seed 20-epoch preregistration; resume requires continuous paired artifacts and full identities | training/eval tests | frozen |
| P0-11 | `cam16-eval-v1` arithmetic incomplete | partial draft only | Spec 3.15, Decision principles | `evaluation.py`, evaluation protocol | yes | exact AUROC/Youden, max-logit slides, 2000 bootstrap, complete authorized-row ledger and identity-bound test gate | evaluation tests | frozen |
| P0-12 | Transfer group open | TBD but later scope | Spec 6, project boundary | ADR 0010 | no for CAM16 Phase 1 entry | no transfer dataset/adaptation in this baseline; later separate preregistration | source/config audit | not applicable now |
| P0-13 | No end-to-end dry run | missing | Task dry-run contract | `run_dry_run` | yes | synthetic-only two-repeat full chain plus negative path | dry report | resolved |
| P0-14 | No formal fail-closed entry | missing | Task completion gate | `preflight`, `train`, release JSON | yes | train always preflights and starts zero steps on failure | entrypoint tests | resolved, currently blocked correctly |
| P0-15 | Historical WSI requirements | withdrawn residual | ADR 0009, commit `3221656` | source/config audit | no | preserve as history; do not restore | tracked-source review | not applicable |

The current stop condition is Checkpoint B: all repository-internal work is
complete, but patient-level acceptance cannot be established from the supplied
slide identifiers. The minimum external input and post-input command are recorded
in the acceptance matrix and final report.
