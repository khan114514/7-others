# Build-context repair candidate

Scope: published pretrained 28-layer decoder only.

Both Docker build contexts now contain every COPY source. The Agent owns the three checksum-bound public CLI executables in a compressed vendor-only archive, restored inside its image from included chunks with per-chunk/archive/executable checks. The Verifier owns the unchanged hidden chunks and combines/checks them inside its separate image. No network or host paths are required for either preparation step.

The build-ready Task has 48 files and manifest a1bf9e27a4adae5e2f224c80d3426960caeb7bf101da6e1fd151fefc43e7534c. The earlier Task f486 and its measured results are historical; the current selected-method source binding has fresh isolation and complete NOP/Oracle/GPT/Seed full GPU and independent CPU evidence. Do not label earlier measurements as this source revision.

To copy the exact build-ready tree, use the declared data-only helper:

    python dependencies/task/materialize_task.py --package . --destination /tmp/bifrost-frozen-task --evidence /tmp/bifrost-copy-evidence

Then the trusted Harbor adapter builds Agent from Task/environment and Verifier from Task/tests. The trusted numerical dependency/checkpoint setup is unchanged and separately checksum-bound.

All current scores and Trial identities are under expert_evidence/release-full-validation. The same tested Task48 and protected Runtime/method/dependency bytes are retained. Full-ZIP privacy/readback and two fresh same-ZIP independent QA reviews remain required; no formal QA PASS is claimed.

The exact prior document is retained under expert_evidence/historical-post-full-consensus-documents/MATERIALIZATION.md. Read current-evidence-clarifications.json for the literal current inventory and historical documentation distinctions.

All current scores and Trial identities are under expert_evidence/release-full-validation. The same tested Task48 and protected Runtime/method/dependency bytes are retained. Full-ZIP privacy/readback and two fresh same-ZIP independent QA reviews remain required; no formal QA PASS is claimed.
