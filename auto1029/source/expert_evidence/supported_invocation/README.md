# Supported complete-private Trial invocation

The advertised complete-private invocation from the three-directory root is:

```sh
python3.12 workspace/operator/run_recover_private_native.py --policy baseline --role gpt
```

Provision the root-only host configuration, independent queues and external frozen
assets as specified in workspace/operator/README.md before running. This selects
NOP and the packaged NoWallPrivateTrial implementation. The ordinary task.toml
Verifier default is 10800 seconds. This supported operator explicitly overrides
that outer total timeout to None; it preserves the complete five-seed, 1380-question
private protocol, all inference and scoring budgets, and the 0.5-second policy
IPC cap. Do not use the resulting record as proof of an ordinary 10800-second Trial.

The supplied current-nop originals show a real separate Verifier, no exception,
6900 completed pairs and a finite matching reward. Its Verifier lasted
66585.788957 seconds (about 18.5 hours), so it cannot support the ordinary timeout.
Read effective-invocation-binding.json for actual source hashes and configuration.

This release makes only a documentation amendment: both synchronized instruction
copies and the operator README explicitly distinguish ordinary timeout defaults
and the complete-private override; the instruction also distinguishes fixed Dev
anchors from the already declared private paired ordinary-control normalization. documentation-amendment.diff contains every changed
line. All executable runtime/configuration/scientific bytes and recorded NOP files
are unchanged. The original task-source-manifest.json and task-name-binding.json
remain original observations of the recorded instruction version; they are not
rewritten as observations of this amended documentation. The binding separately
lists unchanged runtime members and changed document hashes. No NOP rerun with the
new document bytes is claimed. Existing NOP execution supports the same executable
runtime and supported operator; documentation equivalence requires semantic review.

The full private NOP is a no-op control. Formal Reference improvement is supported
by the separate frozen Dev evidence. No current private Reference improvement,
platform acceptance or independent review verdict follows from this document.
