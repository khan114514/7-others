# Delivery topology and execution entry

This fixed pretrained decoder inference task has no training/fine-tuning or newly trained model checkpoint; training model/ evidence is NOT_APPLICABLE.

The physical native Harbor task is ../Task. Its Agent image build context is ../Task/environment; the separate Verifier context is ../Task/tests. Both Docker COPY source sets are included and frozen by ../dependencies/task/task-files.json. The native adapter is ../Runtime/hybrid_harbor_environment.py; candidate containers see only their own tmpfs /workspace/solution, role queue and permitted log shadows. The Reference source under ../Task/solution and ../expert_evidence are excluded from Agent mounts and build contexts.

First restore the declared CLI and hidden assets into ../Task exactly as specified in ../MATERIALIZATION.md. Then use ../dependencies/task/materialize_task.py to make a fresh checksum-verified task copy, then ../Runtime/run_isolation.py and ../Runtime/run_native_validation.py with a trusted external runtime-state and numerical-status. This README is an explicit mapping, not a substitute for their dynamic proof. The four frozen-version full GPU/CPU runs and actual other-host evidence readback are closed; final complete ZIP and independent two-model QA remain pending. Historical evidence retains its original version and source SHA.
