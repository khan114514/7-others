# Trusted GPU backend

This directory is outside both Agent and Verifier image build contexts. Its
code, Reference inputs, target weights, deployment configuration and receipts
belong to the trusted GPU operator. Candidates reach it only through the fixed
lane-owned submit/status API. Candidate modules execute inside the separate
stdlib-only jail after dropping privileges and applying seccomp.

`run_policy_score_v2.py`, `policy_decoder_v1.py`, `mcq_metrics.py` and
`policy_jail_runner.py` are byte-identical to the measured canonical sources.
`source_manifest.json` identifies the configured copies. The trusted Encoder's
local Tensor variable was renamed from `token` to `sampled_token`; its normalized
AST is alpha-equivalent to the measured implementation. The unused direct
method-import driver was removed from the deployment copy.
`asset_manifest.json` freezes code, all96 Dev rows and
images, the five-seed protocol, calibrated anchors and external asset hashes.
It includes no Hidden rows or images. Large weights are provisioned separately;
the manifest is not a grant of redistribution rights.

Use a trusted Linux x86_64 GPU host with Python3.12, libseccomp, the pinned Heima
Torchtune fork and the verified Torch2.5.1+cu121 runtime. Create a root-owned0600
JSON configuration outside candidate visibility. Every field is required:

```json
{
  "project_root": "${HEIMA_ROOT}",
  "scoring_python": "${SCORING_PYTHON}",
  "jail_python": "${JAIL_PYTHON}",
  "stdlib_root": "${PYTHON_STDLIB}",
  "torchtune_package": "${TORCHTUNE_PACKAGE}",
  "gpu_lock_files": ["${SHARED_GPU_MUTEX}"],
  "nvidia_smi": "${NVIDIA_SMI}"
}
```

Replace these placeholders with absolute deployment paths; the loader does not
expand shell expressions. Include every shared GPU mutex used by other active
tasks. The idle check also discovers `/tmp/gpu*.lock` and never truncates locks.
Do not put model provider keys or URLs in this configuration. The GPU backend
does not call GPT or Seed.

Install target assets at the relative paths in `external_assets`, and provide
the exact Torchtune package files in its manifest. Trusted immutable weight
links may reuse an existing asset pool without another21GB copy. The installer
verifies the resolved root-owned files by size and full SHA256; it never changes
the linked weights or imports candidate code.

For a new asset pool, `tools/convert_bf16.py --source "$FP32_SHARDS" --target
"$BF16_SHARDS"` reproduces and checks all nine runtime shards from pinned
original SHA256 values. Its conversion functions are identical to the actual
measured conversion. It preserves originals and refuses different existing or
partial outputs. Provision the separately indexed metadata and restored
tokenizer as well; the conversion tool handles shards only. Source and runtime
weights together require approximately61GB before any scratch space.

```sh
python3 install_backend.py --config "$HEIMA_DEPLOYMENT_CONFIG"
HEIMA_DEPLOYMENT_CONFIG="$HEIMA_DEPLOYMENT_CONFIG" "$JAIL_PYTHON" "$HEIMA_ROOT/private/build_policy_jail.py"
HEIMA_DEPLOYMENT_CONFIG="$HEIMA_DEPLOYMENT_CONFIG" "$JAIL_PYTHON" "$HEIMA_ROOT/private/audit_isolation.py"
```

Installation refuses different existing bytes. Every receipt filename is fresh;
use `--receipt-name` for a later check. `--code-only` installs116 lightweight
files but explicitly leaves assets unverified and scoring NOT READY. Run large
asset verification outside active GPU experiments or while holding the shared
lease. Building the jail requires trusted root privileges. No site packages,
credentials, target weights or host filesystem are copied into a policy jail.

The preceding configured source version passed the real CPU jail audit, code installation,
full SHA256 verification of all25 external assets and197 Torchtune files, and
an actual two-question/seed17 GPU smoke with ordinary AR executed only in the
policy jail. Quality passed; `complete_primary_protocol` is false. Original
failure and success receipts are retained; no large weights were copied. This
staged source requires its own fresh immutable installation and execution
receipts. The earlier proof does not verify its changed Encoder bytes.

The sibling `host_adapter/` also passed a real Native build/transport check
and separate actual Codex/tool calls to both selected model routes. This still
does not establish standalone full scored Native portability or new-machine
reproduction: base-binary delivery, configured complete Trial rewards and the
final adopted source's12h stability need their own evidence. A quick smoke
never replaces the full five-seed score, clean best-method rerun or full private
Hidden validation.


## Private RootVerifier endpoint

The Agent-facing `gpu_evaluator_service.py` remains Dev-only. Configure the
trusted host adapter's `verifier_gpu_service_argv` to invoke the separately
installed `private/private_verifier_service.py`; no provider credential is
passed to either scoring endpoint. Root-only Verifier transport selects the
physical role, and candidates cannot supply a split or alternate seed set.

Install this bundle with `install_backend.py --config "$HEIMA_DEPLOYMENT_CONFIG" --receipt-name portable-full-install-receipt.json`. A `--code-only` receipt is
not accepted for private GPU scoring. The installer freezes the exact installed
code/data hash list in `private/portable-private-route-source-manifest.json`.
Root-owned weights and Torchtune files must pass the full unchanged asset
manifest, and the isolated policy jail must pass the declared boundary probes.

The platform/expert then injects its exact private `samples.json` and `images/`
into the configured deployment's `private/hidden-injected-v1/` (mode0700,
root-owned regular files). They must match the packaged private protocol's
sample/image fingerprints. No private samples, answers, or images are present
in this public bundle. The source/injection guards do not depend on an author's
home directory, canonical checkout name, or a prior transfer-plan layout.

The generic source/injection wrapper is staged deployment preparation. Its
fresh complete private GPU/Native execution proof must be collected before it
is labeled validated. Existing runtime proofs bind their own exact source
version; they are not silently transferred to this version.
