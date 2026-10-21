# Operator base image provisioning

Supply separately authorized Linux ARM64 vendor executables named `codex`,
`codex-code-mode-host`, and `rg`. `binary_manifest.json` fixes the exact tested
sizes and SHA256 digests. No binaries, credentials, configuration or vendor
history are redistributed here. Provisioning does not establish redistribution
rights or substitute a different CLI version.

```sh
python provision.py --source-dir "$AUTHORIZED_VENDOR_BIN_DIR" \
  --output-dir "$FRESH_PRIVATE_BUILD_CONTEXT"
docker build --platform linux/arm64 --tag heima-agent-base:20261005 \
  "$FRESH_PRIVATE_BUILD_CONTEXT"
```

The source directory must contain operator-owned regular files. The output
must be a fresh directory in a trusted operator location. The tool verifies
ELF architecture, size and full digest, copies only those three binaries and
the frozen Dockerfile, verifies copied bytes again, and records a nonsecret
receipt. Source paths stay in the operator invocation and are not printed.
An interrupted or failed context is marked incomplete and must not be used.

The exact provisioning path has passed a genuine separate Docker build and
container run on the tested ARM64 controller, including all three in-container
binary hashes, Codex0.160.0 and an actual Codex sandbox write. The running
research base tag was left unchanged; the proof used a separate tag. This
proves this controller's configured build path. Full new-machine Native reward
reproduction and final adopted source stability are separate requirements.

GPU backend assets use their own manifest and installer. This base-image step
does not copy model weights or run target inference.
