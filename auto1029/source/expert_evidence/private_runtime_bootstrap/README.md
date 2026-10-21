# Private runtime directory deployment supplement

Use this supplement with the frozen generic backend whose asset manifest SHA256 is
`5b5b7e8b78f47e302cb9c9dc4dbc2533ec69126dd272c494d2d094f886000953`.
The installer SHA256 is `793f7c8298218fe63ef580262f43a254d56207eb9b65189d249da381de33bad0`.
This supplement is an operator tool outside the frozen scoring bundle. It changes
no decoder, method, protocol, asset, source manifest or in-flight experiment.

The installer does not create `private/hidden-validation-jobs`. A real first
Native attempt failed before GPU admission because that parent was absent.
The failed attempt remains preserved in the sibling `private_route_preparation`
evidence: `first-full-native-failure.json`, `first-remote-worker-failure.json`
and `runtime-directory-contract.json`. That failure has no score.

1. On a dedicated trusted Linux x86_64 host, provision the frozen Python3.12,
   Torch2.5.1+cu121, libseccomp, Torchtune package and all manifest-listed assets.
   Use a new operator-owned deployment directory and a root-owned mode0600
   deployment config with the required fields from the backend README. Include
   all shared GPU mutexes. Root-only `private/` must be exactly mode0700.
2. Run the frozen installer with the exact receipt name below. Preserve any
   failed receipt and existing nonidentical bytes; do not overwrite them to
   claim successful installation. A code-only receipt does not admit scoring.
3. Build the jail and run the actual isolation audit. The expert/platform
   injects the exact frozen1380 private samples and image files into root-only
   `private/hidden-injected-v1/`. No private corpus goes in the public artifact
   or either model's Actor. A missing or different corpus is an error.
4. Run this supplement as trusted root before the first private worker. It
   verifies the pinned guards, complete installed source snapshot, real full
   asset receipt, exact private corpus, current jail receipt and free space.
   It creates only the missing runtime parent, mode0700. Existing insecure
   directories or symlinks are refused without chmod, deletion or repair.
5. Configure the trusted RootVerifier adapter to call the private endpoint;
   the Agent endpoint remains Dev-only. Perform a real separate Harbor build
   and root-route/isolation probe on the target machine. Before starting a new
   score, inspect both physical roles' active tickets and the shared lease.
   Reattach to an unfinished ticket; do not launch a duplicate. Retain failed
   attempts. The bootstrap tool itself neither acquires a GPU nor submits work.
6. Run the unchanged private full protocol with seeds17,29,43,71,101 and1380
   samples per seed. Candidate code executes only in the verified jail.
   Collect terminal raw receipts, all quality checks, source binding and the
   actual Native reward. Directory readiness is not a Native PASS.

Run these commands from the separately verified public backend bundle, as root.
`HEIMA_DEPLOYMENT_CONFIG` is a literal absolute config path; `HEIMA_ROOT` and
`JAIL_PYTHON` must equal its configured paths. Placeholder strings are not
expanded by the config loader. The supplement path is separate from that bundle.

```sh
python3 -S install_backend.py --config "$HEIMA_DEPLOYMENT_CONFIG" --receipt-name portable-full-install-receipt.json
HEIMA_DEPLOYMENT_CONFIG="$HEIMA_DEPLOYMENT_CONFIG" "$JAIL_PYTHON" -S "$HEIMA_ROOT/private/build_policy_jail.py"
HEIMA_DEPLOYMENT_CONFIG="$HEIMA_DEPLOYMENT_CONFIG" "$JAIL_PYTHON" -S "$HEIMA_ROOT/private/audit_isolation.py"
# Expert injects the exact private corpus before this next command.
"$JAIL_PYTHON" -S "$SUPPLEMENT_DIR/bootstrap_private_runtime.py" --config "$HEIMA_DEPLOYMENT_CONFIG"
"$JAIL_PYTHON" -S "$SUPPLEMENT_DIR/bootstrap_private_runtime.py" --config "$HEIMA_DEPLOYMENT_CONFIG" --check-only
```

Store each command's exact argv, start/end timestamps, exit status and original
stdout/stderr in the operator's private audit. No provider credentials are
needed. Do not run installation or source changes against an active backend.
Keep at least2GiB free on the actual deployment volume; an expanded data disk
being full is not evidence of available scratch space. Never delete peer data.

This new supplement has CPU filesystem guard evidence only. It has not been
executed against either current live GPU backend; their validated bootstrap
and experiment identities stay unchanged. The running full validation and
current source's12h stability are separate pending gates. Historical scores
and previous source versions'12h receipts do not close those gates.
