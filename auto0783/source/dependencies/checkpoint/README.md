# Published checkpoint dependency

The model repository revision and all six published checkpoint shard checksums are immutable. The complete public checkpoint is about 29.7 GB. The numerical task uses only the manifest-declared 677 pretrained decoder tensors; it does not evaluate the entire multimodal or FLUX image pipeline.

Keep these public assets outside the ordinary Agent image and candidate workspace. A trusted operator can recover them with:

```sh
python dependencies/checkpoint/materialize_checkpoint_assets.py --manifest dependencies/checkpoint/public-checkpoint-assets.json --tensor-manifest dependencies/checkpoint/public-tensor-manifest.json --destination checkpoint-assets
```

If the verified original shards already exist, pass their directory as `--destination` and add `--cache-only`. This performs checksum verification without a download. `--plan-only` validates the descriptor but does not prove materialization or task execution.

The trusted tensor server binds only loopback. It serves the declared manifest and tensor ordinals, never arbitrary file paths, hidden inputs, Reference artifacts or credentials. Workers independently check each tensor's size and SHA256 before loading it into CPU memory. CUDA execution remains confined to the short-lived leased numerical child.

Start the tensor server only on a trusted worker host with the original verified shards available, after materialization. Do not start a second server or numerical parent if one already exists.

```sh
python dependencies/checkpoint/public_tensor_server.py --manifest dependencies/checkpoint/public-tensor-manifest.json --checkpoint-directory checkpoint-assets
```

The CUDA worker uses Python 3.12 with Torch 2.5.1+cu121, Transformers 4.49.0, Safetensors 0.7.0 and NumPy 1.26.4. These versions were read from both actual research worker environments. The CPU parent checks the manifest and tensor checksums, and the numerical child applies the existing GPU scheduling leases. This dependency entry does not replace the full trusted Harbor adapter, worker deployment configuration or new release validation evidence.
