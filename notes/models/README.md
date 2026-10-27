# notes/models

The reference record of every `docs-bridge-models` GPU tag: its
`/opt/docs-bridge-models.json`, copied out of the image when it was built. The image
lives only in the EC2 host's Docker store, so this is what a rebuild is checked
against — same sha256 per file means the same model, and every collection embedded
with it stays valid (`tools/onnx-export/FINDINGS.md` finding 5).

One file per tag, `<tag>.json`, never edited after it is committed. Each holds the
tag, the docs-bridge commit and exporter image that built it, the HF revision, the
sha256 and size of every shipped file, and both `check_precision.py` results.

Copy one out of an image:

```bash
c=$(docker create localhost/docs-bridge-models:<tag> x)
docker cp $c:/opt/docs-bridge-models.json - | tar -xO > notes/models/<tag>.json
docker rm $c
```
