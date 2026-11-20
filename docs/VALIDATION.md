# Publication validation

Validated locally on 2026-11-20 against the exported source tree, using the existing CaP-X Python 3.10 environment for integration dependencies.

- 172 tests passed across the recovery compiler, three-step harness, state window, execution guard, known-policy A/B, structural checkpoints, offline recovery, region placement and bin placement suites.
- Two existing PyRoKi deprecation warnings were reported.
- The standard-library evidence audit passed: hybrid 28/30, native 12/30; both-recorded injection counts 6/7/0; L4 physical terminal success and 22,034 tokens across both requests.
- All Python source files parsed successfully.
- New publication Markdown and media links resolved locally.
- Exported text files passed the credential-pattern scan.
- The MP4 metadata and representative frames were checked. GIF previews derive from archived videos; no model or simulator experiments were rerun for publication.

Clean-machine full dependency installation and new model/simulation runs were outside this publication check. Source provenance is retained in SOURCE_SNAPSHOT.json; optional upstream dependencies are pinned Git submodules.

The public response export omits repeated embedded input-image bodies, retaining hashes, sizes, prompts and model response text. Original source archives remain unchanged.
