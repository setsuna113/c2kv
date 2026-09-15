# r001 peer completion wrapper

This wrapper completes the ten missing Base cells for Raw, Text, Full, and
HiAgent. Each config derives a Base10 design from one frozen B500/greedy Long20
design and one hash-bound audited manifest template. The template source delta
is limited to replacing the R5 `independent_holdout` acceptance gate with an
explicit `development_search` gate. It keeps the original command builders,
task isolation, limits, retries, terminal denominator, official scorer checks,
and telemetry.

`freeze` and `preview` are CPU-only. Only the explicit `run` action launches a
model or scorer. Root owns packaging, live device/port checks, and invocation of
`run`.

## Binding layout

The wrapper freeze is not a self-contained remote runtime. Relative paths in a
peer config resolve under `--repo-root` (the current repository root by
default), and their hashes are checked before every preview, freeze, or run.
The remote staging tree must preserve these submitted source roots:

- Raw: `outputs/a_memory_runtime_20260912/native_raw_b500_long20_v1/submitted`
- Text: `outputs/a_memory_runtime_20260913/native_text_b500_long20_metafix_v1/submitted`
- Full: `outputs/a_memory_runtime_20260912/native_full_b500_long20_v1/submitted`
- HiAgent: `outputs/a_memory_runtime_20260913/native_hiagent_b500_long20_interpreterfix_v1/submitted`

It must also preserve `experiments/history_system/peers`, both r001 task
manifests, `outputs/history_system_search/r001/remote_lineage.json`, and the two
audited template directories under `tmp/a_memory_runtime_20260913`. The
checkpoint, BFCL checkout, policy-sampling file, interpreters, device, and ports
remain explicit run-time bindings. The HiAgent `--source-root` may point at its
materialized source snapshot; baseline roots are derived from the bound source
runner location and therefore require the submitted layout above.

## CPU preview and freeze

```bash
python experiments/history_system/peers/runner.py preview \
  --repo-root <staged-repo-root> \
  --config experiments/history_system/peers/configs/raw.base10.json

python experiments/history_system/peers/runner.py freeze \
  --repo-root <staged-repo-root> \
  --config experiments/history_system/peers/configs/raw.base10.json \
  --output-dir <new-raw-base10-freeze-dir>
```

Repeat with `text.base10.json`, `full.base10.json`, and
`hiagent.base10.json`. `freeze` refuses an existing output directory, writes a
ten-task `design.json`, an r001 `index.design.json`, a CPU `preview.json`, and
hash receipts. A failed freeze keeps the partial directory and writes
`freeze.failed.json` when possible.

## Explicit execution

Raw, Text, and Full use the baseline interface:

```bash
python experiments/history_system/peers/runner.py run \
  --repo-root <staged-repo-root> \
  --frozen-dir <peer-base10-freeze-dir> \
  --checkpoint <bound-B500-checkpoint> \
  --output <new-base10-result-root> \
  --benchmark-dir <BFCL-root> \
  --python <server-python> \
  --bfcl-python <BFCL-python> \
  --port-base <checked-free-port-base>
```

HiAgent uses its existing separate actor/auxiliary interface:

```bash
python experiments/history_system/peers/runner.py run \
  --repo-root <staged-repo-root> \
  --frozen-dir <hiagent-base10-freeze-dir> \
  --source-root <materialized-hiagent-source-root> \
  --checkpoint <bound-B500-checkpoint> \
  --policy-sampling <bound-greedy0-json> \
  --output <new-base10-result-root> \
  --benchmark-dir <BFCL-root> \
  --server-python <server-python> \
  --bfcl-python <BFCL-python> \
  --proxy-python <proxy-python> \
  --device <checked-device> \
  --server-port-base <checked-free-server-port-base> \
  --proxy-port-base <checked-free-proxy-port-base>
```

After Base10 returns, combine its root with the method's reused Long10 root or
roots using the exact `collect_results.py` command recorded in
`configs/peer_sources.json`. The collector retains the full r001 denominator;
until all Base cells exist they remain `unknown` rather than algorithm failures.

The authorized queue is now integrated into `../advance.py --dispatch-next`.
Native C1/C3/C4 entries have priority; remaining lanes dispatch HiAgent, Raw,
Text, then Full. `dispatch.py` exposes `preview(method, device)`,
`launch(method, device)`, and `observe(method)` with fresh ownership and port
checks on physical devices 4/7. An occupied lane remains pending with no model
launch. Terminal stages are hash-recovered by `../recover.py --peer-method`,
billed once, and collected with all compatible Long10 roots automatically.

Quality comparisons are descriptive whole-system comparisons. Matching the
ordered manifest is only the collector gate; the B500 checkpoint, greedy
sampling, and official scorer identity must also be verified from the recorded
source bindings. This is not a strict causal comparison.

HiAgent has no B0 contract and no compatible common/full-input H/A field, so its
system-active history reduction remains unknown. Its existing
`hiagent_steps.jsonl`, actor and auxiliary attempt journals, supervisor usage
totals, and per-generation allocator peaks remain available and must be reported
instead of discarding all resource evidence.
