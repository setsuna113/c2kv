"""Assemble an evidence-bound delivery snapshot without starting any model."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def build(out, include_suites=()):
    if out.exists(): raise ValueError("Use a fresh delivery snapshot directory")
    base=ROOT/"outputs/history_system_search/delivery_20260914"
    index=json.loads((base/"suite_execution.json").read_text(encoding="utf-8"))
    selection=json.loads((base/"algorithm.selected.json").read_text(encoding="utf-8")) if (base/"algorithm.selected.json").exists() and not include_suites else None
    native=ROOT/"outputs/history_system_search/r001/r002_d3_prefill_event_mixed20_v1"
    inputs=[("bfcl_mixed20",native,"native")]+[(r["suite_id"],ROOT/r["output"],"suite") for r in index["active_suites"] if r["suite_id"].startswith("d3_")]
    for name in include_suites:
        source=base/"suites"/name
        if source.parent != base/"suites" or name in {row[0] for row in inputs}:
            raise ValueError("Invalid or duplicate suite ID")
        inputs.append((name,source,"suite"))
    out.mkdir(parents=True); (out/"packages").mkdir(); entries=[]
    for name,source,kind in inputs:
        frozen=json.loads((source/"freeze.json").read_text(encoding="utf-8"))
        archive=source/"package.tar.gz"
        assert digest(archive)==frozen["archive_sha256"]
        target=out/"packages"/(name+".tar.gz");shutil.copyfile(archive,target)
        entries.append(dict(id=name,kind=kind,archive=target.relative_to(out).as_posix(),sha256=digest(target),origin=str(source.relative_to(ROOT)).replace("\\","/"),execution_role="scoring_recovery_prefix_only" if name=="d3_acebench_agent8_v5" else "frozen_evaluation",frozen=frozen))
        entry=entries[-1]
        entry["candidate_id"]=frozen.get("candidate_id")
        proof=source/"returned/recovery.validation.json"
        entry["execution_state"]="terminal_recovered" if proof.exists() else "launched_not_terminal_recovered" if (source/"launch.json").exists() else "frozen_not_launched"
    for src,name in [(base/("results.selected.json" if selection else "results.current.json"),"results.snapshot.json"),(ROOT/"reports/weekly/2026-09-14-history-system.md","report.md"),(ROOT/"reports/weekly/2026-09-14-history-system.sources.json","report.sources.json"),(ROOT/"experiments/history_system/configs/checkpoint.selected.json","checkpoint.json")]:shutil.copyfile(src,out/name)
    for name,target in [("progress.latest.json","progress.snapshot.json"),("suite_execution.json","execution.snapshot.json")]:
        if (base/name).exists():shutil.copyfile(base/name,out/target)
    shutil.copyfile(ROOT/"experiments/history_system/configs/delivery_20260914.design.json",out/"delivery.contract.json")
    metrics=ROOT/"reports/weekly/2026-09-14-history-system.metrics.json"
    if metrics.exists():shutil.copyfile(metrics,out/"report.metrics.json")
    shutil.copyfile(Path(__file__).with_name("delivery_launch.py"),out/"launch.py")
    (out/"algorithm.json").write_text(json.dumps(dict(schema="history-delivery-snapshot-v1",status="selected_delivery_algorithm" if selection else "candidate_snapshot_not_final_selection",name=selection["name"] if selection else "C2KV hybrid history candidate collection",checkpoint="checkpoint.json",packages=entries,external_dependencies="Selected C1000 weights, NPU Python/overlay and official benchmark sources remain external, as bound in each frozen package.",automatic_model_calls=0),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (out/"README.md").write_text("# C2KV hybrid history candidates\n\nEach package in algorithm.json has its own candidate identity and execution state. Newer frozen code does not inherit the scores of an earlier candidate.\n\nThis candidate snapshot contains the exact submitted runtime, embedded Prefill head, budget configuration and task manifests in each package archive. It is not a declaration that final system selection is complete. See report.md for completed results and missing benchmarks.\n\nRun `python launch.py verify` to verify the snapshot, and `python launch.py list` to list packages. Run `python launch.py prepare --package PACKAGE_ID --out NEW_DIRECTORY` to extract a verified package and print its existing runner entry point. Preparing does not load a model or repeat any evaluation. The ACE prefix package is retained only for provenance; its first two tasks were scored without replay.\n\nModel weights, Ascend dependencies and official benchmark datasets are external. Their original locations and settings remain in the frozen files. Allocate only a verified idle owned device before using the printed runner; a copied package does not authorize a duplicate run.\n",encoding="utf-8")
    if selection:
        shutil.copyfile(base/"algorithm.selected.json",out/"selection.json")
        (out/"README.md").write_text("# Prefill-guided event recovery\n\nSelected algorithm for the 2026-09-14 delivery: C1000 / ratio8, with the frozen Prefill classifier and event recovery controller. Only this algorithm is packaged. Further research continues separately.\n\nSee report.md for external baseline comparisons and missing benchmark outcomes. The selection is fixed; the five-benchmark evaluation is not yet complete.\n\nUse `python launch.py verify`, `python launch.py list`, or `python launch.py prepare --package PACKAGE_ID --out NEW_DIRECTORY`. Preparation does not run a model. Weights, Ascend dependencies and official datasets remain external as bound in the packages. The ACE prefix package is provenance only.\n",encoding="utf-8")
    figures=ROOT/"reports/weekly/figures/history_20260914"
    if selection and figures.exists():
        shutil.copytree(figures,out/"figures/history_20260914")
        shutil.copyfile(Path(__file__).with_name("plot_selected_baseline_cost.py"),out/"figures/history_20260914/plot_selected_baseline_cost.py")
    files={p.relative_to(out).as_posix():digest(p) for p in out.rglob("*") if p.is_file()}
    (out/"release.manifest.json").write_text(json.dumps(dict(files=files),indent=2)+"\n",encoding="utf-8")
    return dict(output=str(out),packages=len(entries),files=len(files),model_calls=0)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--include-suite",action="append",default=[]);args=ap.parse_args();print(json.dumps(build(args.out.resolve(),args.include_suite)))
