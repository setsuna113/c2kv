"""Size redundant gist on recorded trajectories and compare the completed D7 run."""
import hashlib
import json
from pathlib import Path

from raw_gist_overlap import measure
from extended_metrics import summarize

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs/history_system_search/delivery_20260914'
MIB = 1024 ** 2


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_hashes(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and value['path'].endswith('steps.jsonl') and 'sha256' in value:
            yield value['sha256']
        for item in value.values():
            yield from artifact_hashes(item)
    elif isinstance(value, list):
        for item in value:
            yield from artifact_hashes(item)


def fraction(n, d):
    return 100 * n / d if d else None


def main():
    overlap_path = BASE / 'raw_gist_overlap.json'
    output = {
        'schema': 'dedup-investment-audit-v1',
        'sample_label': 'preliminary, n=1',
        'model_calls': 0,
        'counterfactual_scope': 'Subtract only fully raw-covered gist KV from recorded inputs; hold raw, other gist, trajectory and decode lengths fixed. No reallocation or runtime speedup simulated. Logical KV excludes model weights, encoder scratch, and allocator effects.',
        'cells': [], 'sources': [{'path': str(overlap_path), 'sha256': digest(overlap_path)}],
    }
    d3_ts = None
    for cell in load(overlap_path)['cells']:
        if not cell['records']:
            continue
        rows = []
        for source in cell['sources']:
            path = Path(source['path'])
            assert digest(path) == source['sha256'], str(path)
            rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line)
        committed = [r for r in rows if r.get('status') == 'ok' and r.get('generation_trace')]
        assert len(committed) == cell['final_views']
        final = []
        attempts = []
        unmeasured_attempts = 0
        for row in rows:
            for trace in row.get('generation_trace', []):
                if not isinstance(trace.get('generation'), dict) or not isinstance(trace['generation'].get('stats'), dict):
                    unmeasured_attempts += 1
                    continue
                overlap = measure(trace, row['ratio'])
                mem = trace['prepared_input']
                raw = set(mem['raw_source_indices'])
                event_sources = {}
                for chunk in mem['chunks']:
                    event_sources.setdefault(chunk['event_id'], set()).update(chunk['source_indices'])
                fully_raw = {event for event, sources in event_sources.items() if sources and sources <= raw}
                assert set(overlap['event_ids']) == fully_raw, 'Chunk and whole-event dedup differ'
                c = trace['controller']
                st = trace['generation']['stats']
                a = c['actual_history_bytes']
                common = c['same_prefix_full_reference']['common_live_bytes']
                drop = overlap['raw_covered_gist_bytes']
                assert 0 <= drop <= a
                assert a + common == st['resident_kv_logical_bytes_after_raw_prefill']
                entry = dict(history=a, context=a + common, resident=st['resident_kv_logical_bytes_final'], drop=drop)
                attempts.append(entry)
                if row.get('status') == 'ok' and trace is row['generation_trace'][-1]:
                    final.append(entry)
        assert sum(r['drop'] for r in final) == cell['raw_covered_gist_bytes_sum']
        dropped = sum(r['drop'] for r in final)
        peak = max(r['resident'] for r in attempts)
        new_peak = max(r['resident'] - r['drop'] for r in attempts)
        history_peak = max(r['history'] for r in final)
        new_history_peak = max(r['history'] - r['drop'] for r in final)
        result = dict(benchmark=cell['benchmark'], method=cell['method'], committed_views=len(final), attempts=len(attempts), unmeasured_attempts=unmeasured_attempts,
            mean_removed_gist_mib=dropped / len(final) / MIB,
            cumulative_history_saving_pct=fraction(dropped, sum(r['history'] for r in final)),
            cumulative_prompt_context_saving_pct=fraction(dropped, sum(r['context'] for r in final)),
            cumulative_resident_saving_pct=fraction(sum(r['drop'] for r in attempts), sum(r['resident'] for r in attempts)),
            resident_peak_mib=peak / MIB, counterfactual_resident_peak_mib=new_peak / MIB,
            resident_peak_saving_mib=(peak-new_peak) / MIB, resident_peak_saving_pct=fraction(peak-new_peak, peak),
            active_history_peak_mib=history_peak / MIB, counterfactual_active_history_peak_mib=new_history_peak / MIB,
            sources=cell['sources'])
        output['cells'].append(result)
        if cell['benchmark'] == 'toolsandbox':
            d3_ts = summarize(rows)

    comparisons = []
    for suite, name in [('d3_toolsandbox_lex8_v2', 'D3'), ('d7_toolsandbox_raw_dominant_v1', 'D7')]:
        returned = BASE / 'suites' / suite / 'returned'
        stage_path = returned / 'stage.json'
        stage = load(stage_path)
        hashes = set(artifact_hashes(stage))
        rows = []
        paths = sorted(returned.glob('task_shards/*/server/steps.jsonl'))
        assert len(paths) == stage['fixed_denominator']
        for path in paths:
            assert digest(path) in hashes, f'Unbound steps: {path}'
            rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line)
        m = summarize(rows)
        scores = [task['official_score'] for task in stage['task_outcomes']]
        assert len(scores) == stage['fixed_denominator'] and all(isinstance(s, (int, float)) for s in scores)
        comparisons.append(dict(method=name, official_mean=sum(scores)/len(scores), n=len(scores),
            steps=m['committed_steps'], generations=m['trace_generation_attempts'], regenerated=m['regenerated_steps'],
            extracted_chunks=m['extracted_chunks'], inference_seconds=m['inference_cumulative_seconds'],
            stage_wall_seconds=stage['wall_seconds'] if stage['wall_seconds_final'] else None,
            peak_history_mib=m['peak_active_history_kv_bytes']/MIB,
            peak_resident_mib=m['peak_resident_total_kv_bytes']['value']/MIB,
            peak_allocator_mib=m['peak_device_allocated_bytes']['value']/MIB,
            mean_history_mib=m['aggregate_history_kv_compression']['active_bytes_sum']/m['committed_steps']/MIB,
            history_compression=m['aggregate_history_kv_compression']['ratio'],
            source_coverage=m['source_occurrence_coverage'],
            source={'path': str(stage_path), 'sha256': digest(stage_path)}))
    output['toolsandbox_observed'] = comparisons
    output['comparison_scope'] = 'Same fixed tasks/checkpoint/B0; D7 combines dedup and demand encoding. Trajectories differ; elapsed time and quality are descriptive system outcomes, not isolated dedup effects.'
    before, after = comparisons
    output['observed_changes'] = {
        'official_delta_percentage_points': 100 * (after['official_mean'] - before['official_mean']),
        'generation_increase_pct': 100 * (after['generations'] / before['generations'] - 1),
        'inference_increase_pct': 100 * (after['inference_seconds']['value'] / before['inference_seconds']['value'] - 1),
        'stage_wall_increase_pct': 100 * (after['stage_wall_seconds'] / before['stage_wall_seconds'] - 1),
        'allocator_decrease_mib': before['peak_allocator_mib'] - after['peak_allocator_mib'],
        'allocator_decrease_pct': 100 * (1 - after['peak_allocator_mib'] / before['peak_allocator_mib']),
        'resident_decrease_mib': before['peak_resident_mib'] - after['peak_resident_mib'],
    }
    output['decision'] = {
        'status': 'deprioritize_standalone_d7_keep_unselected_option',
        'reason': 'Measured redundant-gist removal has limited whole-context/peak KV value on existing traces. The only completed D7 combination reduced quality and increased total generation/inference cost, despite a lower allocator peak. This is not evidence of seed stability or isolated dedup causality.',
        'next': 'No new standalone D7 sweeps or promotion. Retain the implementation for compatibility checks when history units/allocation change; prioritize useful fine-grained recovery and intervention value.',
    }
    output['sources'].append({'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).resolve())})
    path = BASE / 'dedup_investment_audit.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({**output, 'cells': [{k:v for k,v in c.items() if k != 'sources'} for c in output['cells']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
