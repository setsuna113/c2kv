"""CPU-only verification of every frozen calibration prefix; no model requests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from generality.calibrate import LABELS, load_rows
from generality.prefix_replay import bind_source_trace, restore_bfcl_prefix
from t02_bfcl import OfficialBFCLBindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', default=LABELS)
    parser.add_argument('--source-root')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source_root = args.source_root or Path(args.labels).parent.parent
    results = []
    for row in load_rows(args.labels):
        item = {key: row[key] for key in ('state_id', 'task_id', 'decision_key')}
        replay = None
        try:
            bound = bind_source_trace(row, source_root)
            replay = restore_bfcl_prefix(bound, OfficialBFCLBindings())
            item.update(status='ok', actual=replay.current_payload()['decision_key'],
                        previous_turn_valid=replay.previous_turn_valid,
                        source_trace=bound['_source_trace'],
                        model_messages=len(replay.current_payload()['messages']))
        except Exception as error:
            item.update(status='failed', error=f'{type(error).__name__}: {error}')
        finally:
            if replay is not None:
                replay.env.close()
        results.append(item)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    failed = sum(item['status'] != 'ok' for item in results)
    print(json.dumps({'states': len(results), 'reconstructed': len(results) - failed,
                      'previous_turn_valid': sum(item.get('previous_turn_valid') is True for item in results),
                      'model_requests': 0, 'output': str(out)}))
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
