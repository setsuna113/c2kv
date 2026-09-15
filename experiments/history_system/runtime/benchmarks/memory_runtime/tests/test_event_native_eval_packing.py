import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.memory_runtime.event_native_eval_packing import resolve_eval_packing
from benchmarks.memory_runtime.event_native_server import parser, _child_command


class EvaluationCapacityTests(unittest.TestCase):
    def test_capacity_is_explicit_and_preserves_training_geometry(self):
        profile = {'packing_contract': {'max_workspace_tokens':4096, 'max_sequence_tokens':16384,
            'max_target_tokens':4096, 'max_chunk_tokens':768, 'chunk_overlap':64, 'max_chunks':48},
            'model_geometry': {'max_position_embeddings':262144}}
        original = copy.deepcopy(profile)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'capacity.json'
            def write(capacity):
                path.write_text(json.dumps({'schema':'a-event-native-eval-capacity-v1','capacity':capacity}))
            write({'max_workspace_tokens':36864, 'max_sequence_tokens':40960})
            receipt = resolve_eval_packing(profile, path)
            self.assertEqual(profile, original)
            self.assertEqual(receipt['training_packing'], original['packing_contract'])
            self.assertEqual(receipt['effective_packing']['max_sequence_tokens'],40960)
            for field in ('max_chunk_tokens','chunk_overlap','max_chunks','max_target_tokens'):
                self.assertEqual(receipt['effective_packing'][field],profile['packing_contract'][field])
            write({'max_workspace_tokens':36864, 'max_sequence_tokens':40960, 'max_chunk_tokens':512})
            with self.assertRaises(ValueError): resolve_eval_packing(profile,path)
            write({'max_workspace_tokens':36864, 'max_sequence_tokens':16384})
            with self.assertRaises(ValueError): resolve_eval_packing(profile,path)
            write({'max_workspace_tokens':36864, 'max_sequence_tokens':300000})
            with self.assertRaises(ValueError): resolve_eval_packing(profile,path)
        self.assertEqual(resolve_eval_packing(profile)['effective_packing'],profile['packing_contract'])

    def test_supervisor_forwards_explicit_override(self):
        args = parser().parse_args(['--checkpoint','checkpoint','--out','output','--run-id','test',
            '--view-mode','ac_protect','--ratio','4','--max-new-tokens','4096','--task-ids','task',
            '--max-decisions','1','--max-generation-calls','1','--max-wall-seconds','60',
            '--eval-capacity','capacity.json'])
        command = _child_command(args)
        self.assertEqual(command[command.index('--eval-capacity')+1],str(Path('capacity.json').resolve()))


if __name__ == '__main__':
    unittest.main()
