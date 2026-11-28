# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Run unchanged native/generated training plus numerical TP/CP/EP negative controls."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import sys
import textwrap

from common import (C8, assert_forward_coverage, command, compare_losses, dump, environment, inventory,
                    numerical_differences)
from compare_bundle_training import read_run

PACK = Path(__file__).resolve().parent


def mutate_copy(source: Path, destination: Path, kind: str) -> dict:
    """Perturb disposable artifact computation without copying native component implementations."""
    shutil.copytree(source, destination)
    path = next(destination.glob('modeling_*_gen_npu.py'))
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    class_name = 'EPGroupedExperts' if kind == 'ep' else 'CPGroupedQueryAttentionModule'
    owner = next((node for node in classes if node.name == class_name), None)
    expected = 'model.layers.0.' + ('mlp.experts' if kind == 'ep' else 'self_attn')
    native_import = kind == 'ep' and owner is None
    if native_import:
        owner, method, statement = _experts_mutation_site(classes)
        class_name, expected = owner.name, 'model.layers.0.mlp'
    else:
        method, statement = _forward_mutation_statement(owner, kind)
    if kind == 'cp':
        value = statement.value
        if isinstance(value, ast.Tuple):
            value.elts[0] = ast.BinOp(value.elts[0], ast.Mult(), ast.Constant(1.125))
        elif isinstance(value, ast.Name):
            statement.value = ast.Tuple(elts=[ast.BinOp(ast.Subscript(value, ast.Constant(0), ast.Load()),
                ast.Mult(), ast.Constant(1.125)), ast.Starred(ast.Subscript(value, ast.Slice(lower=ast.Constant(1)),
                ast.Load()), ast.Load())], ctx=ast.Load())
        else:
            raise ValueError('CP negative control requires one stable tuple result')
    else:
        statement.value = ast.BinOp(statement.value, ast.Mult(), ast.Constant(1.125))
    ast.fix_missing_locations(statement)
    lines = text.splitlines(keepends=True)
    lines[statement.lineno - 1:statement.end_lineno] = [textwrap.indent(ast.unparse(statement),
                                                                   ' ' * statement.col_offset) + '\n']
    path.write_text(''.join(lines), encoding='utf-8')
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    return {'kind': kind, 'class': class_name, 'method': method.name, 'multiplier': 1.125,
            'files': inventory(destination), 'expected_module': expected, 'native_import_call_site': native_import}



def _experts_mutation_site(classes):
    """Find the emitted invocation when EP reuses an imported native component."""
    sites = [(cls, fn, node) for cls in classes for fn in cls.body if isinstance(fn, ast.FunctionDef)
             for node in ast.walk(fn) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'experts'
             and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'self']
    if len(sites) != 1:
        raise ValueError('Expected one emitted EP invocation site')
    return sites[0]


def _forward_mutation_statement(owner, kind):
    """Locate the selected numerical operation in an emitted component body."""
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    candidates = [node for node in method.body if isinstance(node, ast.Return)]
    if kind == 'tp':
        candidates = [node for node in method.body if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == 'reduce_scatter']
    if len(candidates) != 1:
        raise ValueError(f'Expected one numerical mutation site: {owner.name}.{method.name}')
    return method, candidates[0]


def extended_comparison(reference: Path, candidate: Path, world: int, steps: int, negative: bool = False) -> dict:
    """Separate real numerical mismatches from a failed constructor or a rejected source mutation."""
    left = read_run(reference, world, steps)
    right = read_run(candidate, world, steps)
    rows = []
    for rank in range(world):
        before, after = left[1][rank], right[1][rank]
        assert before['initial_schema'] == after['initial_schema'], (rank, 'schema')
        assert before['initial_hashes'] == after['initial_hashes'], (rank, 'initial weights')
        assert before['inputs'] == after['inputs'], (rank, 'input values or RNG')
        if right[2][rank]['mode'] == 'generated':
            assert_forward_coverage(after, steps)
            assert right[2][rank]['constructed_from_artifact']
        differences = numerical_differences(before, after)
        loss_difference = None
        for index, (a, b) in enumerate(zip(left[0][rank]['steps'], right[0][rank]['steps']), 1):
            if a['loss'] != b['loss']:
                loss_difference = {'step': index, 'native': a['loss'], 'candidate': b['loss']}
                break
        if not negative:
            assert not differences, (rank, differences[:8])
            compare_losses(left[0][rank]['steps'], right[0][rank]['steps'], steps, 0, 0)
        rows.append({'rank': rank, 'differences': differences, 'first_loss_difference': loss_difference,
                     'artifact_forwards': len(after['forward_inventory']),
                     'executed_forwards': len(after['forward_calls']),
                     'forward_calls': after['forward_calls'], 'tp_calls': after['tp_calls'],
                     'module_output_calls': {name: len(values) for name, values in after['module_outputs'].items()},
                     'gradient_steps': len(after['gradients'])})
    return {'status': 'DETECTED_NUMERICAL_MISMATCH' if negative else 'PASS', 'ranks': rows}


def main() -> int:
    """Keep all numerical settings in the same YAML and frozen common overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = json.loads(args.manifest.read_text())
    model = settings['models']['qwen3_moe']
    options = [*model['overrides'], *(model.get('training_overrides') or []), *C8]
    steps = 20
    positive = args.output / 'positive'
    env = environment(settings.get('optional_pythonpath', ''))
    env.update(PYTHONHASHSEED='0', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')
    driver = [sys.executable, str(PACK / 'compare_bundle_training.py'), '--source', str(args.source),
              '--config', str(args.source / model['yaml']), '--entry', model['entry'], '--output', str(positive),
              '--nproc-per-node=8', '--steps=20', '--atol=0', '--rtol=0', '--timeout=1800',
              '--optional-pythonpath', settings.get('optional_pythonpath', ''), *options]
    report = {'status': 'RUNNING',
              'scope': 'All instantiated source forwards and all plan boundaries; Qwen3-MoE TP2 CP2 EP2',
              'steps': steps, 'positive': {}, 'negative_controls': []}
    try:
        code = command(driver, args.source, env, args.output / 'positive.log', 7200)
        assert code == 0, 'Positive training must complete before inspecting enhanced numerical coverage'
        for label in ('native-repeat', 'generated'):
            report['positive'][label] = extended_comparison(positive / 'native', positive / label, 8, steps)
        dump(args.output / 'report.json', report)
        base_report = json.loads((positive / 'comparison-report.json').read_text())
        base_argv = base_report['commands'][-1]['argv']
        original_files = inventory(positive / 'bundle')
        for kind in ('tp', 'cp', 'ep'):
            bundle = args.output / ('mutant-' + kind)
            mutation = mutate_copy(positive / 'bundle', bundle, kind)
            evidence = args.output / ('negative-' + kind)
            evidence.mkdir()
            argv = list(base_argv)
            argv[argv.index('--bundle') + 1] = str(bundle)
            argv[argv.index('--evidence') + 1] = str(evidence)
            code = command(argv, args.source, env, args.output / ('negative-' + kind + '.log'), 1800)
            assert code == 0, (kind, 'Negative control failed before completing real training')
            checked = extended_comparison(positive / 'native', evidence, 8, steps, negative=True)
            for rank in checked['ranks']:
                changed = {item.get('module') for item in rank['differences'] if item['category'] == 'module_outputs'}
                assert mutation['expected_module'] in changed, (kind, rank['rank'], 'mutated module not detected')
                assert '' in changed and 'lm_head' in changed, (kind, rank['rank'], 'logits not changed')
                assert any(item['category'] == 'optimizer_gradients' for item in rank['differences'])
                assert rank['first_loss_difference'] is not None, (kind, rank['rank'], 'loss mutation not detected')
            report['negative_controls'].append({'mutation': mutation, 'result': checked})
            assert inventory(positive / 'bundle') == original_files, 'Canonical output was changed'
            dump(args.output / 'report.json', report)
        report['status'] = 'PASS'
        return 0
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        dump(args.output / 'report.json', report)


if __name__ == '__main__':
    raise SystemExit(main())
