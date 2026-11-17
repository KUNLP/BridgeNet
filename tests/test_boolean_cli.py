"""Execute released argparse and mode dispatch AST without importing model dependencies."""
import argparse
import ast
import contextlib
import io
from pathlib import Path
import shlex
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT/'run_nli.py').read_text(encoding='utf-8'))
FLAGS = ['separate_sentence_pair', 'both_label_relation', 'verbose_logging',
         'do_lower_case', 'no_cuda', 'from_init_weight', 'do_train', 'do_eval',
         'do_predict', 'draw_tsne', 'draw_attn_map']

def parse(argv):
    block = next(n for n in TREE.body if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(ast.parse("__name__ == '__main__'", mode='eval').body))
    functions = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'str_to_bool']
    body = functions + block.body[:-1]  # omit main(cli_args), which loads models
    ns = {'argparse': argparse}
    with patch.object(sys, 'argv', ['run_nli.py'] + argv):
        module = ast.parse('')
        module.body = body
        exec(compile(module, 'run_nli.py', 'exec'), ns)
    return ns['cli_args']

def dispatch(args):
    main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in main.body if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(ast.parse('args.do_train', mode='eval').body))
    calls=[]
    ns={'args': args, 'model': None, 'tokenizer': None, 'logger': None,
        'train':lambda *a,**k:calls.append('train'),
        'evaluate':lambda *a,**k:calls.append('evaluate'),
        'predict':lambda *a,**k:calls.append('predict')}
    module = ast.parse('')
    module.body = [branch]
    exec(compile(module, 'run_nli.py', 'exec'), ns)
    return calls

class BooleanTests(unittest.TestCase):
    def test_all_flags(self):
        forms = (
            (True, ('true', 'TRUE', 'TrUe', 't', 'yes', 'YeS', 'y', 'on', '1')),
            (False, ('false', 'FALSE', 'FaLsE', 'f', 'no', 'nO', 'n', 'off', '0')),
        )
        for flag in FLAGS:
            for expected, values in forms:
                for text in values:
                    with self.subTest(flag=flag, text=text):
                        self.assertIs(getattr(parse(['--'+flag, text]), flag), expected)
            for value in ['maybe', '', '2', 'truth', ' true ']:
                with self.subTest(invalid=flag, value=value):
                    error = io.StringIO()
                    with contextlib.redirect_stderr(error):
                        with self.assertRaises(SystemExit) as exc:
                            parse(['--'+flag, value])
                    self.assertEqual(exc.exception.code, 2)
                    self.assertIn('expected a boolean value', error.getvalue())

    def test_flag_coverage(self):
        block = next(n for n in TREE.body if isinstance(n, ast.If))
        boolean_flags = []
        for node in ast.walk(block):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                types = [k.value for k in node.keywords if k.arg == 'type']
                if types and isinstance(types[0], ast.Name) and types[0].id in ('bool', 'str_to_bool'):
                    boolean_flags.append(node.args[0].s.lstrip('-'))
        self.assertCountEqual(boolean_flags, FLAGS)

    def test_defaults(self):
        args=parse([])
        for flag in FLAGS: self.assertIs(getattr(args,flag), flag == 'do_predict')
        self.assertEqual(dispatch(args), ['predict'])

    def test_readme_commands(self):
        commands=[line for line in (ROOT/'README.md').read_text(encoding='utf-8').splitlines() if line.startswith('python run_nli.py ')]
        self.assertEqual(len(commands),2)
        for command,expected in zip(commands,['train','predict']):
            with self.subTest(command=command):self.assertEqual(dispatch(parse(shlex.split(command)[2:])),[expected])
        self.assertEqual(dispatch(parse(['--do_eval','true'])),['evaluate'])

if __name__ == '__main__':unittest.main()
