import builtins
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from olist_pipeline import REQUIRED_COLUMNS, extract, prepare, validate


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'snapshot.csv'
        self.frame = pd.DataFrame({
            'order_id': ['001', '002', '003', '004'],
            'order_purchase_timestamp': ['2017-07-29', '2017-07-30', '2017-07-31', '2017-08-01'],
            'price': [10., 20., 30., 40.], 'freight': [1., 2., 3., 4.],
            'weight': [100., 200., 300., 400.], 'n_items': [1, 1, 2, 1],
            'installments': [1., 2., 3., 1.], 'est_days': [10., 12., 14., 15.],
            'customer_state': ['SP', 'RJ', 'MG', 'SP'], 'seller_state': ['SP'] * 4,
            'category': ['a', 'b', 'unseen', 'a'], 'payment_type': ['card'] * 4,
            'late': [0, 1, 0, 1],
        })
        self.frame.to_csv(self.path, index=False)

    def test_valid_snapshot_and_validate_read_only(self):
        before = self.path.read_bytes()
        original_open = builtins.open

        def readonly_open(file, mode='r', *args, **kwargs):
            if any(flag in mode for flag in ('w', 'a', 'x', '+')):
                raise AssertionError('validate attempted a write')
            return original_open(file, mode, *args, **kwargs)

        with ExitStack() as guards:
            guards.enter_context(patch('builtins.open', readonly_open))
            for name in ('pathlib.Path.mkdir', 'pathlib.Path.write_text', 'pathlib.Path.write_bytes',
                         'os.chdir', 'subprocess.run', 'subprocess.Popen', 'socket.socket'):
                guards.enter_context(patch(name, side_effect=AssertionError(name)))
            self.assertEqual(validate(self.path), str(self.path.resolve()))
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_every_required_column_is_checked(self):
        for column in REQUIRED_COLUMNS:
            with self.subTest(column=column):
                self.frame.drop(columns=column).to_csv(self.path, index=False)
                with self.assertRaises(ValueError):
                    validate(self.path)

    def test_invalid_values_and_duplicates(self):
        cases = [('price', np.nan), ('freight', np.inf), ('weight', -np.inf),
                 ('n_items', 'not-a-number'), ('installments', np.nan), ('est_days', np.inf),
                 ('late', 2), ('late', np.nan), ('order_purchase_timestamp', 'invalid'),
                 ('order_purchase_timestamp', None), ('order_id', None), ('order_id', ' ')]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                frame = self.frame.copy().astype({column: 'object'})
                frame.loc[0, column] = value
                frame.to_csv(self.path, index=False)
                with self.assertRaises(ValueError):
                    validate(self.path)
        pd.concat([self.frame, self.frame.iloc[:1]]).to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            validate(self.path)
        self.frame.iloc[:0].to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            validate(self.path)
        self.path.write_text(','.join([*REQUIRED_COLUMNS, 'price']) + '\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            validate(self.path)

    def test_exact_cutoff_and_original_snapshot_hash(self):
        with patch.dict(os.environ, {'OLIST_CUTOFF': '2017-07-31'}):
            result = prepare(self.path)
        train = pd.read_csv(result['train'], dtype={'order_id': 'string'})
        validation = pd.read_csv(result['validation'], dtype={'order_id': 'string'})
        self.assertEqual(train.order_id.tolist(), ['001', '002'])
        self.assertEqual(validation.order_id.tolist(), ['003', '004'])
        self.assertEqual(result['snapshot_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertNotEqual(result['snapshot_sha256'], hashlib.sha256(Path(result['train']).read_bytes()).hexdigest())
        self.assertEqual((result['train_rows'], result['validation_rows']), (2, 2))
        self.assertLess(len(json.dumps(result)), 4096)

    def test_extract_replaces_instead_of_appending_and_isolates_runs(self):
        engine = MagicMock()
        sqlalchemy = SimpleNamespace(create_engine=MagicMock(return_value=engine), text=lambda sql: sql)
        with patch.dict(sys.modules, {'sqlalchemy': sqlalchemy}), \
                patch.dict(os.environ, {'DATABASE_URL': 'postgresql+psycopg://unused',
                                        'OLIST_WORKDIR': str(self.root)}), \
                patch('pandas.read_sql_query', return_value=self.frame):
            first = Path(extract('manual__first'))
            first_bytes = first.read_bytes()
            repeated = Path(extract('manual__first'))
            second = Path(extract('manual__second'))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertEqual(first_bytes, repeated.read_bytes())
        self.assertEqual(first_bytes, second.read_bytes())
        self.assertEqual(len(pd.read_csv(first)), 4)
        self.assertTrue(first.is_absolute())
        engine.dispose.assert_called()


if __name__ == '__main__':
    unittest.main()
