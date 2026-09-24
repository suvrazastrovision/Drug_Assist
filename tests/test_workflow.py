"""Offline regression checks for the reorganized notebooks and retrieval workflow."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import chromadb
import nbformat
import pandas as pd
from drug_assist.data import clean_data, build_documents
from drug_assist import retrieval


class WorkflowTests(unittest.TestCase):
    def test_cleaning_preserves_existing_dataset(self):
        raw = pd.read_csv(ROOT / 'data/raw/drug.csv')
        expected = pd.read_csv(ROOT / 'data/processed/drug_clean.csv', dtype={'Reviews': 'Int64'})
        actual = clean_data(raw)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        self.assertEqual(len(actual), 1753)
        docs = build_documents(actual)
        self.assertEqual(len(docs), 624)
        self.assertEqual(len({doc['id'] for doc in docs}), 624)
        self.assertEqual(docs[0]['id'], 'drug_0')

    def test_resume_preserves_flags_and_legacy_records_are_searchable(self):
        client = chromadb.EphemeralClient()
        collection = client.create_collection(
            name='test_resume', embedding_function=None,
            configuration={'hnsw': {'space': 'cosine'}},
        )
        vector = [1.0] + [0.0] * (retrieval.EMBEDDING_DIMENSIONS - 1)
        docs = [
            {'id': f'drug_{i}', 'document_text': f'text {i}',
             'metadata': {'drug': f'drug {i}', 'condition': 'example', 'needs_review': False}}
            for i in range(3)
        ]
        # Legacy record has no review flag; another record has been flagged already.
        collection.add(
            ids=['drug_0', 'drug_1'], documents=['text 0', 'text 1'],
            embeddings=[vector, vector],
            metadatas=[{'drug': 'drug 0', 'condition': 'example'},
                       {'drug': 'drug 1', 'condition': 'example', 'needs_review': True,
                        'review_reason': 'description mismatch'}],
        )
        calls = []
        def create(**kwargs):
            calls.append(kwargs['input'])
            return types.SimpleNamespace(data=[
                types.SimpleNamespace(index=i, embedding=vector)
                for i in reversed(range(len(kwargs['input'])))
            ])
        fake_client = types.SimpleNamespace(embeddings=types.SimpleNamespace(create=create))
        retrieval.index_missing(collection, docs, fake_client)
        self.assertEqual(calls, [['text 2']])
        retrieval.index_missing(collection, docs, fake_client)
        self.assertEqual(len(calls), 1)
        saved = collection.get(ids=['drug_1'], include=['metadatas'])['metadatas'][0]
        self.assertTrue(saved['needs_review'])
        self.assertEqual(saved['review_reason'], 'description mismatch')
        matches = retrieval.search(collection, vector, k=3)
        self.assertEqual({row['id'] for row in matches}, {'drug_0', 'drug_2'})
        self.assertEqual(retrieval.search(collection, vector, condition='absent'), [])
        retrieval.flag_for_review(collection, 'drug_0', 'needs source check')
        self.assertEqual([row['id'] for row in retrieval.search(collection, vector)], ['drug_2'])
        with self.assertRaisesRegex(ValueError, 'different text'):
            retrieval.pending_documents(collection, [{**docs[0], 'document_text': 'changed'}])
        with self.assertRaisesRegex(ValueError, 'Unknown record'):
            retrieval.flag_for_review(collection, 'missing', 'test')
        client.delete_collection('test_resume')

    def test_all_notebooks_execute_without_network(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp_root = Path(temporary)
            for name in ['01_data_ingestion.ipynb', '02_eda.ipynb', '03_retrieval.ipynb']:
                notebook = nbformat.read(ROOT / 'notebooks' / name, as_version=4)
                nbformat.validate(notebook)
                namespace = {'__name__': '__main__'}
                # Default notebook execution must not call OpenAI or open the real database.
                with patch.object(retrieval, 'database_path', return_value=temp_root / 'chroma'), \
                     patch.object(retrieval, 'openai_client', side_effect=AssertionError('Unexpected API setup')), \
                     patch('IPython.display.display'), patch.object(plt, 'show'), \
                     patch.object(chromadb, 'PersistentClient', return_value=chromadb.EphemeralClient()):
                    for index, cell in enumerate(notebook.cells):
                        if cell.cell_type == 'code':
                            exec(compile(cell.source, f'{name}:cell {index}', 'exec'), namespace)
                            # Test the offline path regardless of interactive user settings.
                            if 'configuration' in cell.metadata.get('tags', []):
                                namespace.update(ALLOW_EMBEDDING_API=False, INSERT_MISSING=False, CACHE_DIR=temp_root / 'cache')
                            # Redirect ingestion's output to a disposable CSV.
                            if 'CLEAN_PATH' in namespace and name.startswith('01_'):
                                namespace['CLEAN_PATH'] = temp_root / 'drug_clean.csv'
                plt.close('all')


    def test_exact_order_full_workflow_and_cached_rerun(self):
        import matplotlib.pyplot as plt
        from drug_assist.cache import read_embedding, save_embedding
        notebook = nbformat.read(ROOT / 'notebooks/03_retrieval.ipynb', as_version=4)
        headings = [cell.source.splitlines()[0] for cell in notebook.cells
                    if cell.cell_type == 'markdown' and cell.source.startswith('## ')]
        self.assertEqual([heading.split('.')[0] for heading in headings],
                         [f'## {i:02d}' for i in range(1, 12)])
        calls = []
        # Synthetic vectors test execution only; they are not retrieval-quality evidence.
        vector = [1.0] + [0.0] * (retrieval.EMBEDDING_DIMENSIONS - 1)
        def create(**kwargs):
            calls.append(kwargs['input'])
            return types.SimpleNamespace(data=[
                types.SimpleNamespace(index=i, embedding=vector)
                for i in reversed(range(len(kwargs['input'])))
            ])
        fake_api = types.SimpleNamespace(embeddings=types.SimpleNamespace(create=create))
        local_client = chromadb.EphemeralClient()
        local_client.get_or_create_collection(retrieval.COLLECTION_NAME, embedding_function=None)
        local_client.delete_collection(retrieval.COLLECTION_NAME)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            namespace = {'__name__': '__main__'}
            step_number = None
            step_code = {}
            with patch.object(retrieval, 'openai_client', return_value=fake_api), \
                 patch.object(retrieval, 'database_path', return_value=Path(directory) / 'chroma'), \
                 patch.object(chromadb, 'PersistentClient', return_value=local_client), \
                 patch('IPython.display.display'), patch.object(plt, 'show'):
                for cell in notebook.cells:
                    if cell.cell_type == 'markdown' and cell.source.startswith('## '):
                        step_number = int(cell.source.split('.')[0].replace('## ', ''))
                    if cell.cell_type != 'code':
                        continue
                    exec(compile(cell.source, f'notebook step {step_number}', 'exec'), namespace)
                    if 'configuration' in cell.metadata.get('tags', []):
                        namespace.update(ALLOW_EMBEDDING_API=True, INSERT_MISSING=True,
                                         CACHE_DIR=Path(directory) / 'cache')
                    if step_number is not None and step_number not in step_code:
                        step_code[step_number] = cell.source
                    if step_number == 7:
                        self.assertEqual(namespace['collection'].count(), 0,
                                         'Embedding step must not insert records')
                self.assertEqual(namespace['collection'].count(), 624)
                self.assertEqual(namespace['document']['document_text'], namespace['documents'][0]['document_text'])
                self.assertEqual(len(namespace['document']['embedding']), 1536)
                self.assertEqual(len(calls), 10)
                evaluation = namespace['evaluation_df']
                self.assertEqual(len(evaluation), 3)
                for _, row in evaluation.iterrows():
                    expected = row['correct_condition'] / namespace['TOP_K']
                    self.assertEqual(row['Precision (%)'], f'{expected:.1%}')
                # Saved flags must survive reusing and reinserting prepared documents.
                retrieval.flag_for_review(namespace['collection'], 'drug_1', 'test review')
                requests_before = len(calls)
                for number in [4, 7, 8, 9, 10, 11]:
                    exec(step_code[number], namespace)
                self.assertEqual(len(calls), requests_before, 'Cached rerun must not call OpenAI')
                saved = namespace['collection'].get(ids=['drug_1'], include=['metadatas'])
                self.assertTrue(saved['metadatas'][0]['needs_review'])
                self.assertNotIn('drug_1', [item['id'] for item in namespace['results']])
                self.assertNotIn('drug_1', [item['id'] for item in namespace['filtered_results']])
                # Cache keys must separate different models and text.
                cache = Path(directory) / 'cache'
                save_embedding(cache, 'test', 'model-a', 2, [0.5, 0.5])
                self.assertEqual(read_embedding(cache, 'test', 'model-a', 2), [0.5, 0.5])
                self.assertIsNone(read_embedding(cache, 'test', 'model-b', 2))
                with self.assertRaises(ValueError):
                    save_embedding(cache, 'bad', 'model-a', 2, [1.0])
                namespace['QUERY'] = 'a changed question'
                with self.assertRaisesRegex(ValueError, 'QUERY changed'):
                    exec(step_code[10], namespace)
            plt.close('all')
            local_client.delete_collection(retrieval.COLLECTION_NAME)


if __name__ == '__main__':
    unittest.main()
