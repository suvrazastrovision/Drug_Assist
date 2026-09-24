"""Reusable embedding cache keyed by text, model, and dimensions for repeating lookups."""
import hashlib
import json
import math
from pathlib import Path


def cache_key(text, model, dimensions):
    value = json.dumps([model, dimensions, text], ensure_ascii=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def validate_vector(vector, dimensions):
    if len(vector) != dimensions or not all(math.isfinite(float(x)) for x in vector):
        raise ValueError('Embedding must have the expected dimensions and finite values.')


def read_embedding(folder, text, model, dimensions):
    path = Path(folder) / (cache_key(text, model, dimensions) + '.json')
    if not path.exists():
        return None
    item = json.loads(path.read_text(encoding='utf-8'))
    if (item['text'], item['model'], item['dimensions']) != (text, model, dimensions):
        raise ValueError('Cached embedding does not match the requested text and model.')
    validate_vector(item['embedding'], dimensions)
    return item['embedding']


def save_embedding(folder, text, model, dimensions, vector):
    validate_vector(vector, dimensions)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (cache_key(text, model, dimensions) + '.json')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({
        'text': text, 'model': model, 'dimensions': dimensions,
        'embedding': [float(x) for x in vector],
    }), encoding='utf-8')
    temporary.replace(path)
