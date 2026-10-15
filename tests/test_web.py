import base64
import io
import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from datapulse import DataPulseEngine
from index import Options, app, split_data

client = TestClient(app)


def upload(frame, **kwargs):
    return {'source': 'upload', 'file_name': 'sample.csv', 'file_data': base64.b64encode(frame.to_csv(index=False).encode()).decode(), **kwargs}


def test_default_pipeline_export_is_finite_and_preserves_holdout():
    response = client.post('/api/run', json={})
    assert response.status_code == 200, response.text
    report = response.json()
    assert report['output_missing'] == 0
    assert report['holdout_rows_after'] == report['holdout_rows']
    assert report['resampling_applied']
    assert len(set(report['class_after'].values())) == 1
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(report['download_base64'])))
    training = pd.read_csv(archive.open('train_processed.csv'))
    holdout = pd.read_csv(archive.open('holdout_processed.csv'))
    assert list(training.columns) == list(holdout.columns)
    assert len(training) == report['train_rows_after']
    assert np.isfinite(training.to_numpy()).all()
    assert np.isfinite(holdout.to_numpy()).all()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'pipeline.joblib'
        path.write_bytes(archive.read('datapulse_pipeline.joblib'))
        fitted = DataPulseEngine.load(path)
        _, raw_holdout = split_data(Options())
        transformed = fitted.transform(raw_holdout.drop(columns=['churned']))
        np.testing.assert_allclose(transformed, holdout.drop(columns=['churned']), rtol=1e-6, atol=1e-8)
        assert 'resampler' not in fitted.engine.named_steps


def test_changed_seed_changes_actual_data_and_schema_decisions_exist():
    response = client.post('/api/run', json={'seed': 17, 'rows': 300, 'imputer': 'median', 'outliers': 'iqr'})
    assert response.status_code == 200, response.text
    d = response.json()
    assert d['raw_preview'][0]['customer_id'] != split_data(Options())[0].iloc[0]['customer_id']
    dropped = {row['name'] for row in d['schema'] if row['kind'] == 'dropped'}
    assert {'customer_id', 'data_version'} <= dropped
    assert len(d['stages']) == 9


def test_upload_profile_and_disabled_resampling():
    train, test = split_data(Options(rows=300))
    frame = pd.concat([train, test], ignore_index=True)
    data = upload(frame, target='churned', sampler='none', imputer='median', outliers='iqr')
    response = client.post('/api/profile', json=data)
    assert response.status_code == 200
    assert 'churned' in response.json()['columns']
    response = client.post('/api/run', json=data)
    assert response.status_code == 200, response.text
    d = response.json()
    assert not d['resampling_applied']
    assert d['train_rows_before'] == d['train_rows_after']
    assert 'random split' in d['split_note']


def test_regression_upload_works_without_class_balancing():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    data = pd.DataFrame({'x': x, 'z': rng.normal(size=200), 'group': np.where(x > 0, 'A', 'B'), 'target': 3*x + rng.normal(size=200)})
    response = client.post('/api/run', json=upload(data, target='target', task='regression', imputer='median', outliers='iqr'))
    assert response.status_code == 200, response.text
    d = response.json()
    assert not d['resampling_applied']
    assert d['train_rows_after'] == d['train_rows_before']
    assert not d['class_before']


def test_invalid_uploads_and_request_limits():
    duplicate = base64.b64encode(b'a,a,target\n1,2,0\n').decode()
    assert client.post('/api/profile', json={'source': 'upload', 'file_data': duplicate}).status_code == 400
    assert client.post('/api/run', json={'rows': 5000}).status_code == 422
    assert client.post('/api/run', json={'source': 'upload', 'file_data': '???'}).status_code == 400
    assert client.post('/api/run', content=b'x'*1500001).status_code == 413
    assert client.post('/api/run', json={'task': 'regression'}).status_code == 400


def test_schema_profile_empty_target_is_not_required_for_inspection():
    frame = pd.DataFrame({'feature': np.arange(120), 'other': np.arange(120)*2, 'label': [0, 1]*60})
    assert client.post('/api/profile', json=upload(frame)).status_code == 200
    missing = frame.copy()
    missing.loc[0, 'label'] = np.nan
    assert client.post('/api/run', json=upload(missing, target='label')).status_code == 400


def test_selection_and_scaling_can_be_disabled_and_sample_download_works():
    response = client.post('/api/run', json={'rows': 300, 'selection': False, 'scaling': False, 'sampler': 'none', 'imputer': 'median', 'outliers': 'iqr'})
    assert response.status_code == 200, response.text
    assert not response.json()['scaling']
    sample = client.get('/api/sample.csv?seed=17&rows=200')
    assert sample.status_code == 200
    frame = pd.read_csv(io.BytesIO(sample.content))
    assert len(frame) == 200 and 'churned' in frame
