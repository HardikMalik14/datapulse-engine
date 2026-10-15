"""Bounded, stateless portfolio demo for the original DataPulse engine."""
from __future__ import annotations
import base64
import binascii
import csv
import io
import json
import logging
import os
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Literal

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/datapulse-matplotlib')
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split
from datapulse import DataPulseEngine, PipelineConfig
from datapulse.config import NumericalImputationStrategy, OutlierMethod, OutlierAction, ResamplingStrategy
from datapulse.data.synthetic import generate_messy_dataset
from datapulse.exceptions import DataPulseError

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='DataPulse — Pipeline Lab', version='1.0.0')
WORKER = threading.BoundedSemaphore(1)
log = logging.getLogger(__name__)


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    source: Literal['synthetic', 'upload'] = 'synthetic'
    seed: int = Field(42, ge=0, le=10000)
    rows: int = Field(600, ge=200, le=1200)
    missing_rate: float = Field(.12, ge=0, le=.4)
    inject_drift: bool = True
    file_name: str = Field('', max_length=200)
    file_data: str = Field('', max_length=1400000)
    target: str = Field('churned', min_length=1, max_length=120)
    task: Literal['classification', 'regression'] = 'classification'
    test_size: float = Field(.25, ge=.2, le=.4)
    imputer: Literal['median', 'mean', 'knn'] = 'knn'
    outliers: Literal['iqr', 'hybrid', 'none'] = 'hybrid'
    action: Literal['clip', 'mask', 'flag'] = 'clip'
    scaling: bool = True
    selection: bool = True
    sampler: Literal['smote', 'borderline_smote', 'none'] = 'smote'


def records(df, index=False):
    return json.loads((df.reset_index() if index else df).to_json(orient='records', date_format='iso'))


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@app.middleware('http')
async def body_limit(request: Request, call_next):
    if request.method == 'POST':
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 1500000:
                return JSONResponse({'detail': 'Use a CSV under 1 MB.'}, status_code=413)
        request._body = bytes(body)
    return await call_next(request)


def load_upload(o):
    try:
        raw = base64.b64decode(o.file_data, validate=True)
        text = raw.decode('utf-8-sig')
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError('Upload a valid UTF-8 CSV file.') from exc
    if len(raw) > 1000000:
        raise ValueError('Use a CSV under 1 MB.')
    reader = csv.reader(io.StringIO(text))
    header = next(reader, [])
    if not header or len(set(header)) != len(header) or any(not c.strip() for c in header):
        raise ValueError('CSV headers must be unique and nonempty.')
    df = pd.read_csv(io.StringIO(text), nrows=1201)
    if not 100 <= len(df) <= 1200 or not 3 <= len(df.columns) <= 35:
        raise ValueError('Use 100–1,200 rows and 3–35 columns, including the target.')
    if any(len(str(c)) > 120 for c in df.columns):
        raise ValueError('Column names must be under 120 characters.')
    if df.select_dtypes(include='object').apply(lambda c: c.astype(str).str.len().max()).max() > 300:
        raise ValueError('Use tabular features; text cells must be under 300 characters.')
    numeric = df.select_dtypes(include='number').to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError('Replace infinite feature values before uploading; ordinary missing cells are supported.')
    return df


def split_data(o):
    if o.source == 'synthetic':
        if o.task != 'classification':
            raise ValueError('The synthetic churn demo is classification. Upload a CSV for regression.')
        return generate_messy_dataset(n_rows=o.rows, random_state=o.seed, missing_rate=o.missing_rate,
                                      split=True, test_size=o.test_size, inject_drift=o.inject_drift,
                                      target_column=o.target)
    df = load_upload(o)
    if o.target not in df:
        raise ValueError('Choose a target column present in your CSV.')
    if df[o.target].isna().any():
        raise ValueError('Target labels cannot be missing. Imputation applies to feature columns.')
    stratify = None
    if o.task == 'classification':
        counts = df[o.target].value_counts()
        if not 2 <= len(counts) <= 5 or counts.min() < 8:
            raise ValueError('Classification needs 2–5 classes with at least eight rows each. Choose regression for a continuous target.')
        stratify = df[o.target]
    else:
        target = pd.to_numeric(df[o.target], errors='coerce')
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError('Regression target must contain finite numeric values.')
        df[o.target] = target
    train, test = train_test_split(df, test_size=o.test_size, random_state=o.seed, stratify=stratify)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_config(o, directory):
    cfg = PipelineConfig(target_column=o.target, task_type=o.task, random_state=o.seed, n_jobs=1, verbose=False)
    cfg.imputation.numerical_strategy = NumericalImputationStrategy(o.imputer)
    cfg.outliers.method = OutlierMethod(o.outliers)
    cfg.outliers.action = OutlierAction(o.action)
    cfg.outliers.contamination = .03
    cfg.scaling.enabled = o.scaling
    cfg.feature_selection.enabled = o.selection
    cfg.resampling.strategy = ResamplingStrategy(o.sampler)
    cfg.reporting.enabled = False
    cfg.reporting.output_dir = directory / 'reports'
    cfg.serialization.output_dir = directory / 'artifacts'
    return cfg


def profile(frame, target):
    features = frame.drop(columns=[target])
    return [{'column': c, 'dtype': str(features[c].dtype), 'missing': int(features[c].isna().sum()),
             'missing_rate': float(features[c].isna().mean()), 'unique': int(features[c].nunique())} for c in features]


def counts(y):
    return {str(k): int(v) for k, v in y.value_counts().sort_index().items()}


def run_pipeline(o):
    started = time.perf_counter()
    train, holdout = split_data(o)
    with tempfile.TemporaryDirectory(prefix='datapulse-') as folder:
        cfg = build_config(o, Path(folder))
        engine = DataPulseEngine(cfg)
        X, y = engine.fit_transform(train)
        Xh, yh = engine.transform(holdout, with_target=True)
        if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(Xh.to_numpy(dtype=float)).all():
            raise ValueError('This configuration did not yield finite features. Try a different imputer or enable scaling.')
        drift = engine.audit_drift(train, holdout)
        stages = records(engine.summary(detail_chars=1000))
        schema = records(engine.schema.to_frame())
        scaling = records(engine.pipeline_.named_steps['scaler'].skew_report())
        selection = records(engine.pipeline_.named_steps['selector'].ranking())
        corr = X.iloc[:, :12].corr().replace([np.inf, -np.inf], np.nan)
        resampler = engine.resampler_
        artifact = engine.save()
        # Never load a client-provided serialized object. This bundle is generated here.
        output = X.reset_index(drop=True).copy()
        output[o.target] = y.to_numpy()
        holdout_output = Xh.reset_index(drop=True).copy()
        holdout_output[o.target] = yh.to_numpy()
        report = json_safe({'source': o.file_name if o.source == 'upload' else 'Synthetic churn demo',
            'target': o.target, 'task': o.task, 'settings': o.model_dump(exclude={'file_data'}),
            'train_rows_before': len(train), 'train_rows_after': len(X), 'holdout_rows': len(holdout),
            'holdout_rows_after': len(Xh), 'input_features': len(train.columns)-1, 'output_features': X.shape[1],
            'raw_missing': int(train.drop(columns=[o.target]).isna().sum().sum()),
            'output_missing': int(X.isna().sum().sum()), 'elapsed_seconds': round(time.perf_counter()-started, 3),
            'resampling_applied': resampler.was_applied_, 'resampling_reason': resampler.skip_reason_,
            'class_before': counts(train[o.target]) if o.task == 'classification' else {},
            'class_after': counts(y) if o.task == 'classification' else {},
            'holdout_class': counts(yh) if o.task == 'classification' else {},
            'profile': profile(train, o.target), 'schema': schema, 'stages': stages,
            'scaling': scaling, 'selection': selection, 'feature_names': list(X.columns),
            'drift': records(drift.to_frame()), 'drift_summary': drift.to_dict(),
            'correlation': json.loads(corr.to_json()),
            'raw_preview': records(train.head(8)), 'processed_preview': records(output.head(8)),
            'holdout_preview': records(holdout_output.head(8)),
            'split_note': 'The uploaded CSV uses a seeded random split (stratified for classification). It is not a chronological production-drift audit.' if o.source == 'upload' else
                          ('Synthetic holdout has planted income and acquisition-channel shifts.' if o.inject_drift else 'Synthetic holdout has no deliberately injected shifts; sampling variation can still trigger drift flags.')})
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('train_processed.csv', output.to_csv(index=False))
            archive.writestr('holdout_processed.csv', holdout_output.to_csv(index=False))
            archive.writestr('run_report.json', json.dumps(report, indent=2, allow_nan=False))
            archive.writestr('pipeline_config.json', cfg.to_json())
            archive.write(artifact, 'datapulse_pipeline.joblib')
            for sidecar in artifact.parent.glob('*.json'):
                archive.write(sidecar, sidecar.name)
            archive.writestr('README.txt', 'DataPulse generated bundle\n\nTraining data was created by fit_transform: OOF encoding and optional training-only resampling. Holdout was transformed with the fitted chain and never resampled. The target column is included in both CSV files.\n\nUse the datapulse-engine source package and compatible library versions to load the generated pipeline with DataPulseEngine.load(...). This artifact contains inference transformers only, not the training-only resampler. To train a model, use train_processed.csv; do not regenerate the training features with artifact.transform(train_raw). No downstream prediction model is trained by this website.\n')
        bundle = mem.getvalue()
        if len(bundle) > 2000000:
            raise ValueError('The export exceeds the web demo limit. Use fewer rows or features.')
        report['download_base64'] = base64.b64encode(bundle).decode()
        report['download_bytes'] = len(bundle)
        return report


@app.get('/', include_in_schema=False)
def home():
    return FileResponse(ROOT / 'public/index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'project': 'datapulse-engine'}


@app.post('/api/profile')
def inspect_data(o: Options):
    try:
        if o.source == 'upload':
            df = load_upload(o)
        else:
            df = generate_messy_dataset(n_rows=o.rows, random_state=o.seed, missing_rate=o.missing_rate)
        return {'columns': list(df.columns), 'rows': len(df), 'preview': records(df.head(4))}
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post('/api/run')
def run(o: Options):
    if not WORKER.acquire(blocking=False):
        raise HTTPException(429, 'Another pipeline is running. Try again shortly.')
    try:
        return run_pipeline(o)
    except (ValueError, TypeError, KeyError, DataPulseError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception('DataPulse run failed')
        raise HTTPException(500, 'Pipeline failed. Check the deployment logs for details.') from exc
    finally:
        WORKER.release()


@app.get('/api/sample.csv')
def sample(seed: int = Query(42, ge=0, le=10000), rows: int = Query(600, ge=200, le=1200)):
    df = generate_messy_dataset(n_rows=rows, random_state=seed)
    return Response(df.to_csv(index=False), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="datapulse_messy_sample.csv"'})
