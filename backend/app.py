from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / 'public'
MODELS = ROOT / 'models'
DATA = ROOT / 'data' / 'online_shoppers_intention.csv'

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path='')
CORS(app)

model = scaler = pca = feature_columns = None
model_error = None


def load_models():
    global model, scaler, pca, feature_columns, model_error
    try:
        model = joblib.load(MODELS / 'random_forest_model.pkl')
        scaler = joblib.load(MODELS / 'scaler.pkl')
        pca = joblib.load(MODELS / 'pca.pkl')
        feature_columns = joblib.load(MODELS / 'feature_columns.pkl')
        model_error = None
    except Exception as exc:
        model = scaler = pca = feature_columns = None
        model_error = str(exc)

load_models()

CATEGORICAL = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType']
NUMERIC = ['Administrative','Administrative_Duration','Informational','Informational_Duration',
           'ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay',
           'Weekend']


def make_features(payload):
    data = {}
    for key in NUMERIC:
        if key == 'Weekend':
            data[key] = int(bool(payload.get(key, 0)))
        else:
            value = payload.get(key, 0)
            data[key] = float(value)
    for key in CATEGORICAL:
        val = payload.get(key, '')
        if key in ['OperatingSystems', 'Browser', 'Region', 'TrafficType']:
            try:
                data[key] = int(float(val))
            except (ValueError, TypeError):
                data[key] = 1
        else:
            data[key] = str(val)
    df = pd.DataFrame([data])
    df['TotalPages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    df['TotalDuration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['AverageProductTime'] = df['ProductRelated_Duration'] / (df['ProductRelated'] + 1)
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=False, dtype=int)
    if feature_columns is None:
        raise RuntimeError('Model artifacts are not loaded. Run training/train_model.py first.')
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def analytics_from_payload(payload, probability):
    admin = float(payload.get('Administrative', 0))
    info = float(payload.get('Informational', 0))
    product = float(payload.get('ProductRelated', 0))
    admin_d = float(payload.get('Administrative_Duration', 0))
    info_d = float(payload.get('Informational_Duration', 0))
    product_d = float(payload.get('ProductRelated_Duration', 0))
    total_pages = admin + info + product
    total_duration = admin_d + info_d + product_d
    avg_product = product_d / (product + 1)
    page_signal = min(100, (total_pages / 60) * 55 + (total_duration / 4000) * 25 + min(float(payload.get('PageValues',0)), 20))
    engagement = round(max(0, min(100, page_signal)), 1)
    return {
        'total_pages': round(total_pages, 2),
        'total_duration': round(total_duration, 2),
        'average_product_time': round(avg_product, 2),
        'engagement_score': engagement,
        'probability': round(probability * 100, 2),
        'page_visits': [admin, info, product],
        'durations': [admin_d, info_d, product_d]
    }


@app.get('/')
def index():
    return send_from_directory(FRONTEND, 'index.html')


@app.get('/api/health')
def health():
    return jsonify({'ok': True, 'model_loaded': model is not None, 'model_error': model_error})


@app.get('/api/dataset-summary')
def dataset_summary():
    if not DATA.exists():
        return jsonify({'available': False, 'message': 'Dataset not found. Put online_shoppers_intention.csv in data/.'})
    df = pd.read_csv(DATA)
    revenue = df['Revenue'].astype(int)
    return jsonify({
        'available': True,
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'purchase': int(revenue.sum()),
        'no_purchase': int((revenue == 0).sum()),
        'purchase_rate': round(float(revenue.mean() * 100), 2),
        'months': df['Month'].value_counts().to_dict()
    })


@app.post('/api/predict')
def predict():
    if model is None:
        return jsonify({'error': 'Model artifacts are missing. Run python training/train_model.py after adding the dataset.'}), 503
    try:
        payload = request.get_json(force=True, silent=True) or {}
        X = make_features(payload)
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        pred = int(model.predict(X_pca)[0])
        probabilities = model.predict_proba(X_pca)[0]
        purchase_probability = float(probabilities[1])
        analytics = analytics_from_payload(payload, purchase_probability)
        return jsonify({
            'prediction': pred,
            'label': 'Purchase' if pred == 1 else 'No Purchase',
            'probability': purchase_probability,
            'analytics': analytics
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.get('/api/charts')
def charts():
    if not DATA.exists():
        return jsonify({'error': 'Dataset not found'}), 404
    df = pd.read_csv(DATA)
    df['Revenue'] = df['Revenue'].astype(int)
    return jsonify({
        'purchase_distribution': {
            'labels': ['No Purchase', 'Purchase'],
            'values': [int((df.Revenue == 0).sum()), int((df.Revenue == 1).sum())]
        },
        'page_visits': {
            'labels': ['Administrative', 'Informational', 'Product Related'],
            'values': [round(df.Administrative.mean(),2), round(df.Informational.mean(),2), round(df.ProductRelated.mean(),2)]
        },
        'duration': {
            'labels': ['Administrative', 'Informational', 'Product Related'],
            'values': [round(df.Administrative_Duration.mean(),2), round(df.Informational_Duration.mean(),2), round(df.ProductRelated_Duration.mean(),2)]
        },
        'monthly_purchase_rate': [
            {'month': m, 'rate': round(float(df.loc[df.Month == m, 'Revenue'].mean() * 100), 2)}
            for m in ['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec'] if (df.Month == m).any()
        ]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
