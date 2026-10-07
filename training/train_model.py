from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'online_shoppers_intention.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

CATEGORICAL = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType']

def features(df):
    df = df.copy().drop_duplicates()
    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)
    df['TotalPages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    df['TotalDuration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['AverageProductTime'] = df['ProductRelated_Duration'] / (df['ProductRelated'] + 1)
    X = df.drop(columns=['Revenue'])
    X = pd.get_dummies(X, columns=CATEGORICAL, drop_first=True, dtype=int)
    return X, df['Revenue']

if not DATA.exists():
    raise SystemExit(f'Dataset not found: {DATA}\nPut online_shoppers_intention.csv in the data folder and run again.')

df = pd.read_csv(DATA)
X, y = features(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train_pca, y_train)
pred = model.predict(X_test_pca)
proba = model.predict_proba(X_test_pca)[:, 1]

print(f'Accuracy : {accuracy_score(y_test, pred):.4f}')
print(f'Precision: {precision_score(y_test, pred, zero_division=0):.4f}')
print(f'Recall   : {recall_score(y_test, pred, zero_division=0):.4f}')
print(f'F1       : {f1_score(y_test, pred, zero_division=0):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, proba):.4f}')
print(f'PCA components: {pca.n_components_}')

joblib.dump(model, MODELS / 'random_forest_model.pkl')
joblib.dump(scaler, MODELS / 'scaler.pkl')
joblib.dump(pca, MODELS / 'pca.pkl')
joblib.dump(X.columns.tolist(), MODELS / 'feature_columns.pkl')
print('Model artifacts saved in models/.')
