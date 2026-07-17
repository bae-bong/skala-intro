import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def build_and_evaluate_models(df):
    """다중 ML 모델을 학습하고 Accuracy, Precision, Recall, F1 지표를 비교 평가합니다."""
    salary_threshold = df['ConvertedCompYearly'].quantile(0.75)
    df['Target_High_Salary'] = (df['ConvertedCompYearly'] >= salary_threshold).astype(int)

    drop_features = ['ConvertedCompYearly', 'Target_High_Salary', 'CompTotal', 'Currency', 'ResponseId']
    feature_cols = [c for c in df.columns if c not in drop_features]
    X = df[feature_cols]
    y = df['Target_High_Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_num_cols = X.select_dtypes(include=['number']).columns.tolist()
    model_cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), model_num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), model_cat_cols)
        ])

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }

    best_score = 0.0
    best_model_name = ""
    best_pipeline = None
    model_results = []

    for name, clf in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # 다중 평가 지표 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        model_results.append({
            'Model': name, 
            'Accuracy': f"{acc*100:.2f}%", 
            'Precision': f"{prec:.4f}", 
            'Recall': f"{rec:.4f}", 
            'F1_Score': f"{f1:.4f}"
        })
        
        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(model_results)
    return best_pipeline, best_model_name, best_score, results_df