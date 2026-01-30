import pandas as pd
import pickle
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from dataset.utils.text_clean import clean_text
from dataset.utils.url_features import extract_url_features

# Load datasets
msg_data = pd.read_csv("dataset/messages.csv")
url_data = pd.read_csv("dataset/urls.csv")

# -------- Message Model --------
print("Training Message Fraud Detection Model...")
msg_data['text'] = msg_data['text'].apply(clean_text)

# Enhanced TF-IDF vectorizer with better parameters
vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=1,
    max_df=1.0,
    ngram_range=(1, 2),  # Include bigrams for better context
    sublinear_tf=True,
    stop_words='english'
)
X_text = vectorizer.fit_transform(msg_data['text'])
y_text = msg_data['label']

# Use SVC with probability for better spam detection
text_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
text_model.fit(X_text, y_text)

# Cross-validation score
if X_text.shape[0] > 2:
    cv_scores = cross_val_score(text_model, X_text, y_text, cv=min(3, X_text.shape[0]))
    print(f"Text Model Cross-validation Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

pickle.dump(text_model, open("model/text_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
print("✓ Message model trained and saved")

# -------- URL Model --------
print("\nPreparing URL dataset for training (using verified_online.csv as primary source)...")
import pathlib
phish_file = pathlib.Path("dataset/verified_online.csv")

if phish_file.exists():
    phish_df = pd.read_csv(phish_file)
    # keep only verified & online entries
    phish_df = phish_df[(phish_df['verified'].str.lower() == 'yes') & (phish_df['online'].str.lower() == 'yes')]
    phish_df = phish_df[['url', 'phish_id', 'phish_detail_url', 'submission_time', 'verified', 'verification_time', 'online', 'target']].dropna(subset=['url'])
    phish_df['label'] = 1
    print(f"Found {len(phish_df)} verified online phishing URLs (positives)")

    # Prepare negatives: use urls.csv where label==0, plus small seed of common legit sites
    neg_df = url_data[url_data['label'] == 0][['url','label']].copy()
    if len(neg_df) < len(phish_df):
        seed_safe = pd.DataFrame({
            'url': [
                'https://google.com', 'https://youtube.com', 'https://facebook.com', 'https://amazon.com', 'https://wikipedia.org',
                'https://twitter.com', 'https://instagram.com', 'https://linkedin.com', 'https://apple.com', 'https://microsoft.com',
                'https://reddit.com', 'https://netflix.com', 'https://paypal.com', 'https://pinterest.com', 'https://stackoverflow.com',
                'https://bing.com', 'https://yahoo.com', 'https://whatsapp.com', 'https://etsy.com', 'https://booking.com',
                'https://airbnb.com', 'https://dropbox.com', 'https://slideshare.net', 'https://stackoverflow.com', 'https://mozilla.org',
                'https://ubuntu.com', 'https://cloudflare.com', 'https://salesforce.com', 'https://adobe.com', 'https://medium.com',
                'https://quora.com', 'https://telegram.org', 'https://tiktok.com', 'https://stripe.com', 'https://shopify.com',
                'https://paypal.com', 'https://bbc.co.uk', 'https://nytimes.com', 'https://washingtonpost.com', 'https://theguardian.com'
            ],
            'label': [0]*40
        })
        # Augment seed with user-provided negatives file (optional) so you can add hosts like varcode.in to negatives
        try:
            seed_path_csv = pathlib.Path('dataset/negatives_seed.csv')
            seed_path_txt = pathlib.Path('dataset/negatives_seed.txt')
            additional = None
            if seed_path_csv.exists():
                try:
                    additional = pd.read_csv(seed_path_csv)
                    # Expect columns: url,label (label optional, default 0)
                    if 'label' not in additional.columns:
                        additional['label'] = 0
                except Exception:
                    additional = None
            elif seed_path_txt.exists():
                try:
                    lines = [l.strip() for l in open(seed_path_txt, 'r', encoding='utf-8').read().splitlines() if l.strip()]
                    additional = pd.DataFrame({'url': lines, 'label': [0]*len(lines)})
                except Exception:
                    additional = None
            if additional is not None and not additional.empty:
                seed_safe = pd.concat([seed_safe, additional[['url','label']].dropna(subset=['url'])], ignore_index=True)
                print(f"Loaded {len(additional)} additional negative seeds from dataset/negatives_seed.*")
        except Exception as _e:
            print('Could not load additional negative seeds:', _e)

        neg_df = pd.concat([neg_df, seed_safe], ignore_index=True).drop_duplicates(subset=['url']).reset_index(drop=True)
        print(f"Expanded negative seed to {len(neg_df)} unique legit urls")
    # Augment negatives with common variants to reduce false positives (http variants, ww. typos, path variants)
    try:
        from dataset.utils.url_normalize import normalize_url
        aug = []
        sample_hosts = list(neg_df['url'].dropna().unique())[:500]
        base_hosts = []
        for u in sample_hosts:
            info = normalize_url(u)
            h = info.get('netloc','')
            if h and h not in base_hosts:
                base_hosts.append(h)
        # Ensure we have some base hosts from seed too
        if not base_hosts:
            base_hosts = ['google.com','github.com','microsoft.com','stackoverflow.com']

        target_neg_count = min(20000, max(len(phish_df), 2000))  # cap augment size
        i = 0
        while len(neg_df) + len(aug) < target_neg_count:
            host = base_hosts[i % len(base_hosts)]
            # Create several path/query variants to make them unique
            path_variant = f"/safepath/{i % 1000}"
            q_variant = f"?ref=seed{i%50}"
            aug.append({'url': f'https://{host}{path_variant}', 'label': 0})
            aug.append({'url': f'http://{host}{path_variant}{q_variant}', 'label': 0})
            aug.append({'url': f'http://ww.{host}{path_variant}', 'label': 0})
            i += 1
            if i > 100000:
                break

        if aug:
            aug_df = pd.DataFrame(aug).drop_duplicates(subset=['url'])
            neg_df = pd.concat([neg_df, aug_df], ignore_index=True).drop_duplicates(subset=['url']).reset_index(drop=True)
            print(f"Added {len(aug_df)} augmented negative variants (path/query variants) to reduce false positives")
    except Exception as e:
        print('Could not augment negatives:', e)

    # Now we should have many unique negatives; sample or upsample to balance classes
    from sklearn.utils import resample
    if len(neg_df) < len(phish_df):
        neg_df_up = resample(neg_df, replace=True, n_samples=len(phish_df), random_state=42)
    else:
        neg_df_up = neg_df.sample(n=min(len(neg_df), len(phish_df)), random_state=42)

    # Combine positives and negatives with class balancing: sample positives to match negatives
    neg_sample_size = len(neg_df_up)
    neg_sample = neg_df_up[['url','label']].sample(n=neg_sample_size, random_state=42)

    # If too many positives, downsample positives to match negatives for balanced training
    pos_sample_size = min(len(phish_df), neg_sample_size)
    pos_sample = phish_df[['url','label']].sample(n=pos_sample_size, random_state=42)

    combined = pd.concat([pos_sample, neg_sample], ignore_index=True)
    # Keep unique URLs, prefer phishing label if conflict
    combined = combined.groupby('url', as_index=False)['label'].max()

    # If user provided a file of forced negatives (one URL per line or CSV), ensure they are present and forced as negatives
    try:
        forced_txt = pathlib.Path('dataset/forced_negatives.txt')
        forced_csv = pathlib.Path('dataset/forced_negatives.csv')
        forced = []
        if forced_csv.exists():
            try:
                fdf = pd.read_csv(forced_csv)
                if 'url' in fdf.columns:
                    forced = [u for u in fdf['url'].dropna().astype(str).tolist()]
            except Exception:
                forced = []
        elif forced_txt.exists():
            try:
                forced = [l.strip() for l in open(forced_txt, 'r', encoding='utf-8').read().splitlines() if l.strip()]
            except Exception:
                forced = []
        if forced:
            forced_set = set(forced)
            # Add any missing forced negatives
            for u in forced_set:
                if u not in set(combined['url']):
                    combined = pd.concat([combined, pd.DataFrame({'url':[u],'label':[0]})], ignore_index=True)
            # Force labels to 0 for the forced list (override conflicts)
            combined.loc[combined['url'].isin(forced_set), 'label'] = 0
            print(f"Applied {len(forced_set)} forced negative examples to the training set")
    except Exception as e:
        print('Could not apply forced negatives:', e)

    print(f"Combined balanced dataset prepared: {len(combined)} samples (phish={combined['label'].sum()}, legit={len(combined)-combined['label'].sum()})")
else:
    # fallback to original behavior
    combined = url_data[['url','label']].copy()
    print(f"verified_online.csv not found; using existing labeled `urls.csv` ({len(combined)} samples)")

# Clean: drop empty and malformed URLs
combined = combined[combined['url'].notna() & (combined['url'].str.strip() != '')].reset_index(drop=True)
combined.to_csv('dataset/urls_train.csv', index=False)
print('Saved combined training set to dataset/urls_train.csv')

# Extract features and labels
X_url = [extract_url_features(u) for u in combined['url']]
y_url = combined['label']

# Use multiple candidate models for URL fraud detection (including lightweight options)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
import json
import argparse

# Optional CLI arg to limit training data size for faster experiments
parser = argparse.ArgumentParser(description='Train URL models with optional subsampling')
parser.add_argument('--max-samples', type=int, default=0, help='Limit number of training samples to this many (0 = use all)')
args, _rest = parser.parse_known_args()
max_samples = int(getattr(args, 'max_samples', 0)) if args else 0
print(f"[train_model] max_samples={max_samples}")

# If requested, subsample combined dataset for speed (random stratified-like sample)
if max_samples and max_samples < len(combined):
    try:
        combined = combined.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled combined dataset to {len(combined)} samples for faster training")
    except Exception:
        pass

# Re-extract features and labels after potential subsampling
X_url = [extract_url_features(u) for u in combined['url']]
y_url = combined['label']

# Split dataset into train / calibration / test for reliable probability calibration
if len(X_url) >= 10:
    X_tmp, X_test, y_tmp, y_test = train_test_split(X_url, y_url, test_size=0.2, stratify=y_url, random_state=42)
    X_train, X_calib, y_train, y_calib = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)
else:
    X_train, X_calib, X_test = X_url, X_url, X_url
    y_train, y_calib, y_test = y_url, y_url, y_url

# Candidate models: include a lightweight logistic regression and hist-GB for speed
candidates = {
    'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'hgb': HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=5, random_state=42),
    'logreg': LogisticRegression(max_iter=400, C=1.0, class_weight='balanced', solver='liblinear', random_state=42)
}

best_model = None
best_metrics = None
best_name = None

for name, model in candidates.items():
    try:
        model.fit(X_train, y_train)
        # Calibrate on calibration set (sigmoid) if available
        try:
            calibrator = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
            calibrator.fit(X_calib, y_calib)
            used_model = calibrator
            calibrated_used = True
        except Exception:
            used_model = model
            calibrated_used = False

        # Evaluate
        classes = list(used_model.classes_) if hasattr(used_model, 'classes_') else list(model.classes_)
        fraud_index = classes.index(1) if 1 in classes else -1
        probs = used_model.predict_proba(X_test)[:, fraud_index] if fraud_index >= 0 else np.max(used_model.predict_proba(X_test), axis=1)
        brier = brier_score_loss(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float('nan')

        print(f"Model {name}: Brier={brier:.4f}, ROC_AUC={roc_auc:.4f}, calibrated={calibrated_used}")

        # Prefer model with smallest Brier score; if similar, prefer simpler model (logreg) for speed
        if best_metrics is None or brier < best_metrics['brier'] - 1e-6 or (abs(brier - best_metrics['brier']) < 1e-6 and name == 'logreg'):
            best_metrics = {'brier': brier, 'roc_auc': roc_auc, 'calibrated': calibrated_used}
            best_model = used_model
            best_name = name
    except Exception as e:
        print(f"Skipping model {name} due to error: {e}")

if best_model is None:
    # fallback to original gradient boosting
    url_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    url_model.fit(X_url, y_url)
    pickle.dump(url_model, open("model/url_model.pkl", "wb"))
    print("No candidate succeeded; saved default GradientBoosting model")
else:
    # Save selected model and optionally a lightweight alias for faster inference
    pickle.dump(best_model, open("model/url_model.pkl", "wb"))
    if best_name in ('logreg', 'hgb'):
        pickle.dump(best_model, open("model/url_model_light.pkl", "wb"))

    # Determine recommended threshold based on test split
    classes = list(best_model.classes_) if hasattr(best_model, 'classes_') else list(candidates['gb'].classes_)
    fraud_index = classes.index(1) if 1 in classes else -1
    probs = best_model.predict_proba(X_test)[:, fraud_index] if fraud_index >= 0 else np.max(best_model.predict_proba(X_test), axis=1)
    brier = brier_score_loss(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float('nan')

    thresholds = np.linspace(0.0, 1.0, 101)
    best = None
    # Prefer thresholds that keep false positive rate <= 1% and maximize precision (to reduce false alarms)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn = ((preds == 0) & (np.array(y_test) == 0)).sum()
        fp = ((preds == 1) & (np.array(y_test) == 0)).sum()
        fn = ((preds == 0) & (np.array(y_test) == 1)).sum()
        tp = ((preds == 1) & (np.array(y_test) == 1)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if fpr <= 0.01:
            if best is None or precision > best['precision'] or (abs(precision - best['precision']) < 1e-9 and recall > best['recall']):
                best = {'threshold': t, 'fpr': fpr, 'recall': recall, 'precision': precision, 'f1': f1}
    if best is None:
        # fallback: pick threshold that maximizes F1
        best_f1 = {'threshold': 0.5, 'f1': -1}
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tn = ((preds == 0) & (np.array(y_test) == 0)).sum()
            fp = ((preds == 1) & (np.array(y_test) == 0)).sum()
            fn = ((preds == 0) & (np.array(y_test) == 1)).sum()
            tp = ((preds == 1) & (np.array(y_test) == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1['f1']:
                best_f1 = {'threshold': t, 'f1': f1, 'precision': precision, 'recall': recall}
        best = {'threshold': best_f1['threshold'], 'fpr': None, 'recall': best_f1.get('recall', 0), 'precision': best_f1.get('precision', 0), 'f1': best_f1['f1']}

    # Enforce a conservative minimum recommended threshold to avoid pathological cases
    recommended_threshold = float(best['threshold'])
    original_recommended = recommended_threshold
    MIN_RECOMMENDED = 0.5
    threshold_clamped = False
    if recommended_threshold < MIN_RECOMMENDED:
        recommended_threshold = MIN_RECOMMENDED
        threshold_clamped = True

    print(f"Selected model: {best_name}, recommended threshold: {original_recommended:.2f} -> used {recommended_threshold:.2f} (clamped={threshold_clamped}) (brier={brier:.4f}, roc_auc={roc_auc:.4f})")

    calib_info = {
        'brier_score': float(brier),
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'recommended_threshold': recommended_threshold,
        'original_recommended_threshold': original_recommended,
        'threshold_clamped': threshold_clamped,
        'calibrated': bool(best_metrics.get('calibrated', False) if best_metrics else False),
        'model_name': best_name
    }
    with open('model/url_model_calibration_info.json', 'w', encoding='utf-8') as fh:
        json.dump(calib_info, fh, indent=2)
    print('✓ URL model trained, selected, and saved with calibration info')

print("\n✓ All models trained successfully!")