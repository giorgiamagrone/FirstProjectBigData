import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder

# cleanlab >= 2.x
try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    CLEANLAB_AVAILABLE = True
except ImportError:
    print("⚠️  cleanlab non installato. Esegui: pip install cleanlab")
    CLEANLAB_AVAILABLE = False

INPUT_FILE       = "dataset_grezzo_sanitized.csv"
NOISY_FILE       = "noisy_labels.csv"
CLEAN_FILE       = "dataset_clean_subset.csv"
REPORT_FILE      = "noisy_label_report.txt"

MAX_SAMPLES_PER_CLASS = 3000  
CV_FOLDS = 5
RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("STEP 03 — Noisy Label Detection")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: '{INPUT_FILE}' non trovato. Esegui prima Step 02!")
        return

    if not CLEANLAB_AVAILABLE:
        print("❌ Installare cleanlab prima di procedere.")
        return

    # 1. Caricamento e preparazione 
    print(f"\n[1/6] Caricamento '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE).dropna(subset=["text", "rating"])
    df["text"] = df["text"].astype(str)
    df["rating"] = df["rating"].astype(float)
    print(f"      {len(df):,} righe caricate.")

    # 2. Undersampling per bilanciare la CV 
    print(f"\n[2/6] Undersampling (max {MAX_SAMPLES_PER_CLASS} per classe)...")
    if MAX_SAMPLES_PER_CLASS:
        # Metodo robusto per evitare la perdita della colonna 'rating'
        sampled_groups = []
        for r_val, group in df.groupby("rating"):
            n_take = min(len(group), MAX_SAMPLES_PER_CLASS)
            sampled_groups.append(group.sample(n=n_take, random_state=RANDOM_STATE))
        df = pd.concat(sampled_groups, ignore_index=True)
        
    print(f"      Dataset bilanciato: {len(df):,} righe")
    print(df["rating"].value_counts().sort_index().to_string())

    # 3. Vettorizzazione TF-IDF 
    print(f"\n[3/6] Vettorizzazione TF-IDF (max_features=10000)...")
    vectorizer = TfidfVectorizer(max_features=10_000, sublinear_tf=True, min_df=2)
    X = vectorizer.fit_transform(df["text"])

    # Encode labels come interi 0..4
    le = LabelEncoder()
    y = le.fit_transform(df["rating"])  
    print(f"      Classi: {le.classes_}")

    #4. Cross-validation per le Out-Of-Fold predicted probabilities 
    print(f"\n[4/6] Cross-validation {CV_FOLDS}-fold (Logistic Regression)...")
    print("      Questo step richiede qualche minuto...")
    clf = LogisticRegression(
        max_iter=1000,
        C=0.5,
        solver="saga",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    pred_probs = cross_val_predict(
        clf, X, y,
        cv=CV_FOLDS,
        method="predict_proba",
        n_jobs=-1
    )
    print(f"      Pred probs shape: {pred_probs.shape}")

    # 5. Cleanlab: Confident Learning 
    print(f"\n[5/6] Applicazione Confident Learning (Cleanlab)...")
    label_issues_idx = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        frac_noise=0.05  
    )
    quality_scores = get_label_quality_scores(labels=y, pred_probs=pred_probs)

    n_flagged = len(label_issues_idx)
    pct_flagged = n_flagged / len(df) * 100
    print(f"      Campioni flaggati come noisy: {n_flagged:,} ({pct_flagged:.1f}%)")

    # 6. Separazione noisy/clean e salvataggio 
    print(f"\n[6/6] Separazione e salvataggio...")
    df["label_quality_score"] = quality_scores
    df["is_label_issue"]      = False
    df.loc[label_issues_idx, "is_label_issue"] = True

    df["suggested_rating"] = le.inverse_transform(pred_probs.argmax(axis=1))

    df_noisy = df[df["is_label_issue"]].copy()
    df_clean = df[~df["is_label_issue"]].copy()

    df_noisy.to_csv(NOISY_FILE, index=False)
    df_clean.to_csv(CLEAN_FILE, index=False)

    report_lines = [
        "=" * 60,
        "NOISY LABEL DETECTION REPORT (Cleanlab Confident Learning)",
        "=" * 60,
        f"Dataset input         : {len(df):,} campioni",
        f"Campioni flaggati     : {n_flagged:,} ({pct_flagged:.1f}%)",
        f"Campioni puliti       : {len(df_clean):,}",
        "",
        "Distribuzione issues per classe:",
    ]
    for rating in sorted(df["rating"].unique()):
        sub = df_noisy[df_noisy["rating"] == rating]
        report_lines.append(f"  {rating:.0f}★ : {len(sub):,} flaggati")

    report_lines += [
        "",
        "Top-5 campioni più rumorosi (quality score più basso):",
    ]
    top5 = df_noisy.nsmallest(5, "label_quality_score")[["text", "rating", "suggested_rating", "label_quality_score"]]
    for _, row in top5.iterrows():
        report_lines.append(
            f"  rating={row['rating']:.0f}★ → suggested={row['suggested_rating']:.0f}★ "
            f"| score={row['label_quality_score']:.3f} | testo: {str(row['text'])[:80]}..."
        )
    report_lines.append("=" * 60)
    report_str = "\n".join(report_lines)
    print("\n" + report_str)
    with open(REPORT_FILE, "w") as f:
        f.write(report_str)

    print(f"\n Step 03 completato.")
    print(f"   → '{NOISY_FILE}' ({len(df_noisy):,} campioni da curare con LLM)")
    print(f"   → '{CLEAN_FILE}' ({len(df_clean):,} campioni già affidabili)\n")


if __name__ == "__main__":
    main()
