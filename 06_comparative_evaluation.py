import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RAW_WITH_DUPES_FILE = "dataset_grezzo.csv"            # Step 01 — no dedup
RAW_FILE             = "dataset_grezzo_sanitized.csv"  # Step 02 — dedup
CLEAN_FILE           = "dataset_clean_subset.csv"
CORRECTED_FILE       = "dataset_gold.csv"

OUTPUT_REPORT = "comparative_results_FINAL.txt"
OUTPUT_TABLE  = "comparison_table_FINAL.csv"
TEST_SET_FILE = "fixed_test_set.csv"
MODEL_FILE    = "svm_model_final.pkl"

TEST_PER_CLASS = 200
RANDOM_STATE   = 42
CLASSES        = [1.0, 2.0, 3.0, 4.0, 5.0]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10_000, sublinear_tf=True, min_df=2, ngram_range=(1, 2)
        )),
        ("svm", LinearSVC(random_state=RANDOM_STATE, dual=False, max_iter=2000, C=1.0))
    ])


def build_fixed_test_set(df_clean: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for rating, group in df_clean.groupby("rating"):
        n_take = min(len(group), TEST_PER_CLASS)
        groups.append(group.sample(n=n_take, random_state=RANDOM_STATE))
        if n_take < TEST_PER_CLASS:
            print(f"      ⚠️  Rating {rating:.0f}★: solo {n_take} disponibili (< {TEST_PER_CLASS})")
    df_test = pd.concat(groups, ignore_index=True)
    df_test = df_test[["text", "rating"]].drop_duplicates(subset=["text"])
    return df_test.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_matched_stratified(df_raw: pd.DataFrame, df_corrected: pd.DataFrame,
                              random_state: int) -> pd.DataFrame:
    target_counts = df_corrected["rating"].value_counts().to_dict()
    groups = []
    for rating, n_target in sorted(target_counts.items()):
        pool = df_raw[df_raw["rating"] == rating]
        n_take = min(len(pool), n_target)
        if n_take < n_target:
            print(f"      ⚠️  Rating {rating:.0f}★: solo {n_take:,} disponibili nel raw "
                  f"(< {n_target:,} richiesti dalla distribuzione target)")
        groups.append(pool.sample(n=n_take, random_state=random_state))
    df_matched = pd.concat(groups, ignore_index=True)
    return df_matched.sample(frac=1, random_state=random_state).reset_index(drop=True)


def remove_leakage(df: pd.DataFrame, test_texts: set) -> pd.DataFrame:
    before = len(df)
    df = df[~df["text"].isin(test_texts)].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"      🧹 Rimossi {removed:,} campioni di training presenti anche nel test set (leakage)")
    return df


def evaluate(name: str, df_train: pd.DataFrame, X_test, y_test, save_model: bool = False) -> dict:
    df_train = df_train.dropna(subset=["text", "rating"]).copy()
    df_train["text"]   = df_train["text"].astype(str)
    df_train["rating"] = df_train["rating"].astype(float)

    # Difesa in profondità: scarta rating non validi (es. 2.5) prima del training
    before_valid = len(df_train)
    df_train = df_train[df_train["rating"].apply(lambda x: round(x) in CLASSES and abs(x - round(x)) < 1e-9)]
    invalid_dropped = before_valid - len(df_train)
    if invalid_dropped > 0:
        print(f"  ⚠️  Scartati {invalid_dropped} campioni con rating non valido prima del training")

    print(f"\n{'─'*55}")
    print(f"  Esperimento: {name}")
    print(f"{'─'*55}")
    print(f"  Campioni di training: {len(df_train):,}")
    dist = df_train["rating"].value_counts().sort_index()
    for r, n in dist.items():
        print(f"    {r:.0f}★ : {n:>7,}")

    pipe = build_pipeline()
    pipe.fit(df_train["text"], df_train["rating"])
    y_pred = pipe.predict(X_test)

    if save_model:
        joblib.dump(pipe, MODEL_FILE)
        print(f"\n  💾 Modello salvato: '{MODEL_FILE}'")

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=CLASSES, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy   : {acc:.4f}")
    print(f"  F1 Macro   : {f1_macro:.4f}  ← su TEST SET FISSO E CONDIVISO")
    print(f"  F1 Weighted: {f1_weighted:.4f}")

    return {
        "name": name,
        "n_train": len(df_train),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_per_class": {str(r): round(f, 4) for r, f in zip(CLASSES, f1_per_class)},
        "report": report,
    }


def main():
    print("=" * 60)
    print("=" * 60)

    missing = [f for f in [RAW_WITH_DUPES_FILE, RAW_FILE, CLEAN_FILE, CORRECTED_FILE] if not os.path.exists(f)]
    if missing:
        print(f"❌ File mancanti: {missing}")
        return

    # 1. Test set fisso 
    print("\n[1/4] Costruzione test set fisso e bilanciato (da dataset_clean_subset.csv)...")
    df_clean_subset = pd.read_csv(CLEAN_FILE)
    df_clean_subset["text"] = df_clean_subset["text"].astype(str)
    df_clean_subset["rating"] = df_clean_subset["rating"].astype(float)

    df_test = build_fixed_test_set(df_clean_subset)
    test_texts = set(df_test["text"])
    df_test.to_csv(TEST_SET_FILE, index=False)
    print(f"      ✅ Test set fisso: {len(df_test):,} campioni ({TEST_PER_CLASS} per classe circa)")
    print(df_test["rating"].value_counts().sort_index().to_string())

    X_test = df_test["text"]
    y_test = df_test["rating"]

    # 2. Caricamento e pulizia (con rimozione leakage) 
    print("\n[2/4] Caricamento training set e rimozione overlap col test set...")

    # 2a. V0_with_duplicates Step 01, filtri strutturali ma NO dedup
    print("\n      Caricamento dataset_grezzo.csv (pre-deduplicazione, Step 01)...")
    df_v0_dupes = pd.read_csv(RAW_WITH_DUPES_FILE)
    df_v0_dupes["text"] = df_v0_dupes["text"].astype(str)
    df_v0_dupes["rating"] = df_v0_dupes["rating"].astype(float)
    df_v0_dupes = remove_leakage(df_v0_dupes, test_texts)

    # 2b. V1_full — Step 02, sanitizzato e deduplicato
    df_raw = pd.read_csv(RAW_FILE)
    df_raw["text"] = df_raw["text"].astype(str)
    df_raw["rating"] = df_raw["rating"].astype(float)
    df_raw = remove_leakage(df_raw, test_texts)

    print(f"\n      V0_with_duplicates: {len(df_v0_dupes):,} campioni")
    print(f"      V1_full (deduplicato): {len(df_raw):,} campioni")
    print(f"      Differenza (duplicati esatti non rimossi in V0): {len(df_v0_dupes) - len(df_raw):,}")

    # 2c. V1_corrected — Step 04+05
    df_corrected = pd.read_csv(CORRECTED_FILE)
    df_corrected["text"] = df_corrected["text"].astype(str)
    df_corrected["rating"] = df_corrected["rating"].astype(float)
    df_corrected = remove_leakage(df_corrected, test_texts)

    # 2d. V1_matched — raw deduplicato, campionato stratificato per classe
    print("\n      Costruzione V1_matched con campionamento stratificato per classe...")
    df_v1_matched = build_matched_stratified(df_raw, df_corrected, RANDOM_STATE)
    n_match = len(df_v1_matched)
    print(f"      V1_matched: {n_match:,} campioni totali, distribuzione:")
    print(df_v1_matched["rating"].value_counts().sort_index().to_string())

    # 3. Esperimenti 
    print("\n[3/4] Esecuzione esperimenti (stesso test set per tutti)...")
    results = []
    results.append(evaluate(
        f"V0_with_duplicates — Raw + duplicati esatti non rimossi ({len(df_v0_dupes):,} camp.)",
        df_v0_dupes, X_test, y_test
    ))
    results.append(evaluate(
        "V1_full — Raw deduplicato (sbilanciato, noisy)",
        df_raw, X_test, y_test
    ))
    results.append(evaluate(
        f"V1_matched — Raw, {n_match:,} camp. (stessa size E stesso bilanciamento per classe, no curation)",
        df_v1_matched, X_test, y_test
    ))
    results.append(evaluate(
        "V1_corrected — Label correction mirata (Step 04+05)",
        df_corrected, X_test, y_test, save_model=True
    ))

    # Indici per leggibilità (l'ordine in `results` è: V0, V1_full, V1_matched, V1_corrected)
    r_v0        = results[0]
    r_full      = results[1]
    r_matched   = results[2]
    r_corrected = results[3]

    # ── 4. Report comparativo ─────────────────────────────────────────
    print("\n[4/4] Report comparativo finale...")
    print("\n" + "=" * 90)
    print("  CONFRONTO FINALE — stesso test set fisso per tutti gli esperimenti")
    print("=" * 90)
    print(f"  {'Esperimento':<70} {'N train':>10} {'Accuracy':>10} {'F1 Macro':>10}")
    print("  " + "─" * 102)
    for r in results:
        print(f"  {r['name']:<70} {r['n_train']:>10,} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}")


    # Effetto 1: deduplicazione (V0 → V1_full)
    delta_dedup = r_full["f1_macro"] - r_v0["f1_macro"]
    pct_dedup = delta_dedup / r_v0["f1_macro"] * 100 if r_v0["f1_macro"] else 0
    print(f"\n  1) Deduplicazione (V0_with_duplicates → V1_full):")
    print(f"     {r_full['name']:<70} {delta_dedup:+.4f} ({pct_dedup:+.1f}%)")

    # Effetto 2: bilanciamento delle classi (V1_full → V1_matched)
    delta_balance = r_matched["f1_macro"] - r_full["f1_macro"]
    pct_balance = delta_balance / r_full["f1_macro"] * 100 if r_full["f1_macro"] else 0
    print(f"\n  2) Bilanciamento delle classi (V1_full → V1_matched):")
    print(f"     {r_matched['name']:<70} {delta_balance:+.4f} ({pct_balance:+.1f}%)")

    # Effetto 3: correzione delle etichette (V1_matched → V1_corrected)
    delta_labels = r_corrected["f1_macro"] - r_matched["f1_macro"]
    pct_labels = delta_labels / r_matched["f1_macro"] * 100 if r_matched["f1_macro"] else 0
    print(f"\n  3) Correzione delle etichette (V1_matched → V1_corrected):")
    print(f"     {r_corrected['name']:<70} {delta_labels:+.4f} ({pct_labels:+.1f}%)")

    print("=" * 90)

    comparison_rows = []
    for r in results:
        row = {"experiment": r["name"], "n_train": r["n_train"],
               "accuracy": r["accuracy"], "f1_macro": r["f1_macro"], "f1_weighted": r["f1_weighted"]}
        for cls, val in r["f1_per_class"].items():
            row[f"f1_class_{cls}"] = val
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_TABLE, index=False)

    with open(OUTPUT_REPORT, "w") as f:
        f.write("COMPARATIVE EVALUATION REPORT (FINAL v3 — 4 esperimenti, shared held-out test set)\n")
        f.write("Data-Centric AI — Amazon CDs & Vinyl Sentiment\n")
        f.write(f"Test set fisso: {len(df_test):,} campioni reali, verificati, bilanciati\n")
        f.write("Decomposizione in tre effetti isolati: deduplicazione, bilanciamento, correzione etichette.\n")
        f.write("=" * 90 + "\n\n")
        for r in results:
            f.write(f"Esperimento: {r['name']}\n")
            f.write(f"N training: {r['n_train']:,}\n")
            f.write(f"Accuracy: {r['accuracy']:.4f}\n")
            f.write(f"F1 Macro: {r['f1_macro']:.4f}\n")
            f.write(f"F1 Weighted: {r['f1_weighted']:.4f}\n")
            f.write(f"\nClassification Report:\n{r['report']}\n")
            f.write("─" * 90 + "\n\n")

        f.write("DECOMPOSIZIONE DEGLI EFFETTI\n")
        f.write("─" * 90 + "\n")
        f.write(f"1) Deduplicazione (V0 → V1_full): {delta_dedup:+.4f} F1 Macro ({pct_dedup:+.1f}%)\n")
        f.write(f"2) Bilanciamento (V1_full → V1_matched): {delta_balance:+.4f} F1 Macro ({pct_balance:+.1f}%)\n")
        f.write(f"3) Correzione etichette (V1_matched → V1_corrected): {delta_labels:+.4f} F1 Macro ({pct_labels:+.1f}%)\n")

    print(f"\n Step 06 (FINAL v3) completato.")
    print(f"   → '{TEST_SET_FILE}' (test set condiviso, per riproducibilità)")
    print(f"   → '{OUTPUT_REPORT}'")
    print(f"   → '{OUTPUT_TABLE}'")
    print(f"   → '{MODEL_FILE}' (modello V1_corrected, pronto per Step 07)")


if __name__ == "__main__":
    main()