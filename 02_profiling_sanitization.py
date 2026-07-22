import pandas as pd
import numpy as np
import os

INPUT_FILE  = "dataset_grezzo.csv"
OUTPUT_FILE = "dataset_grezzo_sanitized.csv"
REPORT_FILE = "profiling_report.txt"


def profile_dataset(df: pd.DataFrame) -> dict:
    stats = {}

    # Dimensioni
    stats["n_rows"]    = len(df)
    stats["n_cols"]    = len(df.columns)
    stats["columns"]   = list(df.columns)

    # Missing values
    stats["missing"] = df.isnull().sum().to_dict()
    stats["missing_pct"] = (df.isnull().mean() * 100).round(2).to_dict()

    # Distribuzione classi
    class_counts = df["rating"].value_counts().sort_index()
    stats["class_distribution"] = class_counts.to_dict()
    stats["class_pct"] = (class_counts / len(df) * 100).round(2).to_dict()

    # Statistiche testo
    df_tmp = df.copy()
    df_tmp["text_len"] = df_tmp["text"].astype(str).str.len()
    stats["text_len_mean"]   = df_tmp["text_len"].mean()
    stats["text_len_median"] = df_tmp["text_len"].median()
    stats["text_len_min"]    = df_tmp["text_len"].min()
    stats["text_len_max"]    = df_tmp["text_len"].max()
    stats["text_len_p5"]     = df_tmp["text_len"].quantile(0.05)
    stats["text_len_p95"]    = df_tmp["text_len"].quantile(0.95)

    # Duplicati esatti (stessa text + rating)
    stats["n_duplicates"] = df.duplicated(subset=["text", "rating"]).sum()
    stats["n_near_dup_text"] = df.duplicated(subset=["text"]).sum()  # stesso testo, rating diverso

    # Sbilanciamento: rapporto max/min
    majority  = class_counts.max()
    minority  = class_counts.min()
    stats["imbalance_ratio"] = round(majority / minority, 2)

    return stats


def write_report(stats: dict, path: str):
    lines = [
        "=" * 60,
        "=" * 60,
        "",
        f"Righe totali      : {stats['n_rows']:,}",
        f"Colonne           : {stats['columns']}",
        "",
    ]
    for col, n in stats["missing"].items():
        lines.append(f"  {col:<20}: {n:>6} ({stats['missing_pct'][col]:.2f}%)")
    lines += [
        "",
        f"  {'Rating':<8} {'Count':>8} {'%':>8}",
        "  " + "-" * 26,
    ]
    for rating, count in sorted(stats["class_distribution"].items()):
        pct = stats["class_pct"][rating]
        bar = "█" * int(pct / 2)
        lines.append(f"  {rating:.0f} stelle  {count:>8,}  {pct:>7.2f}%  {bar}")
    lines += [
        "",
        f"  Imbalance ratio (max/min) : {stats['imbalance_ratio']:.2f}x",
        "",
        f"  Lunghezza media           : {stats['text_len_mean']:.1f} chars",
        f"  Lunghezza mediana         : {stats['text_len_median']:.1f} chars",
        f"  Min / Max                 : {stats['text_len_min']:.0f} / {stats['text_len_max']:.0f} chars",
        f"  Percentile 5%  / 95%      : {stats['text_len_p5']:.0f} / {stats['text_len_p95']:.0f} chars",
        "",
        f"  Duplicati esatti (text+rating) : {stats['n_duplicates']:,}",
        f"  Stesso testo, rating diverso   : {stats['n_near_dup_text']:,}",
        "",
        "=" * 60,
    ]
    report_str = "\n".join(lines)
    print(report_str)
    with open(path, "w") as f:
        f.write(report_str)


def main():
    print("=" * 60)
    print("STEP 02 — Data Profiling & Sanitization")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: '{INPUT_FILE}' non trovato. Esegui prima Step 01!")
        return

    # 1. Caricamento 
    print(f"\n[1/4] Caricamento '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    print(f"      {len(df):,} righe caricate.")

    # 2. Profiling 
    print("\n[2/4] Esecuzione profiling...")
    stats = profile_dataset(df)
    write_report(stats, REPORT_FILE)
    print(f"\n      Report salvato in '{REPORT_FILE}'")

    # 3. Sanitizzazione 
    print("\n[3/4] Sanitizzazione...")
    before = len(df)

    # Rimozione duplicati esatti
    df = df.drop_duplicates(subset=["text", "rating"])
    after_dedup = len(df)
    print(f"      Duplicati rimossi: {before - after_dedup:,}")

    # Filtro testi troppo corti (< 10 caratteri dopo strip)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 10]
    after_len = len(df)
    print(f"      Testi troppo corti rimossi: {after_dedup - after_len:,}")

    # Reset index
    df = df.reset_index(drop=True)
    print(f"      Dataset sanitizzato: {len(df):,} righe totali.")

    # ── 4. Salvataggio 
    print(f"\n[4/4] Salvataggio '{OUTPUT_FILE}'...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"       Salvato: '{OUTPUT_FILE}' ({len(df):,} righe)")

    majority_class = max(stats["class_distribution"], key=stats["class_distribution"].get)
    minority_class = min(stats["class_distribution"], key=stats["class_distribution"].get)
    print(f"   Classe maggioritaria ({majority_class:.0f}★): {stats['class_pct'][majority_class]:.1f}%")
    print(f"   Classe minoritaria   ({minority_class:.0f}★): {stats['class_pct'][minority_class]:.1f}%")
    print(f"   Imbalance ratio: {stats['imbalance_ratio']:.1f}x → target per Cleanlab nel Step 03\n")


if __name__ == "__main__":
    main()
