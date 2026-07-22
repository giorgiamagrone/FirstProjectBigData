import pandas as pd
import numpy as np
import os

INPUT_FILE     = "curated_samples.csv"
METADATA_FILE  = "dataset_con_metadati.csv"   # opzionale, da Step 01b
OUTPUT_FILE    = "dataset_gold.csv"
REPORT_FILE    = "dataset_assembly_report.txt"

RANDOM_STATE = 42


def compute_class_balance_score(df: pd.DataFrame) -> float:
    counts = df["rating"].value_counts()
    n_classes = len(counts)
    if n_classes <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(n_classes)
    return float(entropy / max_entropy)


def main():
    print("=" * 60)
    print("STEP 05 (FINAL) — Dataset Assembly (provenance + metadati)")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: '{INPUT_FILE}' non trovato. Esegui prima Step 04!")
        return

    # 1. Caricamento 
    print(f"\n[1/4] Caricamento '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    df["text"]   = df["text"].astype(str).str.strip()
    df["rating"] = df["rating"].astype(float)
    df = df.dropna(subset=["text", "rating"])
    df = df[df["text"].str.len() >= 10]
    print(f"      {len(df):,} campioni disponibili dopo pulizia.")

    print("\n      Distribuzione classi (nessun taglio applicato):")
    dist = df["rating"].value_counts().sort_index()
    for r, n in dist.items():
        print(f"        {r:.0f}★ : {n:,}")

    if "source" not in df.columns:
        df["source"] = "unknown"

    # 2. ID e provenance 
    print(f"\n[2/4] Aggiunta ID e metadati di provenance...")
    df = df.reset_index(drop=True)
    df["sample_id"] = ["GOLD_" + str(i).zfill(6) for i in range(len(df))]

    if os.path.exists(METADATA_FILE):
        df_meta_source = pd.read_csv(METADATA_FILE)
        meta_cols = [c for c in ["asin", "title", "brand", "price", "main_category"] if c in df_meta_source.columns]
        if "asin" in df_meta_source.columns and len(meta_cols) > 1:
            # Join per testo (curated_samples.csv non porta 'asin' con sé dopo Step 04;
            # se disponibile un match testo->asin dal file metadati arricchito, lo usiamo)
            meta_lookup = df_meta_source[["text"] + [c for c in meta_cols if c != "asin"] + ["asin"]].drop_duplicates(subset=["text"])
            df = df.merge(meta_lookup, on="text", how="left")
            n_matched = df["asin"].notna().sum() if "asin" in df.columns else 0
            print(f"      Metadati prodotto uniti per {n_matched:,}/{len(df):,} campioni "
                  f"({100*n_matched/len(df):.1f}%)")
        else:
            print(f"      '{METADATA_FILE}' trovato ma senza le colonne attese; salto l'arricchimento.")
    else:
        print(f"      '{METADATA_FILE}' non trovato: procedo senza metadati prodotto "
              f"(esegui Step 01b prima di questo step se li vuoi includere).")

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 4. Statistiche finali e salvataggio 
    print(f"\n[4/4] Statistiche finali e salvataggio...")
    balance_score = compute_class_balance_score(df)

    report_lines = [
        "=" * 60,
        "DATASET GOLD — ASSEMBLY REPORT (no undersampling)",
        "=" * 60,
        f"Campioni totali    : {len(df):,}",
        f"Classi             : {sorted(df['rating'].unique())}",
        f"Balance score      : {balance_score:.4f} (1.0 = perfetto; qui NON forzato a 1.0,",
        f"                      riflette il bilanciamento naturale ereditato dallo Step 03)",
        "",
        "Distribuzione finale (nessun undersampling applicato in questo step):",
        f"  {'Rating':<8} {'Count':>8} {'%':>8}  Provenance",
    ]
    for rating in sorted(df["rating"].unique()):
        sub = df[df["rating"] == rating]
        pct = len(sub) / len(df) * 100
        provenance = sub["source"].value_counts().to_dict()
        prov_str = ", ".join([f"{k}:{v}" for k, v in provenance.items()])
        report_lines.append(f"  {rating:.0f}★       {len(sub):>8,}  {pct:>7.2f}%  [{prov_str}]")

    if "main_category" in df.columns:
        report_lines += ["", "Copertura metadati prodotto:"]
        report_lines.append(f"  Con metadati    : {df['title'].notna().sum():,}" if "title" in df.columns else "")
        report_lines.append(f"  Senza metadati  : {df['title'].isna().sum():,}" if "title" in df.columns else "")

    report_lines += [
        "",
        f"File output: {OUTPUT_FILE}",
        "=" * 60,
    ]
    report_str = "\n".join(l for l in report_lines if l != "")
    print("\n" + report_str)
    with open(REPORT_FILE, "w") as f:
        f.write(report_str)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n Step 05 (FINAL) completato.")
    print(f"   → '{OUTPUT_FILE}' ({len(df):,} campioni, balance_score={balance_score:.3f})")
    print(f"   → '{REPORT_FILE}'\n")


if __name__ == "__main__":
    main()