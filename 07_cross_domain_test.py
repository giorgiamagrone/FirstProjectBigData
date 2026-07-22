
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, f1_score, accuracy_score

MODEL_FILE = "svm_model_final.pkl"
OOD_FILE   = "out_of_domain_test.csv" 

def main():
    print("=" * 60)
    print("STEP 07 — Cross-Domain Generalization Test")
    print("=" * 60)

    if not os.path.exists(MODEL_FILE):
        print(f"❌ Modello '{MODEL_FILE}' non trovato. Esegui lo Step 06!")
        return
    if not os.path.exists(OOD_FILE):
        print(f"❌ Dataset OOD '{OOD_FILE}' non trovato.")
        return

    # 1. Caricamento Modello Gold
    print(f"\n[1/3] Caricamento modello addestrato '{MODEL_FILE}'...")
    pipe = joblib.load(MODEL_FILE)

    # 2. Caricamento Dati Out-Of-Domain
    print(f"\n[2/3] Caricamento dataset Out-Of-Domain '{OOD_FILE}'...")
    df_ood = pd.read_csv(OOD_FILE).dropna(subset=["text", "rating"])
    df_ood["text"] = df_ood["text"].astype(str)
    df_ood["rating"] = df_ood["rating"].astype(float)
    print(f"      Trovati {len(df_ood):,} campioni di test.")

    # 3. Predizione e Valutazione
    X_test = df_ood["text"]
    y_test = df_ood["rating"]
    
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "─" * 50)
    print("─" * 50)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Macro : {f1_macro:.4f}")
    print("\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 60)

if __name__ == "__main__":
    main()