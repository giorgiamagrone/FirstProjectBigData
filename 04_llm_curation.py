import pandas as pd
import requests
import json
import os
import time
from tqdm import tqdm

NOISY_FILE      = "noisy_labels.csv"
CLEAN_FILE      = "dataset_clean_subset.csv"
CURATED_FILE    = "curated_samples.csv"
CHECKPOINT_FILE = "curated_checkpoint.csv"
FAILURE_LOG     = "llm_failures.log"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
BATCH_SIZE   = 5
TIMEOUT      = 120
MAX_RETRIES  = 3
TEMPERATURE  = 0.3   

VALID_RATINGS = {1, 2, 3, 4, 5}

# Contatori globali di visibilità
STATS = {"llm_calls": 0, "llm_success": 0, "llm_failures": 0, "failure_reasons": [],
         "invalid_ratings_discarded": 0}


def sanitize_llm_record(rec: dict, required_keys=("text", "rating")) -> dict | None:
    if not all(k in rec for k in required_keys):
        STATS["invalid_ratings_discarded"] += 1
        return None
    try:
        r = float(rec["rating"])
    except (TypeError, ValueError):
        STATS["invalid_ratings_discarded"] += 1
        return None
    r_rounded = round(r)
    if r_rounded not in VALID_RATINGS:
        STATS["invalid_ratings_discarded"] += 1
        STATS["failure_reasons"].append(
            f"[rating_invalid] valore={r} scartato (testo: {str(rec.get('text',''))[:60]}...)"
        )
        return None
    if abs(r - r_rounded) > 1e-9:
        STATS["failure_reasons"].append(
            f"[rating_normalized] {r} → {r_rounded} (testo: {str(rec.get('text',''))[:60]}...)"
        )
    rec["rating"] = float(r_rounded)
    return rec


# Ollama health check 

def check_ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = [m.get("name", "") for m in r.json().get("models", [])]
        available = any(OLLAMA_MODEL in m for m in models)
        if not available:
            print(f"⚠️  Ollama raggiungibile, ma il modello '{OLLAMA_MODEL}' non risulta scaricato.")
            print(f"    Modelli disponibili: {models}")
        return available
    except Exception as e:
        print(f"Ollama non raggiungibile su {OLLAMA_URL.split('/api')[0]}: {e}")
        return False


# LLM helper 

def call_llm_label_correction(reviews: list[dict]) -> list[dict] | None:
    prompt = f"""Sei un esperto annotatore di sentiment. Analizza queste recensioni Amazon di musica.
Per ogni recensione, controlla se il rating assegnato corrisponde al sentiment del testo.
Se non corrisponde (es. testo molto positivo ma rating=1★, o testo negativo ma rating=5★), correggi il rating.
Il rating corretto suggerito è fornito come hint, ma usa il tuo giudizio.

RISPONDI ESCLUSIVAMENTE con un array JSON, anche se la lista di recensioni contiene un solo elemento.
Ogni elemento ha:
  "text": il testo ORIGINALE invariato
  "rating": il rating CORRETTO come numero INTERO (1, 2, 3, 4 o 5 — mai valori intermedi come 2.5)
  "corrected": true se hai cambiato il rating, false altrimenti
  "reason": breve motivazione (max 10 parole)

Dati:
{json.dumps(reviews, ensure_ascii=False)}
"""
    return _call_llm(prompt)


def _call_llm(prompt: str) -> list[dict] | None:
    STATS["llm_calls"] += 1
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": TEMPERATURE}
                },
                timeout=TIMEOUT
            )
            response.raise_for_status()
            raw = response.json().get("response", "")

            raw = raw.strip()
            if not raw.startswith("["):
                start = raw.find("[")
                end   = raw.rfind("]") + 1
                if start != -1 and end > start:
                    raw = raw[start:end]
                elif raw.startswith("{"):
                
                    single = json.loads(raw)
                    if all(k in single for k in ("text", "rating")):
                        STATS["llm_success"] += 1
                        return [single]

            data = json.loads(raw)
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        STATS["llm_success"] += 1
                        return v
                if all(k in data for k in ("text", "rating")):
                    STATS["llm_success"] += 1
                    return [data]
                raise ValueError(f"Risposta JSON è un dict senza liste interne: {list(data.keys())}")

            if isinstance(data, list):
                STATS["llm_success"] += 1
                return data

            raise ValueError(f"Formato JSON inatteso: {type(data)}")

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
                continue

    STATS["llm_failures"] += 1
    STATS["failure_reasons"].append(f"[label_correction] {type(last_error).__name__}: {last_error}")
    return None


# Main 

def main():
    print("=" * 60)
    print("STEP 04 (FINAL) — Label Correction Mirata (no augmentation)")
    print("=" * 60)

    if not os.path.exists(NOISY_FILE):
        print(f" Errore: '{NOISY_FILE}' non trovato. Esegui prima Step 03!")
        return
    if not os.path.exists(CLEAN_FILE):
        print(f"Errore: '{CLEAN_FILE}' non trovato. Esegui prima Step 03!")
        return

    print(f"\n[0/3] Verifica connessione a Ollama ({OLLAMA_URL})...")
    if not check_ollama_available():
        print(f"\nSTOP: Ollama non è raggiungibile o '{OLLAMA_MODEL}' non è scaricato.")
        print(f"   Apri un terminale separato e lancia: ollama serve")
        print(f"   Poi, se necessario: ollama pull {OLLAMA_MODEL}")
        return
    print("      Ollama raggiungibile e modello disponibile.")

    df_noisy = pd.read_csv(NOISY_FILE)
    df_clean = pd.read_csv(CLEAN_FILE)
    print(f"      Campioni flaggati (da correggere): {len(df_noisy):,}")
    print(f"      Campioni puliti (già affidabili) : {len(df_clean):,}")

    print(f"\n[2/3] Label Correction LLM su {len(df_noisy):,} campioni flaggati...")

    corrected_records = []
    if os.path.exists(CHECKPOINT_FILE):
        corrected_records = pd.read_csv(CHECKPOINT_FILE).to_dict("records")
        start_idx = len(corrected_records)
        print(f"      Checkpoint trovato: riprendo da riga {start_idx}")
    else:
        start_idx = 0

    noisy_list = df_noisy.to_dict("records")

    for i in tqdm(range(start_idx, len(noisy_list), BATCH_SIZE), desc="Label Correction"):
        batch = noisy_list[i:i + BATCH_SIZE]
        payload = [
            {
                "text": str(r["text"])[:500],
                "rating": int(r["rating"]),
                "suggested_rating": int(r.get("suggested_rating", r["rating"]))
            }
            for r in batch
        ]
        result = call_llm_label_correction(payload)

        if result:
            n_valid_in_batch = 0
            for rec in result:
                rec["source"] = "corrected" if rec.get("corrected") else "verified"
                rec = sanitize_llm_record(rec)
                if rec is None:
                    continue
                corrected_records.append(rec)
                n_valid_in_batch += 1
        
            if n_valid_in_batch < len(batch):
                covered_texts = {rec["text"] for rec in corrected_records[-n_valid_in_batch:]} if n_valid_in_batch else set()
                for r in batch:
                    if r["text"] not in covered_texts:
                        corrected_records.append({
                            "text": r["text"],
                            "rating": r["rating"],
                            "corrected": False,
                            "reason": "llm_partial_batch_fallback",
                            "source": "original_noisy"
                        })
        else:
            for r in batch:
                corrected_records.append({
                    "text": r["text"],
                    "rating": r["rating"],
                    "corrected": False,
                    "reason": "llm_error_fallback",
                    "source": "original_noisy"
                })

        if i > 0 and i % 50 == 0:
            pd.DataFrame(corrected_records).to_csv(CHECKPOINT_FILE, index=False)

    df_corrected = pd.DataFrame(corrected_records)[["text", "rating", "source"]]
    n_actually_corrected = sum(1 for r in corrected_records if r.get("corrected") is True)
    print(f"      Labels corrette dall'LLM: {n_actually_corrected:,} su {len(df_noisy):,}")
    print(f"      Chiamate LLM fallite: {STATS['llm_failures']}/{STATS['llm_calls']}")
    print(f"      Record scartati per rating non valido: {STATS['invalid_ratings_discarded']}")

    print(f"\n[3/3] Assemblaggio curated_samples.csv...")
    df_clean_tagged = df_clean[["text", "rating"]].copy()
    df_clean_tagged["source"] = "clean_original"

    df_all = pd.concat([df_clean_tagged, df_corrected], ignore_index=True)
    df_all["text"]   = df_all["text"].astype(str)
    df_all["rating"] = df_all["rating"].astype(float)
    df_all = df_all.dropna(subset=["text", "rating"])

    # Validazione finale di sicurezza (difesa in profondità)
    before_final = len(df_all)
    df_all = df_all[df_all["rating"].apply(lambda x: round(x) in VALID_RATINGS and abs(x - round(x)) < 1e-9)]
    dropped_final = before_final - len(df_all)
    if dropped_final > 0:
        print(f"      🧹 Scartati {dropped_final} campioni con rating non valido, sfuggiti ai controlli precedenti.")

    df_all.to_csv(CURATED_FILE, index=False)

    with open(FAILURE_LOG, "w") as f:
        f.write(f"Chiamate LLM totali : {STATS['llm_calls']}\n")
        f.write(f"Chiamate riuscite   : {STATS['llm_success']}\n")
        f.write(f"Chiamate fallite    : {STATS['llm_failures']}\n")
        f.write(f"Record scartati per rating non valido: {STATS['invalid_ratings_discarded']}\n\n")
        if STATS["failure_reasons"]:
            f.write("Dettaglio (fino a 100):\n")
            for reason in STATS["failure_reasons"][:100]:
                f.write(f"  - {reason}\n")

    print(f"   Chiamate LLM: {STATS['llm_calls']} totali, {STATS['llm_success']} riuscite, "
          f"{STATS['llm_failures']} fallite ({100*STATS['llm_failures']/max(1,STATS['llm_calls']):.1f}%)")
    print(f"   Campioni totali: {len(df_all):,}")
    print(f"   → clean_original     : {len(df_clean_tagged):,}")
    print(f"   → corrected/verified/original_noisy : {len(df_corrected):,}")
    print(f"   Salvato: '{CURATED_FILE}'")
    print(f"   Log: '{FAILURE_LOG}'\n")
    print("   Distribuzione finale:")
    print(df_all["rating"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()