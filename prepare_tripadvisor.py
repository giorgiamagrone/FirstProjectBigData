import pandas as pd
import ast

INPUT_TRIPADVISOR = "tripadvisor_hotel_reviews.csv"
OUTPUT_OOD = "out_of_domain_test.csv"

MIN_TEXT_LEN = 10
VALID_RATINGS = {1, 2, 3, 4, 5}
SAMPLE_SIZE = 3000
RANDOM_STATE = 42


def extract_overall_rating(rating_str):
    try:
        rating_dict = ast.literal_eval(str(rating_str))
        return float(rating_dict.get('overall', 0.0))
    except Exception:
        return None


def main():
    print("=" * 60)
    print("Preparazione dataset TripAdvisor (con controlli di sanità)")
    print("=" * 60)

    df = pd.read_csv(INPUT_TRIPADVISOR)
    n_raw = len(df)
    print(f"\n[1/5] Righe grezze caricate: {n_raw:,}")

    #  Estrazione rating e testo 
    df['rating'] = df['ratings'].apply(extract_overall_rating)
    testo_colonna = 'text' if 'text' in df.columns else 'title'
    df['text'] = df[testo_colonna]

    # Controllo 1: valori mancanti 
    df = df[['text', 'rating']].dropna()
    n_after_na = len(df)
    print(f"[2/5] Dopo rimozione valori mancanti: {n_after_na:,} "
          f"({n_raw - n_after_na:,} scartati)")

    # Controllo 2: testo strutturalmente valido 
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() >= MIN_TEXT_LEN]
    n_after_len = len(df)
    print(f"[3/5] Dopo rimozione testi < {MIN_TEXT_LEN} caratteri: {n_after_len:,} "
          f"({n_after_na - n_after_len:,} scartati)")

    # Controllo 3: duplicati esatti 
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text'])
    n_after_dedup = len(df)
    print(f"[4/5] Dopo rimozione duplicati esatti: {n_after_dedup:,} "
          f"({before_dedup - n_after_dedup:,} scartati)")

    #  Controllo 4: rating strutturalmente valido (intero 1-5) 
    before_rating = len(df)
    df['rating_rounded'] = df['rating'].round()
    df = df[
        df['rating_rounded'].isin(VALID_RATINGS) &
        (df['rating'] - df['rating_rounded']).abs().lt(1e-9)
    ]
    df['rating'] = df['rating_rounded'].astype(float)
    df = df.drop(columns=['rating_rounded'])
    n_after_rating = len(df)
    print(f"[5/5] Dopo validazione rating (intero 1-5): {n_after_rating:,} "
          f"({before_rating - n_after_rating:,} scartati, es. rating 0.5 o 3.7)")

    # Campionamento finale 
    if len(df) < SAMPLE_SIZE:
        print(f"\n⚠️  Attenzione: solo {len(df):,} campioni disponibili dopo i controlli, "
              f"sotto il target di {SAMPLE_SIZE}. Uso tutti quelli disponibili.")
        df_sample = df
    else:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    df_sample.to_csv(OUTPUT_OOD, index=False)

    print(f"\n Completato.")
    print(f"   Righe grezze iniziali : {n_raw:,}")
    print(f"   Righe finali (test)   : {len(df_sample):,}")
    print(f"   Scartate in totale    : {n_raw - len(df_sample):,} "
          f"({100*(n_raw - len(df_sample))/n_raw:.1f}%)")
    print(f"   Salvato: '{OUTPUT_OOD}'\n")
    print("   Distribuzione rating nel campione finale:")
    print(df_sample['rating'].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()