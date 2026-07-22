# Data Quality in Data-Centric AI

## Dataset

I dati grezzi provengono dal dataset **Amazon Reviews 2023** (categoria
*CDs and Vinyl*), disponibile su Hugging Face:
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Il dataset out-of-domain per il test di generalizzazione (Step 07) è
**TripAdvisor Hotel Reviews**, disponibile su Kaggle:
https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews


## Pipeline

La pipeline è organizzata in 7 step sequenziali, ciascuno con file di
input/output intermedi per tracciabilità:

| Step | Script | Descrizione |
|---|---|---|
| 01 | `01_ingestion.py` | Ingestion da HDFS con PySpark, filtri strutturali, campionamento 10% |
| 02 | `02_profiling_sanitization.py` | Profiling statistico e deduplicazione |
| 03 | `03_noisy_label_detection.py` | Detection etichette rumorose con Cleanlab + bilanciamento classi |
| 04 | `04_llm_curation.py` | Correzione mirata delle etichette con Llama 3 (via Ollama) |
| 05 | `05_dataset_assembly.py` | Assemblaggio del dataset finale con metadati di provenance |
| 06 | `06_comparative_evaluation.py` | Valutazione comparativa: 4 esperimenti, stesso test set fisso |
| 07 | `07_cross_domain_test.py` | Test di generalizzazione su dominio diverso (TripAdvisor) |

Script di supporto: `prepare_tripadvisor.py` (pulizia strutturale del
dataset out-of-domain, va eseguito prima di `07_cross_domain_test.py`).

Prerequisiti: Python 3.x, Apache Spark, HDFS, Ollama con il modello `llama3` scaricato
e in esecuzione.

Il dataset grezzo (`CDs_and_Vinyl.jsonl`, Amazon Reviews 2023) va
caricato su HDFS prima di eseguire `01_ingestion.py`. Il dataset
TripAdvisor (`tripadvisor_hotel_reviews.csv`) va scaricato a parte e
posizionato nella cartella prima di eseguire `prepare_tripadvisor.py`.

```bash
pip install -r requirements.txt

python 01_ingestion.py
python 02_profiling_sanitization.py
python 03_noisy_label_detection.py
python 04_llm_curation.py
python 05_dataset_assembly.py
python 06_comparative_evaluation.py
python prepare_tripadvisor.py
python 07_cross_domain_test.py
```

Ogni script controlla l'esistenza dei file di input dello step
precedente e si interrompe con un errore chiaro se mancano.
