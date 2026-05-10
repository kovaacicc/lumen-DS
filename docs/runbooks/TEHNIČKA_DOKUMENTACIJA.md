# Vodič za pokretanje — Lumen DS: Sustav preporuka za cross-sell i up-sell

**Upute za korištenje — kako pokrenuti kod i dobiti identične rezultate.**

---

## 1. Preduvjeti

| Zahtjev | Verzija |
|---|---|
| Python | 3.10 |
| Conda | bilo koja novija verzija |

Sve ostale ovisnosti instaliraju se putem postavljanja okoline opisanog u nastavku.

---

## 2. Struktura projekta

```
lumen-DS/
├── api/
│   ├── main.py                  ← REST API (FastAPI)
│   └── requirements.txt         ← ovisnosti API-ja
├── dataprep/
│   ├── fixer.py                 ← skripta za pripremu podataka (pokreće se prije bilježnica)
│   └── cleaner.py               ← skripta za pripremu podataka (pokreće se prije bilježnica)
├── data/
│   ├── raw/
│   │   └── LUMEN_DS.csv         ← originalni skup podataka (nije u gitu, obavezno)
│   └── processed/               ← generira se pokretanjem bilježnica
├── models/                      ← istrenirani modeli (generiraju se pokretanjem bilježnica)
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── customer_segmentation.ipynb
│   ├── encoding_and_feature_engineering.ipynb
│   ├── market_basket.ipynb
│   ├── cross_sell_recommender.ipynb
│   ├── upsell_model.ipynb
│   └── cross_sell_model.ipynb   ← dodatna bilježnica, nije potrebna za API
├── visualization/
└── docs/
    └── runbooks/
        ├── RUN_BOOK.md              ← engleski vodič za pokretanje
        └── TEHNIČKA_DOKUMENTACIJA.md ← ova datoteka (hrvatski)
```

---

## 3. Postavljanje okoline

### 3.1 Kreiranje i aktivacija conda okoline

```bash
conda create -n lumen-ds python=3.10 -y
conda activate lumen-ds
```

### 3.2 Instalacija ovisnosti za bilježnice

```bash
pip install pandas==2.3.3 numpy==2.2.6 scikit-learn==1.7.2 xgboost==3.2.0 \
    imbalanced-learn==0.14.1 mlxtend==0.23.4 matplotlib seaborn \
    jupyter ipykernel joblib
```

### 3.3 Instalacija ovisnosti za API

```bash
pip install -r api/requirements.txt
```

### 3.4 Registracija kernela u Jupyteru

```bash
python -m ipykernel install --user --name lumen-ds --display-name "lumen-ds"
```

---

## 4. Postavljanje podataka

Originalni skup podataka postavite u direktorij `data/raw/`:

```
data/raw/LUMEN_DS.csv
```

Ova datoteka nije uključena u repozitorij zbog veličine (~795 MB). Sve ostale datoteke u `data/processed/` i `models/` ili su commitane u repozitorij ili se generiraju pokretanjem bilježnica.

---

## 5. Pokretanje bilježnica

**Pokrenite bilježnice sljedećim redoslijedom.** Svaka bilježnica čita datoteke koje je prethodna proizvela. Ne preskačite korake.

Pokrenite Jupyter iz korijenskog direktorija projekta:

```bash
conda activate lumen-ds
jupyter notebook
```

Prije pokretanja svake bilježnice provjerite je li kernel postavljen na **lumen-ds**.

---

### Korak 1 — `notebooks/data_cleaning.ipynb`

**Čita:** `data/raw/LUMEN_DS.csv`

**Proizvodi:** `data/processed/data_clean_final.csv`

**Što radi:** Učitava originalni skup podataka, uklanja nevaljane retke (negativne cijene, nulte količine, interkompanije, placeholder kupce i artikle), imputira nedostajuće vrijednosti i sprema čisti skup podataka. Ova datoteka je ulaz za sve sljedeće bilježnice.

---

### Korak 2 — `notebooks/customer_segmentation.ipynb`

**Čita:** `data/processed/data_clean_final.csv`

**Proizvodi:** `data/processed/rfm_segments.csv`

**Što radi:** Izračunava RFM (Recency, Frequency, Monetary) bodove za sve kupce s referentnim datumom 2020-07-01, zatim primjenjuje KMeans clustering (K=4) kako bi dobio četiri segmenta kupaca: Hibernating, Mid-Tier At Risk, Active Small Buyers i Whales/Champions. Oznake klastera koriste se kao značajke u modelu za cross-sell i za personalizirano rangiranje artikala u oba endpointa.

---

### Korak 3 — `notebooks/encoding_and_feature_engineering.ipynb`

**Čita:** `data/processed/data_clean_final.csv`

**Proizvodi:** `data/processed/data_clean_enc_fe.csv`, `data/processed/customer_features.csv`, `data/processed/invoice_items.csv`

**Što radi:** Label-enkodira i one-hot enkodira kategoričke varijable, izrađuje vremenski zasnovane značajke, izračunava agregate na razini kupca i analizira obrasce zajedničkog pojavljivanja faktura. Izlazne datoteke ove bilježnice ulaz su za dopunsku bilježnicu `cross_sell_model.ipynb`.

> **Napomena:** Izlazne datoteke ove bilježnice **nisu** potrebne za glavni API pipeline. Koraci 4–6 čitaju izravno iz `data_clean_final.csv`.

---

### Korak 4 — `notebooks/market_basket.ipynb`

**Čita:** `data/processed/data_clean_final.csv`

**Proizvodi:** `data/processed/association_rules.csv`, `data/processed/basket_matrix.csv`

**Što radi:** Priprema košare kupac × kalendarski mjesec (košare na razini fakture su prerijetke — 1,8% višeproduktnih), pokreće Apriori i FP-Growth algoritme te filtrira pravila po lift > 1. Dobivenih 14 asocijacijskih pravila koriste se kao značajke u modelu za cross-sell.

---

### Korak 5 — `notebooks/cross_sell_recommender.ipynb`

**Čita:** `data/processed/data_clean_final.csv`, `data/processed/rfm_segments.csv`, `data/processed/association_rules.csv`

**Proizvodi:**
- `models/recommender/recommender_xgb.pkl` — istrenirani XGBoost model
- `models/recommender/model_meta.json` — metapodaci modela i AUC rezultati
- `data/processed/recommender/customer_feats.csv` — predračunate značajke za 9.017 poznatih kupaca
- `data/processed/recommender/customer_purchased_pgs.csv` — povijest kupnje po kupcu
- `data/processed/recommender/product_feats.csv` — značajke za svih 30 produktnih grupa
- `data/processed/recommender/copurchase_rates.csv` — stope zajedničke kupnje PG-para
- `data/processed/recommender/feature_medians.json` — populacijski medijani za nepoznate kupce
- `data/processed/recommender/customer_recommendations.csv` — predračunate top-5 preporuke za sve kupce

**Što radi:** Gradi matricu Kupac × Produktna grupa (~257k redaka, 0,34% pozitivnih), izrađuje 27 značajki po retku, trenira i evaluira tri modela (Logistička regresija, Random Forest, XGBoost) s GroupKFold unakrsnom validacijom i SMOTE oversamplingom, zatim generira cross-sell preporuke za sve kupce. XGBoost model postiže ROC-AUC od **0,9248**.

---

### Korak 6 — `notebooks/upsell_model.ipynb`

**Čita:** `data/processed/data_clean_final.csv`, `data/processed/rfm_segments.csv`

**Proizvodi:**
- `data/processed/upsell/item_tiers.csv` — klasifikacija standard/premium za sve artikle
- `data/processed/upsell/pg_median_prices.csv` — medijalna cijena po produktnoj grupi
- `data/processed/upsell/upsell_item_popularity.csv` — rangiranje premium artikala po klasteru
- `data/processed/upsell/upsell_item_popularity_all.csv` — rangiranje svih artikala po klasteru (za Layer 2 cross-sella)
- `data/processed/upsell/upsell_recommendations.csv` — predračunate up-sell preporuke za sve kupce
- `models/upsell/upsell_meta.json` — sažetak statistika

**Što radi:** Klasificira svaki artikl kao standard ili premium (na temelju toga je li njegova prosječna cijena iznad ili ispod medijana produktne grupe), identificira kupce koji kupuju ispod medijana u svakoj PG i gradi rangiranja popularnosti artikala uvjetovana klasterom. Validirano proxy metricom: 44,2% up-sell ciljeva pokazalo je pozitivan rast cijene u label periodu.

---

### Dopunska bilježnica — `notebooks/cross_sell_model.ipynb` *(nije obavezna)*

**Čita:** `data/processed/customer_features.csv` (iz Koraka 3)

**Proizvodi:** `models/cross_sell/cross_sell_rf.pkl`, `models/cross_sell/cross_sell_xgb.pkl`

**Što radi:** Raniji, jednostavniji pristup cross-sellu koji klasificira kupce kao cross-sell kandidate na razini kupca (umjesto na razini kupac × produktna grupa). API ne koristi ovu bilježnicu. Uključena radi cjelovitosti prikaza procesa modeliranja.

---

## 6. Pokretanje API-ja

Iz korijenskog direktorija projekta s aktivnom okolinom `lumen-ds`:

```bash
conda activate lumen-ds
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Očekivani izlaz pri pokretanju:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

API učitava sve modele i tablice za pretragu u memoriju pri pokretanju (~3–5 sekundi). Nakon pokretanja, svi zahtjevi se poslužuju iz memorije bez čitanja s diska.

**Interaktivna dokumentacija** (Swagger UI) dostupna je na:
```
http://localhost:8000/docs
```

Pruža sučelje za testiranje oba endpointa bez pisanja koda.

---

## 7. API Endpointi

### `GET /health`

Potvrđuje da server radi i da je model učitan.

**Zahtjev:**
```bash
curl http://localhost:8000/health
```

**Odgovor:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "known_customers": 9017,
  "layer2_available": true
}
```

---

### `POST /cross-sell`

Vraća do 5 preporuka produktnih grupa za kupca, poredanih po pouzdanosti. Svaka preporučena produktna grupa uključuje do 5 prijedloga konkretnih artikala.

**Ulazna polja:**
| Polje | Tip | Obavezno | Opis |
|---|---|---|---|
| `customer_id` | cijeli broj | Ne | ID postojećeg kupca. Ako se izostavi ili je nepoznat, koriste se populacijski prosjeci. |
| `item_codes` | lista stringova | Ne | Kodovi artikala koje kupac trenutno naručuje. Koriste se za isključivanje aktivnih PG i računanje signala zajedničke kupnje. |

**Primjer — postojeći kupac s trenutnom narudžbom:**
```bash
curl -X POST http://localhost:8000/cross-sell \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 12345, "item_codes": ["03650", "2158"]}'
```

**Primjer — novi kupac (bez povijesti):**
```bash
curl -X POST http://localhost:8000/cross-sell \
  -H "Content-Type: application/json" \
  -d '{"item_codes": ["03650", "2158"]}'
```

**Odgovor:**
```json
{
  "customer_id": 12345,
  "recommendations": [
    {
      "product_group": "PC016",
      "confidence": 0.4642,
      "items": [
        {"item_code": "BOE001-TY8.375", "confidence": 0.1098, "avg_price": 6.02},
        {"item_code": "BOE001-348.5",   "confidence": 0.0854, "avg_price": 11.0}
      ]
    }
  ]
}
```

`confidence` na razini produktne grupe je XGBoost vjerojatnost (0–1) da će kupac kupiti iz te grupe. `confidence` na razini artikla je udio sličnih kupaca iz istog klastera koji su kupili taj artikl.

---

### `POST /up-sell`

Za svaki dostavljeni kod artikla provjerava je li artikl standardne kategorije i vraća premium alternative iz iste produktne grupe, personalizirane prema klasteru kupca.

**Ulazna polja:**
| Polje | Tip | Obavezno | Opis |
|---|---|---|---|
| `customer_id` | cijeli broj | Ne | ID postojećeg kupca. Ako se izostavi, personalizacija po klasteru se preskače. |
| `item_codes` | lista stringova | Da | Kodovi artikala za koje se traže up-sell prilike. |

**Primjer:**
```bash
curl -X POST http://localhost:8000/up-sell \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 12345, "item_codes": ["03650", "03673"]}'
```

**Odgovor:**
```json
{
  "customer_id": 12345,
  "recommendations": [
    {
      "current_item": "03650",
      "product_group": "PC001",
      "current_tier": "standard",
      "current_avg_price": 0.0,
      "premium_alternatives": [
        {"item_code": "2158",          "confidence": 0.0495, "avg_price": 78.99},
        {"item_code": "BOE001-30517",  "confidence": 0.0155, "avg_price": 220.44}
      ]
    }
  ]
}
```

---

## 8. Reproducibilnost rezultata

Sljedeće postavke su fiksirane u svim bilježnicama kako bi se osigurala reproducibilnost:

| Postavka | Vrijednost |
|---|---|
| Vremenski split trening/test | Trening: prije 2019-07-01 / Label: 2019-07-01 do 2020-07-01 |
| Nasumični seed | `random_state=42` na svim modelima i SMOTE-u |
| SMOTE strategija uzorkovanja | 0,1 (10% pozitivnih u treningu) |
| Unakrsna validacija | GroupKFold, 5 foldova, grupirano po CustomerID |
| Top-K preporuke | 5 |
| RFM referentni datum | 2020-07-01 |
| KMeans klasteri | K=4 |

> **Napomena o manjim varijacijama AUC-a:** XGBoost i unakrsna validacija koriste `n_jobs=-1` (paralelna obrada). Operacije s pomičnim zarezom u paraleli su nedeterminističke, što uzrokuje varijacije AUC-a od ±0,002 između pokretanja. To ne utječe na kvalitetu modela ni preporuke.

---

## 9. Rješavanje problema

**`FileNotFoundError: data_clean_final.csv`**
Pokrenite prvo `data_cleaning.ipynb`. Provjerite da se `LUMEN_DS.csv` nalazi u `data/raw/`.

**`FileNotFoundError: recommender_xgb.pkl`**
Pokrenite `cross_sell_recommender.ipynb` prije pokretanja API-ja.

**`FileNotFoundError: association_rules.csv`**
Pokrenite `market_basket.ipynb` prije `cross_sell_recommender.ipynb`.

**API vraća `"known_customers": 0`**
Datoteka `customer_feats.csv` nedostaje. Ponovno pokrenite `cross_sell_recommender.ipynb` i provjerite da se ćelija koja sprema inference artifakte izvršila bez greške.

**API vraća `"layer2_available": false`**
Datoteka `upsell_item_popularity_all.csv` nedostaje. Ponovno pokrenite `upsell_model.ipynb`.

**Port je već zauzet**
Promijenite port: `uvicorn api.main:app --port 8001`
