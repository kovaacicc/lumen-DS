# Projektna Dokumentacija — Lumen DS Cross-Sell & Up-Sell Sustav za Preporuke

> **Metodologija:** CRISP-DM (Cross-Industry Standard Process for Data Mining)
>
> Ovaj dokument opisuje cjelokupni proces data science-a u okviru natjecateljskog projekta Lumen DS — od razumijevanja poslovnog problema do implementacije (deployment) API-ja za preporuke u stvarnom vremenu. Nadopunjuje operativne dokumente README.md i RUN_BOOK.md, koji se fokusiraju na *kako pokrenuti* pipeline; ovdje dokumentiramo *što smo izgradili i zašto*.

---

## Sadržaj

1. [Razumijevanje poslovanja](#1-razumijevanje-poslovanja)
2. [Razumijevanje podataka](#2-razumijevanje-podataka)
3. [Priprema podataka](#3-priprema-podataka)
4. [Modeliranje](#4-modeliranje)
5. [Evaluacija](#5-evaluacija)
6. [Implementacija (Deployment)](#6-implementacija-deployment)

---

## 1. Razumijevanje poslovanja

### 1.1 Zašto su Cross-Sell i Up-Sell važni

Monetizacija baze kupaca jedna je od aktivnosti s najvećim polugom u B2B distribuciji. Istraživanja industrije dosljedno pokazuju da:

- **Akvizicija novog kupca košta 5–25× više od zadržavanja postojećeg.** Postojeći kupci već vjeruju brendu, integrirani su u procese nabave i imaju račun u ERP sustavu — što novi potencijalni kupac ne posjeduje.
- **Vjerojatnost prodaje postojećem kupcu iznosi 60–70%, naspram 5–20% za novog prospekta.** Čak i skromno poboljšanje stope uspješnosti cross-sell-a kumulativno se pretvara u značajan rast prihoda.
- **Up-selling postojećih kupaca prema proizvodima višeg razreda povećava prosječnu vrijednost narudžbe bez povećanja troškova akvizicije.** To izravno poboljšava bruto maržu po odnosu s kupcem.

Za B2B distributera s ~12.000 aktivnih računa, pitanje nije *treba li* investirati u cross-sell i up-sell, već *koji kupac treba dobiti koji proizvod i kada*.

### 1.2 Cilj projekta

Izgraditi sustav za preporuke temeljen na podacima koji:

1. **Segmentira kupce** u bihevioralne grupe (buyer persone) kako bi se preporuke mogle personalizirati za svaki segment.
2. **Otkriva asocijacije između proizvoda** u povijesnim transakcijskim podacima kako bi se identificiralo koji se proizvodi često kupuju zajedno.
3. **Poslužuje preporuke u stvarnom vremenu** putem REST API-ja koji tim prodaje ili e-commerce front-end može pozvati kada je kupac na telefonu ili na naplatnom mjestu. Potrebne su dvije endpoint-točke:
   - `POST /cross-sell` — za zadanog kupca (i opcionalno proizvode koji se trenutno naručuju), preporuči dodatne grupe proizvoda koje vjerojatno želi kupiti.
   - `POST /up-sell` — za specifične proizvode koje kupac kupuje, preporuči alternative višeg razreda unutar iste grupe proizvoda.

### 1.3 Kriteriji uspjeha

| Dimenzija | Cilj | Razlog |
|---|---|---|
| Kvaliteta segmentacije | Silhouette score interpretabilan, segmenti bihevioralno distinktni | Tim prodaje mora moći djelovati na temelju segmenata |
| Rangiranje cross-sell-a | ROC-AUC ≥ 0.85 bez vremenskog leakage-a (curenja informacija) | Kvaliteta rangiranja važnija je od apsolutne preciznosti u uvjetima neravnoteže 290:1 |
| Up-sell signal | Pozitivan dokaz rasta cijene u hold-out validacijskom prozoru | Ne možemo provesti A/B test, pa je proxy temeljen na naprednom vremenu najbolji raspoloživi dokaz |
| API latencija | Ispod sekunde po pozivu, svi artefakti u memoriji | API se poziva tijekom aktivnih prodajnih razgovora |
| Reproducibilnost | Fiksni seedovi, dokumentiran vremenski raspon | Ponovni pokret mora producirati identične artefakte |

### 1.4 Ograničenja i pretpostavke

- **Podaci su jedini izvor istine.** Nije dostupan domenski stručnjak koji bi označio "ovo je cross-sell par" — asocijacije se moraju ekstrahirati iz povijesti transakcija.
- **Dataset je B2B.** To je empirijski utvrđeno u §2.4 i oblikuje svaki modelarski izbor (košarica na razini kupca, rangiranja uvjetovana segmentom, fokus na ključne račune).
- **Nije dostupna A/B test infrastruktura.** Validacija se mora oslanjati na vremenski hold-out (treniraj na prošlosti, evaluiraj na budućnosti), a ne na eksperimentalne grupe tretmana i kontrole.
- **API mora biti jednostavan.** Prema projektnom zadatku: dvije endpoint-točke koje primaju `CustomerID` i listu proizvoda, a vraćaju rangirane preporuke s ocjenama pouzdanosti.

---

## 2. Razumijevanje podataka

### 2.1 Izvorni dataset

| Svojstvo | Vrijednost |
|---|---|
| Naziv datoteke | `LUMEN_DS.csv` |
| Kodiranje | UTF-16, pipe-odvojen, ~795 MB |
| Redovi | ~1.300.000 stavki faktura |
| Stupci | 32 sirova polja koja pokrivaju dimenzije kupca, proizvoda, transakcije, financija, prodajnog kanala i proizvodnje |
| Granularnost | Jedan red po stavci fakture (jedan proizvod na jednoj fakturi) |

### 2.2 Stupci korišteni u pipelineu

Sirovi dataset ima 32 stupca, ali modelarski pipeline koristi samo fokusirani podskup. Tablica u nastavku navodi svaki stupac koji je stvarno pročitan u barem jednoj bilježnici, zajedno s ulogom koju ima. Stupci koji nisu navedeni ne koriste se ni u jednom modelu — sažeti su u §2.3.

> *"Gdje se koristi" odnosi se na produkcijski pipeline: `01 data_cleaning`, `02 customer_segmentation`, `03 market_basket`, `04 cross_sell_recommender`, `05 upsell_model`. Dvije bilježnice u mapi `notebooks/archive/` su istraživačke i nisu dio API puta — vidi §3.7 i §4.3.1.*

| Stupac | Gdje se koristi | Uloga u projektu |
|---|---|---|
| `CustomerID` | Sve bilježnice | Primarni ključ za svaku analizu na razini kupca (RFM, segmentacija, grupiranje košarica, preporuke). |
| `Invoice Date` | Sve bilježnice | Recency u RFM-u (segmentacija), ključ košarice kupac-mjesec (market basket), granična vrijednost raspodjele trening/label (cross-sell, up-sell), izračun trajnosti (cross-sell). |
| `Invoice #` | Segmentacija, market basket, cross-sell | Frequency u RFM-u; jedinica ko-pojavljivanja košarice na razini fakture (korištenjem za potvrdu da su košarice na razini fakture prerijetke i prelazak na kupac-mjesec); značajka `train_orders` za cross-sell model. |
| `Invoiced price` | Čišćenje, segmentacija, cross-sell, up-sell | Monetary u RFM-u (s količinom); imputira se gdje ≥ 50% redova nekog proizvoda ima valjane cijene; prosječna cijena proizvoda za klasifikaciju razreda (up-sell); značajka `pg_avg_price` (cross-sell). |
| `Invoiced qty (shipped)` | Čišćenje, segmentacija, cross-sell | Množi se s `Invoiced price` za izračun `line_revenue` (Monetary, `train_revenue`, `pg_avg_revenue`). Negativni ili nulti redovi se filtriraju pri čišćenju. |
| `Item Code` | Čišćenje, market basket (bonus pokretanje na razini stavke), cross-sell, up-sell | Identifikator proizvoda; značajka `train_items`; ciljna varijabla za klasifikaciju razreda; razina preporuke stavke Level-2 unutar API-ja. Kodovi `SAMPLE` i oni s prefiksom `/` se filtriraju pri čišćenju. |
| `Product group` | Market basket, cross-sell, up-sell | Granularnost preporuke za cross-sell ranker; jedinica ko-pojavljivanja na razini košarice; medijan cijene na PG razini za up-sell određivanje razreda. |
| `GM%` | Čišćenje, cross-sell | Ponovo se izračunava iz cijene i troška gdje su oba dostupna; winsorize (ograničavanje ekstremnih vrijednosti) na `[-5, 1]`; značajke `train_avg_gm` i `pg_avg_gm`. |
| `Customer Region` | Cross-sell | Label-enkodiran → značajka `region_enc`. |
| `Customer industry` | Cross-sell | Label-enkodiran → značajka `industry_enc`. |
| `Top Customer Group` | Cross-sell | Label-enkodiran → značajka `top_group_enc`. |
| `Customer First Invoice Date` | Cross-sell | Pokreće značajku `tenure_days` (datum snapshota − prva faktura). |
| `Cost of part` | Čišćenje | Koristi se za oporavak `GM%` gdje je null; za filtriranje "smeće" redova gdje `Cost > 20 × Invoiced price`. Ne koristi se izravno kao značajka. |
| `Intercompany` | Čišćenje | Filter pre-modeliranja: redovi s `Intercompany = YES` (intrakorporativni prijenosi) uklanjaju se prije bilo koje analize. |

Uz to, izvedeni stupac `Cluster_name` (izlaz segmentacije) hrani oba nizvodna preporučitelja: kao značajka `cluster_enc` u cross-sell modelu, i kao ključ personalizacije za popularnost stavki uvjetovanu klasterom u up-sell modelu.

### 2.3 Nekorišteni stupci

Svaki drugi stupac u sirovom datasetu ispušta se rano pri čišćenju ili se jednostavno nikad ne čita. Tablica u nastavku bilježi što je isključeno i zašto — čuvajući transparentnost revizije umjesto impliciranja da je svako polje bilo obrađeno.

| Stupac | Razlog nekorištenja |
|---|---|
| `Invoice Line #` | Metapodatak pozicije unutar fakture; irelevantan nakon agregacije na razini fakture ili kupac-mjesec; gotovo jednako jedinstvenih vrijednosti kao i redova. |
| `Order #`, `Order Date`, `Order Line #` | Artefakti strane narudžbe PO ciklusa; `Invoice Date` je kanonska vremenska oznaka za RFM i vremensku raspodjelu. |
| `Ordered qty` | Koristi se samo kao provjera zdravog razuma pri čišćenju; kanonski signal "kupac je ovo kupio" jest `Invoiced qty (shipped)`. |
| `Material cost`, `Labor cost`, `Overhead cost` | Komponente `Cost of part`; agregat je dovoljan za naša pravila čišćenja. |
| `Invoiced price (TX)` | Semantika polja je nejasna (vjerojatno porezna varijanta); isključena kako bi se izbjeglo uvođenje šumnih značajki. |
| `Sales Channel - Internal`, `Sales Channel - External`, `Sales Channel - Grouping` | Metapodaci atribucije kanala; nisu značajka na razini kupca ili proizvoda relevantna za "što bi ovaj kupac sljedeće trebao kupiti." |
| `Make vs Buy` | Atribut s ponudbene strane (izrađeno interno ili nabavljeno); nije prediktivan za cross-sell ponašanje. |
| `Manufacturing Region`, `Manufacturing Location Code` | Gdje se stavka proizvodi — ponudbena, ne kupčeva strana, pa irelevantno za cross-sell i up-sell. |
| `Product family` | Zamijenjeno s `Product group`, koji je odabrana granularnost košarice i preporuke. |
| `Born on date`, `Price last modified date in the ERP` | Metapodaci životnog ciklusa stavke / ažuriranja cijene; nema očekivanog signala za cross-sell ili up-sell. |
| `# of unique products on a quote` | Metrika na razini ponude, nije usklađena s granularnošću stavke fakture u datasetu. |

### 2.4 Utvrđivanje B2B prirode dataseta

Zadatak ne označava dataset eksplicitno, ali svaki strukturni i statistički signal upućuje na **B2B proizvodnju ili industrijsku distribuciju**:

**Strukturni signali (shema):**
- Zastavica `Intercompany` — ima smisla samo kada prodaja može biti intrakorporativna.
- `Customer industry` (npr. `IC000`, `IC001`) — B2C kupci nemaju "industriju."
- `Make vs Buy`, `Cost of part`, `Manufacturing Location Code` — rječnik proizvodnje.
- `Top Customer Group`, `Customer First Invoice Date` — terminologija upravljanja ključnim računima.
- Odvojenost `Ordered qty` i `Invoiced qty (shipped)` — klasičan B2B PO ciklus s djelomičnim isporukama.
- 6-znamenkasti numerički `CustomerID` — tipično za SAP/Oracle/JDE matičnu datoteku kupaca.

**Statistički signali (nakon filtriranja Intercompany=YES):**
- **11.775 jedinstvenih kupaca** — reda tisuća, ne milijuna.
- **Medijan prihoda kupca ≈ \$6.737, P90 ≈ \$211.738, maksimum ≈ \$1,15B.** Omjer max/medijan od ~171.000× nemoguć je u B2C kontekstu.
- **Medijan 5 stavki fakture po kupcu, P90 ≈ 87 stavki** — bimodalna distribucija: dugi rep malih povremenih kupaca + manjina velikih korporativnih računa s tisućama stavki godišnje.

Ovaj zaključak pokreće nekoliko modelarskih odluka: košarice na razini kupca (ne fakture), segmentacija RFM-om koji je osjetljiv na "kitove" (whale kupce), i rangiranja uvjetovana klasterom.

### 2.5 Početni nalazi o kvaliteti podataka

Istraživačka analiza identificirala je šest klasa problematičnih podataka koje su morale biti riješene prije modeliranja:

1. **Privremeni `CustomerID` `-99`.** NULL marker izvornog sustava, nije pravi kupac.
2. **Kodovi stavki s prefiksom `SAMPLE` i `/`.** Tehnički zapisi i besplatni uzorci — nisu stvarne kupovine.
3. **Nulta / null `Invoiced price`.** ~9% redova. Mješavina tehničkih artefakata i oporavljenih vrijednosti.
4. **Negativna `Invoiced price` i negativne količine.** Kreditne note / povrati — mali po volumenu, ali iskrivljuju RFM.
5. **`Cost of part` ≫ `Invoiced price`.** Nekoliko redova gdje je trošak bio 20× viši od cijene — jasno smeće.
6. **Outlieri u `GM%`.** Repne vrijednosti do −1000+ i nekolicina iznad 1.0 — potrebno winsorize (ograničavanje ekstremnih vrijednosti).

Kodovi uzoraka/slash stavki samo su iznosili stotine tisuća redova; `Intercompany=YES` uklonio je još jedan veliki blok. Oni su kvantificirani u §3.

---

## 3. Priprema podataka

Pipeline čišćenja izvodi se u potpunosti u `notebooks/data_cleaning.ipynb`. Svaka odluka donesena je namjerno, s obrazloženjem navedenim u nastavku.

### 3.1 Popravak kodiranja i početno ispuštanje stupaca

Prvi korak bilježnice pretvara sirovu `LUMEN_DS.csv` iz UTF-16 → UTF-8 i uklanja ugrađene dvostruke navodnike koji bi inače slomili parser za pipe format. Rezultat se sprema u cache kao `data/raw/data.csv` kako bi se konverzija preskočila pri sljedećim pokretanjima. Zatim se ispušta 11 stupaca jer su redundantni ili previše granularni za bilo koji od planiranih modela:

| Ispušteni stupac | Razlog |
|---|---|
| `Invoice Line #` | Redoslijed unutar fakture — irelevantan za preporuke |
| `Order Date`, `Order Line #`, `Order #` | `Invoice Date` je kanonska vremenska oznaka |
| `Material cost`, `Labor cost`, `Overhead cost` | Agregirani u `Cost of part` |
| `Sales Channel - Grouping` | Redundantno s `Sales Channel - Internal/External` |
| `Born on date`, `Price last modified date in the ERP` | Metapodaci životnog ciklusa stavke, ne prediktivno za cross-sell |
| `# of unique products on a quote` | Metrika na razini ponude, ne fakture |

### 3.2 Filter za Intercompany

Redovi s `Intercompany=YES` predstavljaju intrakorporativne prijenose, a ne stvarne transakcije kupaca. Filtrirani su prije bilo koje druge analize kako bi se izbjeglo zagađivanje RFM i statistike košarica.

### 3.3 Odluke o čišćenju na razini reda

#### 3.3.1 Ispuštanje privremenog `CustomerID` `-99`
NULL marker izvornog sustava. Uklanjanje ovih redova je nedvosmisleno.

#### 3.3.2 Ispuštanje kodova stavki s prefiksom `SAMPLE` i `/`
Ovo su besplatni uzorci (nisu tražene kupovine) i tehnički zapisi izvornog sustava. Pojavljuju se na stotinama tisuća stavki i stvorili bi lažne signale ko-kupovine. Filter:
```
Item Code ~ "SAMPLE" or Item Code starts with "/"
```

#### 3.3.3 Obrada nulte / null `Invoiced price`

Otprilike 9% redova imalo je nultu ili null cijenu. Dvije opcije: ispustiti ili imputirati. Odabrali smo strategiju **imputacije temeljene na pragu**:

- Za svaki `Item Code`, izračunaj udio njegovih redova koji imaju pozitivnu cijenu.
- Ako ≥ 50% redova nekog proizvoda ima valjanu cijenu (odabrano analizom osjetljivosti na pragovima 10/30/50/70/90%), imputa se null/nulte redove s **medijanom pozitivnih cijena za taj proizvod**.
- Ako < 50% redova nekog proizvoda ima valjanu cijenu, proizvod je previše nepouzdan za imputaciju — ispusti te redove.

Ovo balansira zadržavanje podataka naspram kontaminacije. Prag od 50% čuva većinu oporavljenih redova isključujući pritom stavke gdje je polje cijena uglavnom pokvareno.

#### 3.3.4 Ispuštanje negativne `Invoiced price`
Redovi s negativnom zaračunatom cijenom su kreditne note (povrati / refundacije). Uključivanje ih kao "kupovine" iskrivilo bi Monetary i signale košarica.

#### 3.3.5 Ispuštanje negativnih količina; izdvajanje redova s nultom isporukom
- Negativni `Invoiced qty (shipped)` i negativni `Ordered qty` su povrati — ispuštaju se.
- Redovi s `Invoiced qty (shipped) == 0` izdvojeni su u sporedni dataframe (namjera-ali-nije-isporučeno) ali se ne koriste nizvodnje jer je kanonski signal "kupac je ovo kupio" isporučena količina veća od nule.

#### 3.3.6 Ispuštanje "smeće" redova troška
Redovi gdje je `Cost of part > 20 × Invoiced price` su greške unosa podataka (npr. pogrešna konverzija jedinice). Iskrivljuju `GM%` prema ekstremno negativnim vrijednostima. Filter je konzervativan — sve što je 20× cijene je jasno pokvareno; legitimne stavke s niskom maržom nisu pogođene.

#### 3.3.7 Winsorize `GM%` na `[-5, 1]`
Nakon popravka smeće redova troška vs. cijene, rezidualna distribucija `GM%` još uvijek ima dugi lijevi rep i nekoliko vrijednosti iznad 1.0 (rubni slučajevi gdje je cijena = 0 koji su izbjegli ranije filtere). Isjecanje na `[-5, 1]` zadržava smjer signala eliminiranjem šuma.

### 3.4 Imputacija kategorijalnih null vrijednosti

Sljedeći stupci se popunjavaju doslovnim nizom `'Unknown'`:

```
Manufacturing Region, Manufacturing Location Code, Intercompany,
Customer industry, Customer Region, Top Customer Group,
Product family, Product group, Make vs Buy,
Sales Channel - Internal, Sales Channel - External
```

Obrazloženje: `'Unknown'` postaje vlastita kategorija u label enkodiranju, čuvajući "odsutnost" kao značajku umjesto tihog uvođenja nevjerojatnih vrijednosti imputacijom modusom.

### 3.5 Oporavak GM% gdje je moguće

Gdje je `GM%` bio null, ali su bili prisutni i `Invoiced price ≠ 0` i `Cost of part`, GM% je reizračunat iz prvih načela:

```
GM% = (Invoiced price - Cost of part) / Invoiced price
```

Ovo oporavlja značajan udio redova. Preostali null `GM%` ostaje kakav jest — tim redovima nedostaje jedan od ulaza (tipično je `Cost of part` null za sve redove određene stavke, pa ne postoji izvor za imputaciju).

### 3.6 Konačni čisti dataset

Izlaz pipelinea (`data/processed/data_clean_final.csv`) je kanonski ulaz za **svaku nizvodnu bilježnicu**. Koraci čišćenja smanjuju dataset s ~1,3M sirovih stavki faktura na **~930K redova** zadržanih za modeliranje. Nakon gore navedenih koraka:

- Svi ključni identifikatori (`CustomerID`, `Item Code`, `Invoice Date`, `Invoiced price`) su popunjeni.
- Sve cijene su ≥ 0 i sve količine su > 0.
- `GM%` je ograničen na `[-5, 1]`.
- Kategorijalni null-ovi su eksplicitne `'Unknown'` vrijednosti.

### 3.7 Arhivirana istraživanja — `encoding_and_feature_engineering.ipynb`

Ranija bilježnica (`notebooks/archive/encoding_and_feature_engineering.ipynb`) label/one-hot enkodirala je čisti dataset i producirala enkodiranu datoteku s 928k redova i tablice agregata na razini kupca. Bila je korisna kao istraživački korak — potvrđujući distribucije značajki i agregate — ali produkcijski pipeline **ne ovisi** o njoj: svaka nizvodna bilježnica ponovo čita `data_clean_final.csv` i gradi značajke koje treba interno. Arhivirana bilježnica čuva se radi transparentnosti ali nije dio API puta.

---

## 4. Modeliranje

Sustav za preporuke ima tri modelarska komponente koje se nadograđuju jedna na drugu:

```
                ┌─────────────────────────────────┐
                │  RFM + K-Means Segmentacija      │  →  Cluster_name (4 segmenta)
                └─────────────────────────────────┘
                              │
        ┌─────────────────────┴────────────────────┐
        ▼                                          ▼
┌──────────────────────┐                ┌─────────────────────────┐
│ Market-Basket        │                │  Popularnost stavki      │
│ (Apriori/FP-Growth   │                │  uvjetovana klasterom    │
│  + stope ko-kupovine)│                │  (premium + sve stavke) │
└──────────────────────┘                └─────────────────────────┘
        │                                          │
        ▼                                          ▼
┌────────────────────────────┐         ┌─────────────────────────┐
│  Cross-Sell preporučitelj  │         │  Up-Sell preporučitelj   │
│  (XGBoost na Kupac×PG)     │         │  (ciljanje razreda +     │
│                            │         │   premium rangiranje     │
│                            │         │   uvjetovano klasterom)  │
└────────────────────────────┘         └─────────────────────────┘
```

Svi vremenski modeli koriste fiksnu raspodjelu:

- **Trening period:** fakture prije **2019-07-01** (~2,5 godina povijesti)
- **Label window (period oznake):** fakture od **2019-07-01 do 2020-07-01** (12-mjesečni napredni prozor)

Nasumični seedovi su fiksirani (`random_state=42`) na svakom modelu i na SMOTE-u.

### 4.1 Segmentacija kupaca — RFM + K-Means

Bilježnica: `notebooks/customer_segmentation.ipynb`.

#### 4.1.1 Izračun RFM-a

Datum snapshota: **2020-07-01** (jedan dan nakon maksimalnog datuma fakture u čistom datasetu).

Po kupcu:
- **Recency** — dani od kupčeve posljednje fakture do datuma snapshota (niže je bolje)
- **Frequency** — broj distinktnih faktura u kupčevoj povijesti
- **Monetary** — zbroj `Invoiced price × Invoiced qty (shipped)` kroz kupčevu povijest

**Izlaz: 11.753 kupaca** s potpunim RFM ocjenama.

#### 4.1.2 RFM ocjene (1–5) i segmenti temeljeni na pravilima

Svaka od R, F, M binira se u kvintile pomoću `pd.qcut` (Recency obrnuto: niže je bolje → ocjena 5). Kupci se zatim dodjeljuju čitljivom segmentu na temelju kanonskog R-F mapiranja (Champions, Loyal, At Risk, Hibernating, itd.). Ove oznake temeljene na pravilima čuvaju se kao opisni stupci uz K-Means klaster.

#### 4.1.3 K-Means klasteriranje

Ulazi su log-transformirani (`log1p`) i standardizirani (`StandardScaler`) kako bi se komprimirali teški desni repovi u Frequency i Monetary vrijednostima.

**Odabir K:** analiza lakta + silhouette za `K ∈ [2, 7]` (`n_init=10`, `random_state=42`). K=4 odabran je kao najbolji kompromis između silhouette score-a i interpretabilnosti klastera.

**Konačni klasteri i poslovna interpretacija:**

| Klaster | Naziv | Profil | Poslovno značenje |
|---|---|---|---|
| 1 | **Whales / Champions** | Nedavni, česti, vrlo visoka potrošnja | Korporativni ključni računi — razred s najvišim prihodom. Zaštiti odnos; preporuči nove kategorije oprezno. |
| 3 | **Active Small Buyers** | Nedavni, skromna frekvencija, skromna potrošnja | Zdrava jezgrena baza kupaca. Cross-sell kandidati s najvišom stopom pogodaka po kontaktu. |
| 2 | **Mid-Tier At Risk** | Umjerena nedavnost, umjerena potrošnja, opadajući angažman | Ciljevi za povratak. Personalizirani cross-sell može ih reaktivirati. |
| 0 | **Hibernating** | Dugo bez kontakta, niska frekvencija, niska potrošnja | Najveći rizik od odljeva. Jeftino reaktivirati, ali najniža očekivana vrijednost po ponudi. |

`Cluster_name` je eksportiran u `data/processed/rfm_segments.csv` i konzumiran od strane i cross-sell preporučitelja (kao značajka) i up-sell modela (za rangiranje stavki uvjetovano klasterom).

### 4.2 Analiza tržišne košarice (Market-Basket)

Bilježnica: `notebooks/market_basket.ipynb`.

#### 4.2.1 Odluka o granularnosti: Kupac × Kalendarski mjesec

Prva intuicija je koristiti `Invoice #` kao košaricu. Testirali smo to:

> **98,2% faktura sadrži samo jednu grupu proizvoda.** Fakture s više PG-ova čine 1,8% svih faktura — prerijetko da bi se producirala smislena asocijacijska pravila.

Kupčevi ritmi kupovine u B2B-u su PO-vođeni i tipično mjesečni (medijan razmaka između uzastopnih faktura ≈ 20 dana za ponovljene kupce). Redefiniranjem košarice kao **`CustomerID + YYYY-MM`** stopa košarica s više PG-ova raste na **27,5%** — dovoljno signala za Apriori/FP-Growth da pronađe pravila.

Ovo je namjerna, B2B-specifična metodološka odluka. U B2C maloprodajnom kontekstu, košarice na razini fakture bile bi pravi izbor.

#### 4.2.2 Odluka o granularnosti: Grupa proizvoda, ne kod stavke

`Item Code` ima ~52.000 jedinstvenih vrijednosti; pravila na toj granularnosti imala bi gotovo nulti support i preučila bi se na šum. Product group (~30 vrijednosti) je prirodna razina preporuke — o njoj razmišlja tim prodaje i po njoj e-commerce stranica kategorizira. Bonus analiza na razini Item Code-a u bilježnici potvrdila je da je PG prava granularnost.

#### 4.2.3 Algoritmi: Apriori i FP-Growth

Oba algoritma pokrenuta su s `min_support=0.01`, `max_len=3`. Kao što se i očekivalo, producirali su **identične česte skupove stavki** (verificirano unutar bilježnice). FP-Growth je preferirani za performanse; Apriori se čuva zbog didaktičke jasnoće (antimonotonost koja opravdava podrezivanje).

#### 4.2.4 Filtriranje pravila

Pravila su filtrirana prema tri praga:

| Metrika | Prag | Obrazloženje |
|---|---|---|
| Lift | > 2.0 | Pravila barem 2× jača od slučajnosti — jasno ne-slučajne asocijacije |
| Confidence | > 0.30 | Barem 30% vjerojatnost kupovine konsekventa uz dani antecedens |
| Conviction | > 1.3 | Nesuglasje s pretpostavkom nezavisnosti — provjera robusnosti |

**Rezultat: 14 jakih asocijacijskih pravila** (`data/processed/association_rules.csv`).

#### 4.2.5 Stope ko-kupovine kao proširenje signala

14 čvrstih pravila pokriva samo ~0,6% svih usmjerenih PG-prema-PG parova. Za proširenje signala, izračunali smo i `P(PG_b | PG_a)` za **sve** opažene usmjerene parove (do 870 = 30 × 29). Ovo producira 27 značajki po (kupcu, kandidat-PG) paru u cross-sell preporučitelju:

- `max_copurchase_rate` — najjači signal ko-kupovine od bilo kojeg postojećeg PG-a kupca
- `mean_copurchase_rate` — prosječni signal ko-kupovine kroz sve postojeće PG-ove

To znači da cross-sell model ima i **značajke čvrstih pravila** (iz 14 pravila) i **značajke mekog signala** (iz 870 parova ko-kupovine).

### 4.3 Cross-Sell preporučitelj — model Kupac × Grupa proizvoda

Bilježnica: `notebooks/cross_sell_recommender.ipynb`.

#### 4.3.1 Formulacija problema

Raniji rad (`notebooks/archive/cross_sell_model.ipynb`) formulirao je cross-sell kao binarnu klasifikaciju na razini kupca: "hoće li ovaj kupac ikad cross-sell-ati?" To odgovara na pitanje, ali ne na korisno — ne govori API-ju *koji* proizvod preporučiti.

Produkcijski preporučitelj preformulira zadatak kao **predikciju po paru**:

> Za svaki (kupac, kandidat-grupa proizvoda koja nije kupljena u trening periodu), predvidi vjerojatnost da će kupac kupiti iz te grupe u 12-mjesečnom label window-u (periodu oznake).

Ovo producira **rangirani popis po kupcu** koji izravno mapira na API izlaz.

#### 4.3.2 Konstrukcija matrice

Za svakog kupca u trening periodu:
- Uzmi skup PG-ova koje su već kupili u trening periodu (oni su isključeni — preporučujemo samo nove kategorije).
- Za svaki drugi PG u katalogu, stvori red.
- Ciljna varijabla = 1 ako je kupac kupio iz tog PG-a u label window-u, 0 inače.

**Rezultat:** ~257K (kupac, PG) parova, **0,34% stopa pozitivne klase** (≈ neravnoteža 290:1). Ova klasna neravnoteža temeljna je svojstvo problema: većina cross-sell mogućnosti ne konvertira u bilo kojoj danoj godini.

#### 4.3.3 Inženjerstvo značajki — 27 značajki po paru

Tri obitelji značajki:

**Značajke kupca (15)** — sve izračunate samo iz trening perioda:
- Volumetrijske: `train_revenue`, `train_orders`, `train_items`, `train_n_groups`
- Bihevioralne: `train_avg_gm`, `recency_days`, `tenure_days`, `order_frequency`
- Kategorijalne (label-enkodirane): `region_enc`, `industry_enc`, `top_group_enc`, `cluster_enc`
- RFM ocjene: `R_score`, `F_score`, `M_score`

**Značajke grupe proizvoda (7):**
- `pg_n_customers`, `pg_avg_price`, `pg_avg_gm`, `pg_total_orders`, `pg_avg_revenue`, `pg_popularity`, `pg_enc`

**Asocijacijske/ko-kupovne značajke (5):**
- Iz 14 čvrstih pravila: `rule_exists`, `rule_max_lift`, `rule_max_confidence`
- Iz 870 ko-kupovnih parova: `max_copurchase_rate`, `mean_copurchase_rate`

#### 4.3.4 Obrada klasne neravnoteže — SMOTE

Neravnoteža 290:1 previše je ekstremna za samo `class_weight`. **SMOTE** s `sampling_strategy=0.1` primjenjuje se **samo na fold za treniranje** (koristeći `imblearn.pipeline.Pipeline` tako da fold za testiranje ostaje netaknut). Ovo povećava stopu pozitivne klase u foldovima za treniranje s 0,34% na ~9%, dovoljno za modele temeljene na stablima da pronađu čistu granicu odluke.

#### 4.3.5 Usporedba modela

Tri modela trenirana na istom skupu značajki s identičnim SMOTE-om:

| Model | Test ROC-AUC | 5-Fold CV ROC-AUC | Napomene |
|---|---|---|---|
| Logistic Regression | 0.8654 | — | Osnova — linearno po značajkama |
| Random Forest | 0.9141 | — | Hvata nelinearne interakcije |
| **XGBoost** | **0.9248** | **0.9291 ± 0.0055** | Najbolji — tijesno ograničena varijanca kroz foldove |

5-fold unakrsna validacija koristi `GroupKFold` ključiran na `CustomerID` tako da svi parovi za zadanog kupca padaju u isti fold — sprječavajući leakage (curenje informacija) na razini kupca između treniranja i validacije.

Konfiguracija XGBoost-a:
```
n_estimators=300, max_depth=6, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8, random_state=42
```

XGBoost je odabran kao produkcijski model. Njegova `.pkl` datoteka sprema se u `models/recommender/recommender_xgb.pkl` i učitava API.

Naknadni eksperiment s kliznim prozorom u §13 bilježnice ponovo je trenirao XGBoost na tri dodatna povijesna prozora (srp–pro 2018, lis 2018–ožu 2019, sij–lip 2019) naslagana na osnovni trening period. Prošireno treniranje **nije nadmašilo** osnovni model — AUC je ostao na istoj razini — pa je implementirani `recommender_xgb.pkl` pre-rolling-window, 27-značajkasti XGBoost izvještavan gore (AUC 0.9248). Kod kliznog prozora čuva se u bilježnici radi reproducibilnosti ali nije dio produkcijskog artefakta.

#### 4.3.6 Razina 2 — Prijedlozi stavki unutar preporučenog PG-a

XGBoost model rangira **grupe proizvoda**, ali API vraća i **specifične kodove stavki** unutar svake preporučene grupe. Ovo je "Razina 2" pretraživanje — ponovo koristi popularnost stavki uvjetovanu klasterom iz up-sell bilježnice (bez premium-only filtera) za preporuku najpopularnijih stavki u preporučenom PG-u **među kupcima u istom klasteru**.

To znači da Whale kupac i Hibernating kupac dobivaju isti skup preporučenih PG-ova ali potencijalno različite prijedloge stavki unutar svakog PG-a.

### 4.4 Up-Sell model

Bilježnica: `notebooks/upsell_model.ipynb`.

#### 4.4.1 Klasifikacija razreda — Standard vs Premium

Za svaki `(Product group, Item Code)`, izračunaj prosječnu cijenu stavke kroz trening period. Zatim unutar svakog PG-a uzmi **medijan tih prosjeka stavki**. Stavke na medianu ili iznad njega označavaju se kao **`premium`**; ispod su **`standard`**.

**Rezultat: 38.734 razvrstane stavke, od kojih je 19.382 premium** (gornja polovica svakog PG-a prema prosjeku cijene stavke).

PG medijan prosjeka stavki — a ne sirovih linijskih cijena — odabran je kako jeftine stavke s iskrivljenim volumenom ne bi snizile prag.

#### 4.4.2 Identifikacija up-sell ciljeva

Za svaki (kupac, PG) par, izračunaj kupčevu prosječnu plaćenu cijenu u tom PG-u za vrijeme treniranja. Ako je njihov prosjek **ispod PG medijana**, oni su up-sell cilj — kupuju u kategoriji ali po cijenama nižeg razreda.

**Rezultat: 6.458 ciljnih (kupac, PG) parova kroz 4.993 jedinstvena kupca.**

Ovo je skup up-sell kandidata. Imajte na umu da API također obrađuje slučaj "trenutna stavka je već premium" vraćanjem bez preporuke.

#### 4.4.3 Popularnost premium stavki uvjetovana klasterom

Za svaku kombinaciju `(Cluster_name, Product group, Item Code)` gdje je stavka `premium`:

```
confidence = (# kupaca u klasteru koji su kupili ovu stavku u ovom PG-u)
             / (# kupaca u klasteru koji su kupili išta u ovom PG-u)
```

Filter `n_buyers ≥ 2` uklanja šum s jednom kupovinom. (Jedan poznati rubni slučaj: PG `SF001` ima samo jednu stavku, pa njegova rangiranja degeneriraju u "kupi jedinu dostupnu stavku" — zabilježeno u bilježnici.)

Ovo daje rangiranje "koje premium stavke kupuju moji vršnjaci?" po klasteru, što je personalizacijski sloj za up-sell preporuke.

#### 4.4.4 Generiranje preporuka

Za svaki up-sell cilj (kupac × PG):
1. Pronađi kupčev klaster.
2. Iz popularnosti premium stavki tog klastera u ciljnom PG-u, ispusti stavke koje je kupac već kupio.
3. Uzmi top-K (K=5) prema confidence-u.

**Rezultat: 27.744 up-sell preporuke** pokrivajući 4.843 kupaca kroz 26 grupa proizvoda.

---

## 5. Evaluacija

### 5.1 Evaluacija segmentacije

- **Veličine klastera** su dobro uravnotežene (nijedan klaster ne guta populaciju, nijedan nije degeneriran). Sva četiri klastera popunjena su s stotinama do tisuća kupaca.
- **Snake plot** (srednja vrijednost skalirane R, F, M po klasteru) pokazuje jasno različite profile — Whales su visoko na F i M s niskim R; Hibernating je zrcalna slika; dva srednja klastera jasno su odvojena po nedavnosti.
- **Interaktivni 3D plot** (`visualization/clusters_3d.html`) vizualno potvrđuje da četiri klastera zauzimaju distinktne regije log-skaliranog RFM prostora.
- **Silhouette score** pri K=4 bio je viši nego pri K=3 i kompetitivan s K=5/6 uz daleko bolju interpretabilnost za poslovne korisnike.

Klasteri nisu samo statistički odvojivi — mapiraju se na prepoznatljive poslovne arhetipove (ključni računi, zdrava jezgra, u riziku, neaktivni), što je teži kriterij za zadovoljiti.

### 5.2 Evaluacija cross-sell preporučitelja

#### 5.2.1 Primarna metrika: ROC-AUC

| Postavka | Ocjena |
|---|---|
| Test set ROC-AUC (XGBoost) | **0.9248** |
| 5-fold GroupKFold CV ROC-AUC | **0.9291 ± 0.0055** |
| Random Forest test AUC | 0.9141 |
| Logistic Regression test AUC | 0.8654 |

Interval unakrsne validacije je uzak (±0.005), što znači da je AUC stabilan kroz različite podskupove kupaca — nije slučajnost iz jedne povoljne raspodjele trening/test.

#### 5.2.2 Sekundarna metrika: Precision@5

Za svakog kupca u test skupu, uzmi prvih 5 preporučenih PG-ova modela i izračunaj udio koje je kupac stvarno kupio u label window-u.

- **Prosječni Precision@5: 0,38%**
- **Slučajna osnova (neravnoteža 290:1): 0,34%**

Poboljšanje u odnosu na slučajno je malo u apsolutnim iznosima — ovo je svojstvo osnovne klasne neravnoteže i dugorepe prirode B2B cross-sell-a, *ne* neuspjeh modela. AUC je odgovarajuća metrika za sustave rangiranja s ekstremnom neravnotežom; Precision@5 se izvještava kao provjera zdravog razuma, a ne kao glavna figura.

#### 5.2.3 Važnost značajki (Random Forest, kvalitativno)

Najjače značajke grupiraju se u tri grupe, otprilike ovim redoslijedom važnosti:

1. **Ko-kupovni signal** (`max_copurchase_rate`, `mean_copurchase_rate`) — proširenje 14 čvrstih pravila na cijeli krajolik od 870 parova imalo je visoki učinak.
2. **Bihevioralne značajke kupca** (`train_revenue`, `train_orders`, `train_n_groups`, `recency_days`).
3. **Bazne stope grupe proizvoda** (`pg_popularity`, `pg_n_customers`).

Značajke čvrstih pravila (`rule_exists`, `rule_max_lift`, `rule_max_confidence`) su korisne ali pokrivaju premalo parova da bi dominirale.

#### 5.2.4 Metodologija validacije — Zašto vremenski hold-out

Nasumične raspodjele na (kupac, PG) parovima dozvolile bi modelu da vidi jedan PG za kupca u treniranju i predvidi drugi za istog kupca u testiranju — to nije stvarni cross-sell scenarij. **Vremenski hold-out** (trening < 2019-07-01, label 2019-07-01 → 2020-07-01) jedini je način da se iskreno odgovori "s obzirom na ono što znam o kupcu danas, što će kupiti u sljedećih 12 mjeseci?" `GroupKFold` na `CustomerID` je analog unakrsne validacije istog načela.

### 5.3 Evaluacija up-sell modela

Validacija je teža jer ne postoji direktni signal "je li ovaj kupac prihvatio up-sell?". Koristimo **proxy naprednog rasta cijene**:

> Ako je kupac bio cilj u trening periodu (kupuje ispod PG medijana) i ponovo se pojavi u label window-u, je li njegova prosječna cijena u tom PG-u porasla?

**Ključni brojevi** (iz `models/upsell/upsell_meta.json`):

| Metrika | Vrijednost | Interpretacija |
|---|---|---|
| Ciljevi s podacima iz label window-a | 3.720 / 6.458 (59,9%) | Pokrivenost proxy metrike |
| Prosječni rast cijene (label vs. trening) | **+\$35,62** | Ciljevi su u prosjeku platili više u sljedećih 12 mjeseci |
| Udio s pozitivnim rastom | **44,25%** | Gotovo polovica pomakla se gore u cijeni unutar istog PG-a |
| Udio koji je prešao u premium razred | **7,61%** | Čvrsti signal — stvarno su kupili premium stavku |

Ovi brojevi se prikazuju iskreno s njihovim ograničenjima:

- **Nema kontrolne grupe.** Ne možemo odvojiti učinak modela od prirodnog rasta cijena na tržištu. Bilježnica to označava i predlaže izračunavanje iste stope migracije na slučajnom uzorku ne-ciljanih kupaca kao provjeru zdravog razuma.
- **Pokrivenost je 60%, ne 100%.** 40% ciljeva koji se ne pojave u label window-u ne pridonose ni dokazu ni protuargumentu.

Zajedno gledano: **up-sell ciljanje jasno identificira segment čije se cjenovno ponašanje kreće prema gore u naprednom prozoru.** Je li to kretanje kauzalno ili korelacijsko zahtijeva A/B test; proxy je dovoljno jak signal da opravda implementaciju (deployment) preporučitelja.

### 5.4 Kvaliteta preporuka API-ja

Osim metrika modela, kvaliteta preporuka API-ja vidljiva je na dva načina:

1. **Pokrivenost.** S 9.017 poznatih kupaca učitanih u memoriju i 30 grupa proizvoda, svaki postojeći kupac uvijek prima 5 preporuka. Cold-start (nepoznati `CustomerID` ili nije proslijeđen `customer_id`) vraća se na medijane populacije i još uvijek producira rangirani izlaz.
2. **Personalizacija.** Dva kupca u različitim klasterima koji prime isti preporučeni PG vide različite liste `items` — vođene `upsell_item_popularity_all.csv` filtriranim prema `Cluster_name`.

Živa latencija (nakon toplog pokretanja) je ispod sekunde po pozivu jer se svaki artefakt učitava u rječnik memorije pri pokretanju; pri obradi zahtjeva ne dolazi do čitanja s diska.

---

## 6. Implementacija (Deployment)

### 6.1 Arhitektura

Implementacija je **jednoprocesna FastAPI aplikacija** (`api/main.py`) koja učitava sve modelske artefakte i tablice za pretraživanje u memoriju pri pokretanju putem `lifespan` hook-a.

```
                   ┌──────────────────────────────────┐
                   │   POST /cross-sell                │
                   │   {customer_id, item_codes}       │
                   └──────────────────┬────────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │  Pretraživanja u        │
                          │  memoriji:              │
                          │  - customer_feats      │
                          │  - product_feats       │
                          │  - rules_lookup        │
                          │  - copurchase_rates    │
                          │  - feature_medians     │
                          │  - item_popularity_all │
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │  XGBoost preporučitelj │
                          │  (.pkl, joblib-loaded) │
                          └───────────┬────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │ Top-K PG-ovi + Razina  │
                          │ 2 stavke po PG-u        │
                          │ (uvjetovano klasterom) │
                          └────────────────────────┘
```

### 6.2 Endpoint-točke

#### `GET /health`
Provjera zdravog razuma da je model učitan i da su značajke kupaca u memoriji:
```json
{ "status": "ok", "model_loaded": true, "known_customers": 9017, "layer2_available": true }
```

#### `POST /cross-sell`

**Ulaz**
| Polje | Tip | Obavezno | Ponašanje ako nedostaje |
|---|---|---|---|
| `customer_id` | int | Ne | Vraća se na medijane značajki populacije |
| `item_codes` | list[str] | Ne | Nema ko-kupovnog / asocijacijskog signala |

**Logika**
1. Učitaj kupčeve pohranjene značajke (ili medijan kao zamjena) i `Cluster_name`.
2. Mapiraj proslijeđene `item_codes` na njihove PG-ove putem pretraživanja `item_tiers`.
3. Izgradi skup kandidata: svi PG-ovi **minus** oni koje kupac već kupuje **minus** PG-ovi iz trenutne narudžbe.
4. Za svaki kandidat PG, izgradi red s 27 značajki u trenutku zahtjeva (značajke pravila i ko-kupovne značajke ovise o živim `item_codes`, pa se ne računaju unaprijed).
5. Ocijeni s XGBoost-om, sortiraj po vjerojatnosti, uzmi top 5.
6. Za svaki top-K PG, pronađi Razinu 2 stavki u kupčevom klasteru.

**Izlaz**
```json
{
  "customer_id": 12345,
  "recommendations": [
    {
      "product_group": "PC016",
      "confidence": 0.4642,
      "items": [
        {"item_code": "BOE001-TY8.375", "confidence": 0.1098, "avg_price": 6.02}
      ]
    }
  ]
}
```

`confidence` na PG-u je XGBoost-om predviđena vjerojatnost da će kupac kupiti iz te grupe. `confidence` na svakoj stavci je udio kupaca u klasteru koji kupuju tu stavku unutar PG-a.

#### `POST /up-sell`

**Ulaz**
| Polje | Tip | Obavezno |
|---|---|---|
| `customer_id` | int | Ne (preskače personalizaciju klastera ako nedostaje) |
| `item_codes` | list[str] | Da |

**Logika**
Za svaki ulazni kod stavke:
1. Pronađi razred i PG stavke (preskoči ako nepoznat).
2. Ako je stavka već `premium`, preskoči — nema čega up-sell-ati.
3. Filtriraj `upsell_popularity` na isti PG, zatim opcionalno na kupčev klaster (ako je klaster pronađen; inače poslužuj bez klastera).
4. Uzmi top-K premium alternativa prema confidence-u, isključujući stavku koju kupac već kupuje.

**Izlaz**
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
        {"item_code": "2158", "confidence": 0.0495, "avg_price": 78.99}
      ]
    }
  ]
}
```

### 6.3 Cold-Start ponašanje

Oba endpoint-a prihvaćaju nedostajući `customer_id` (B2B walk-up slučaj ili kupac koji dolazi prvi put):

- **Cross-sell:** značajke se vraćaju na `feature_medians.json` (medijani populacije iz treniranja); Razina 2 stavki specifičnih za klaster vraća se na rangiranje neovisno o klasteru. Rezultat: razborito "što bi prosječni kupac sljedeće htio?" odgovor.
- **Up-sell:** personalizacija klastera se preskače i vraća se rangiranje premium stavki unutar PG-a neovisno o klasteru.

### 6.4 Pokretanje u produkciji

Referentna naredba (razvojna okolina):

```bash
conda activate lumen-ds
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Napomene o produkcijskom implementaciji:

- **Pretpostavka jednog radnika.** `state: dict` je unutar procesa; implementacija s više radnika (Gunicorn) zahtijevala bi ili dijeljenu cache ili učitavanje istih artefakata po radniku.
- **Nema autentikacije ni ograničenja brzine.** Oba endpoint-a su otvorena po dizajnu (API je interan). Dodaj autentikaciju na razini gatewaya ako je API izložen izvan mreže tvrtke.
- **Nema verzioniranja modela osim `model_meta.json`.** Kada se modeli ponovo treniraju, meta JSON se prepisuje — nema automatskog zapisa koji je hash dataseta producirao koji `.pkl`.
- **Zahtjevi API-ja su verzijski fiksirani** (`api/requirements.txt`) — okruženje bilježnice dokumentirano je u README/Run Book-u.

### 6.5 Reproducibilnost

Svaki korak u pipelineu koristi fiksne seedove i determinističke datume raspodjele:

| Parametar | Vrijednost | Gdje je postavljen |
|---|---|---|
| Nasumični seed | 42 | Svi sklearn/XGBoost/SMOTE konstruktori |
| Granična vrijednost treniranja | 2019-07-01 | Bilježnice cross-sell, up-sell |
| Kraj label window-a | 2020-07-01 | Bilježnice cross-sell, up-sell |
| RFM snapshot | 2020-07-01 | Segmentacija kupaca |
| K (klasteri) | 4 | Segmentacija kupaca |
| SMOTE strategija | 0.1 | Cross-sell preporučitelj |
| Top-K | 5 | Oba endpoint-a, obe bilježnice za treniranje |

Manji izvor varijance — XGBoost `n_jobs=-1` može producirati ±0.002 AUC varijaciju zbog nedeterminističke akumulacije decimalnih zareza u paralelu — dokumentiran je u Run Book-u i ne utječe na poslovne odluke ni rangiranja.

---

## Dodatak — Sažetak izlaza pipelinea

| Bilježnica | Primarni izlaz | Koristi ga |
|---|---|---|
| `01 data_cleaning` | `data/processed/data_clean_final.csv` | Svaka nizvodna bilježnica |
| `02 customer_segmentation` | `data/processed/rfm_segments.csv` | Cross-sell preporučitelj, up-sell model |
| `03 market_basket` | `data/processed/association_rules.csv` | Cross-sell preporučitelj |
| `04 cross_sell_recommender` | `models/recommender/*.pkl`, `data/processed/recommender/*.csv`, `feature_medians.json` | API `/cross-sell` |
| `05 upsell_model` | `data/processed/upsell/*.csv`, `models/upsell/upsell_meta.json` | API `/up-sell` i `/cross-sell` Razina 2 |
| `archive/encoding_and_feature_engineering` | `customer_features.csv`, `data_clean_enc_fe.csv`, `invoice_items.csv` | Samo istraživački — ne konzumira ga produkcijski pipeline (vidi §3.7) |
| `archive/cross_sell_model` | `models/cross_sell/*.pkl` | Samo istraživački — zamijenjeno produkcijskim cross-sell preporučiteljem (vidi §4.3.1) |

---

*Ovaj dokument prati [README.md](../../README.md) (brzi start i postavke okruženja) i [Run Book](../runbooks/RUN_BOOK.md) (operativni postupci i rješavanje problema).*
