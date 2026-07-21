# Error Analysis Report

## Overall Error Distribution

| Config | n | Exact Match | Partial (F1>0.5) | Partial (F1≤0.5) | Wrong (F1=0) | Empty Prediction | NER Format Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline mT5 (1x) | 848 | 27 (3.2%) | 23 (2.7%) | 51 (6.0%) | 745 (87.9%) | 2 (0.2%) | 0 (0.0%) |
| Baseline ByT5 (1x) | 848 | 21 (2.5%) | 24 (2.8%) | 63 (7.4%) | 740 (87.3%) | 0 (0.0%) | 0 (0.0%) |
| Matched-Vol mT5 (20x) | 848 | 167 (19.7%) | 31 (3.7%) | 29 (3.4%) | 621 (73.2%) | 0 (0.0%) | 0 (0.0%) |
| Matched-Vol ByT5 (20x) | 848 | 186 (21.9%) | 54 (6.4%) | 59 (7.0%) | 549 (64.7%) | 0 (0.0%) | 0 (0.0%) |
| Multitask mT5 | 848 | 178 (21.0%) | 39 (4.6%) | 38 (4.5%) | 578 (68.2%) | 0 (0.0%) | 15 (1.8%) |
| Multitask ByT5 | 848 | 163 (19.2%) | 64 (7.5%) | 56 (6.6%) | 562 (66.3%) | 0 (0.0%) | 3 (0.4%) |
| LoRA mT5 | 848 | 139 (16.4%) | 29 (3.4%) | 27 (3.2%) | 653 (77.0%) | 0 (0.0%) | 0 (0.0%) |
| LoRA ByT5 | 848 | 38 (4.5%) | 42 (5.0%) | 33 (3.9%) | 735 (86.7%) | 0 (0.0%) | 0 (0.0%) |
| Translation Pipeline | 848 | 98 (11.6%) | 35 (4.1%) | 135 (15.9%) | 578 (68.2%) | 2 (0.2%) | 0 (0.0%) |

## Per-Language Error Breakdown

### Matched-Vol ByT5

| Language | Exact Match | Partial (F1>0.5) | Partial (F1≤0.5) | Wrong (F1=0) | Empty Prediction | NER Format Leakage |
|---|---:|---:|---:|---:|---:|---:|
| SWA | 53 (18.0%) | 21 (7.1%) | 19 (6.4%) | 202 (68.5%) | 0 (0.0%) | 0 (0.0%) |
| HAU | 83 (27.7%) | 15 (5.0%) | 15 (5.0%) | 187 (62.3%) | 0 (0.0%) | 0 (0.0%) |
| YOR | 50 (19.8%) | 18 (7.1%) | 25 (9.9%) | 160 (63.2%) | 0 (0.0%) | 0 (0.0%) |

### Multitask mT5

| Language | Exact Match | Partial (F1>0.5) | Partial (F1≤0.5) | Wrong (F1=0) | Empty Prediction | NER Format Leakage |
|---|---:|---:|---:|---:|---:|---:|
| SWA | 52 (17.6%) | 12 (4.1%) | 12 (4.1%) | 213 (72.2%) | 0 (0.0%) | 6 (2.0%) |
| HAU | 83 (27.7%) | 8 (2.7%) | 5 (1.7%) | 200 (66.7%) | 0 (0.0%) | 4 (1.3%) |
| YOR | 43 (17.0%) | 19 (7.5%) | 21 (8.3%) | 165 (65.2%) | 0 (0.0%) | 5 (2.0%) |

## Qualitative Examples (Matched-Vol ByT5)

### Exact Match

**SWA:**

- **Gold:** Webuye
  **Pred:** Webuye
  *(F1=1.0)*

- **Gold:** Oginga Odinga
  **Pred:** Oginga Odinga
  *(F1=1.0)*

- **Gold:** Joseph Kasa-Vubu
  **Pred:** Joseph Kasa-Vubu
  *(F1=1.0)*

**HAU:**

- **Gold:** Eni Njoku
  **Pred:** Eni Njoku
  *(F1=1.0)*

- **Gold:** 1951
  **Pred:** 1951
  *(F1=1.0)*

- **Gold:** 1937
  **Pred:** 1937
  *(F1=1.0)*

**YOR:**

- **Gold:** Osogun
  **Pred:** Osogun
  *(F1=1.0)*

- **Gold:** Port-au-Prince
  **Pred:** Port-au-Prince
  *(F1=1.0)*

- **Gold:** yes
  **Pred:** yes
  *(F1=1.0)*

### Partial (F1>0.5)

**SWA:**

- **Gold:** Irvine, California
  **Pred:** Irvine
  *(F1=0.667)*

- **Gold:** Stephen Wilfred Omondi Oludhe
  **Pred:** Stephen Wilfred
  *(F1=0.667)*

- **Gold:** Thomas Alva Edison
  **Pred:** Thomas Alva Edis
  *(F1=0.667)*

**HAU:**

- **Gold:** James Francis Cameron
  **Pred:** James Francis Ca
  *(F1=0.667)*

- **Gold:** Burj Khalifa
  **Pred:** Burj Khalifa na
  *(F1=0.8)*

- **Gold:** Babagana Umara Zulum
  **Pred:** Babagana Umara Z
  *(F1=0.667)*

**YOR:**

- **Gold:** Francis Scott Key
  **Pred:** Francis Scott Ke
  *(F1=0.667)*

- **Gold:** John Stith Pemberton
  **Pred:** John Stith Pembe
  *(F1=0.667)*

- **Gold:** 1927
  **Pred:** ọdun 1927
  *(F1=0.667)*

### Partial (F1≤0.5)

**SWA:**

- **Gold:** William Macewen
  **Pred:** Sir William Mace
  *(F1=0.4)*

- **Gold:** Joshua Irungu
  **Pred:** Joshua Wakahora
  *(F1=0.5)*

- **Gold:** Mark Zuckerberg pamoja na wanafunzi wenzake wa Harvard College ambao pia walishiriki chumba, Eduardo Saverin, Andrew McCollum, Dustin Moskovitz na Chris Hughes
  **Pred:** Mark Zuckerberg
  *(F1=0.167)*

**HAU:**

- **Gold:** biliyan 1.413
  **Pred:** 1.413 bilion
  *(F1=0.5)*

- **Gold:** Sergio Mattarella
  **Pred:** Sergio Mattarell
  *(F1=0.5)*

- **Gold:** Robert Guérin
  **Pred:** Robert Guerin
  *(F1=0.5)*

**YOR:**

- **Gold:** 6 Agẹmọ 1967 – 15 ṣẹẹrẹ 1970
  **Pred:** 1970
  *(F1=0.286)*

- **Gold:** The Dilema of Rev. Father Michael
  **Pred:** The Dilemma of R
  *(F1=0.4)*

- **Gold:** There Was a Country: A Personal History of Biafra
  **Pred:** There Was a Coun
  *(F1=0.462)*

### Wrong (F1=0)

**SWA:**

- **Gold:** Mt Petro
  **Pred:** Annuario Pontifi
  *(F1=0.0)*

- **Gold:** kila baada ya miaka 10
  **Pred:** 23
  *(F1=0.0)*

- **Gold:** Mwaka wa 1990
  **Pred:** National Party
  *(F1=0.0)*

**HAU:**

- **Gold:** gabas ta kudu
  **Pred:** bahagwayen mutan
  *(F1=0.0)*

- **Gold:** Doha
  **Pred:** Al Jazeera
  *(F1=0.0)*

- **Gold:** bakwai
  **Pred:** eh
  *(F1=0.0)*

**YOR:**

- **Gold:** 21
  **Pred:** 1977
  *(F1=0.0)*

- **Gold:** 2005
  **Pred:** 2 Ọ̀wàwà
  *(F1=0.0)*

- **Gold:** Oluṣẹgun ọbasanjọ
  **Pred:** General Yakubu G
  *(F1=0.0)*
