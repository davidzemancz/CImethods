# Optimalizace rozložení skladu - Experimentální report

## Přehled

Tento projekt implementuje **evoluční algoritmus s náhradním modelem (surrogate-assisted EA)** pro optimalizaci rozložení zboží ve skladu s multi-agentním systémem pro vyzvedávání a doručování (MAPD). Cílem je minimalizovat počet nákladných MAPD simulací pomocí náhradního modelu pro predikci fitness.

## Popis problému

- **Sklad**: Mřížkové rozložení se zbožím ve vnitřních buňkách a delivery points na okrajích
- **Agenti**: Autonomní roboti, kteří vyzvedávají zboží a doručují ho na delivery points
- **Cíl**: Maximalizovat propustnost (dokončené objednávky za časový krok)
- **Optimalizace**: Najít optimální umístění zboží (permutační problém)

## Metodologie

1. **Baseline metody**: Náhodné prohledávání, Greedy (populární zboží blízko okraje), Inverse Greedy
2. **Čisté EA**: Evoluční algoritmus s reálnými MAPD simulacemi pro všechny evaluace
3. **EA s náhradním modelem**: Používá surrogate model pro většinu evaluací, reálná simulace každých N generací

### Testované náhradní modely
- Lineární regrese
- Ridge regrese
- Gaussovský proces (GP)
- Random Forest (RF)

### Features (40 celkem)
- Vzdálenostní: vážený průměr/std vzdálenosti od okraje, top-N vzdálenosti
- Poziční: populární položky na okraji, podíl v rozích
- Kongescní: skóre shlukování, párové vzdálenosti, rozptyl
- Distribuční: entropie, Gini koeficient, pravděpodobnostní hmota na okraji/středu
- Prostorové: kvadrantové features, rozprostření, vyváženost
- Konfliktní potenciál: hotspot, bottleneck, překryv cest

---

## Shrnutí experimentů

### Experiment 1: Velká mapa, jeden agent (SAT Planner)

| Parametr | Hodnota |
|----------|---------|
| Velikost mřížky | 9×9 |
| Agenti | 1 |
| Plánovač | SAT |
| Distribuce | Zipf (výchozí) |
| Populace | 30 |
| Generace | 100 |

**Výkon náhradního modelu:**
| Model | R² skóre |
|-------|----------|
| Linear | 0.013 ± 0.210 |
| GP | 0.036 ± 0.152 |
| **RF** | **0.136 ± 0.178** |

**Výsledky:**
| Metoda | Nejlepší fitness | Reálných eval. | Čas |
|--------|------------------|----------------|-----|
| Random (best) | 0.2000 | - | - |
| Random (mean) | 0.1376 | - | - |
| Greedy | 0.1500 | - | - |
| EA bez surrogate | 0.2600 | 3030 | 696s |
| **EA se surrogate** | **0.2700** | **256** | **92s** |

- **Úspora evaluací**: 91.6%
- **Zrychlení**: 7.6x
- **Poznámka**: EA se surrogate dosáhlo *lepší* fitness než čisté EA

---

### Experiment 2: Strmá Zipf distribuce (A* Planner)

| Parametr | Hodnota |
|----------|---------|
| Velikost mřížky | 9×9 |
| Agenti | 2 |
| Plánovač | A* (space-time) |
| Distribuce | **Zipf 1.5** |
| Populace | 100 |
| Generace | 50 |

**Výkon náhradního modelu:**
| Model | R² skóre |
|-------|----------|
| Linear | -0.492 ± 0.569 |
| Ridge | -0.054 ± 0.246 |
| GP | 0.093 ± 0.068 |
| **RF** | **0.125 ± 0.114** |

**Výsledky:**
| Metoda | Nejlepší fitness | Reálných eval. | Čas |
|--------|------------------|----------------|-----|
| Random (best) | 0.3560 | - | - |
| Random (mean) | 0.2065 | - | - |
| Greedy | 0.1900 | - | - |
| EA bez surrogate | **0.4000** | 5100 | 5898s |
| EA se surrogate | 0.3760 | 477 | 814s |

- **Úspora evaluací**: 90.6%
- **Zrychlení**: 7.25x
- **Poznámka**: Strmější Zipf (1.5) vytváří větší rozdíly ve fitness, RF dosahuje R²=0.125

---

### Experiment 3: Plochá Zipf, více agentů (A* Planner)

| Parametr | Hodnota |
|----------|---------|
| Velikost mřížky | 9×9 |
| Agenti | **4** |
| Plánovač | A* (space-time) |
| Distribuce | **Zipf 0.9** |
| Populace | 5 |
| Generace | 100 |

**Výkon náhradního modelu:**
| Model | R² skóre |
|-------|----------|
| Linear | -0.002 ± 0.093 |
| Ridge | 0.036 ± 0.063 |
| **GP** | **0.065 ± 0.035** |
| RF | -0.002 ± 0.105 |

**Výsledky:**
| Metoda | Nejlepší fitness | Reálných eval. | Čas |
|--------|------------------|----------------|-----|
| Random (best) | 0.5367 | - | - |
| Random (mean) | 0.3718 | - | - |
| Greedy | 0.4633 | - | - |
| EA bez surrogate | **0.5567** | 505 | 346s |
| EA se surrogate | 0.4867 | 35 | 30s |

- **Úspora evaluací**: 93.1%
- **Zrychlení**: 11.7x
- **Poznámka**: Plochá Zipf (0.9) ztěžuje predikci (nízké R²), ale stále dosahuje 11.7x zrychlení

---

### Experiment 4: Feature Selection & Ensemble (A* Planner)

| Parametr | Hodnota |
|----------|---------|
| Velikost mřížky | 9×9 |
| Agenti | **4** |
| Plánovač | A* (space-time) |
| Distribuce | **Zipf 1.2** |
| Populace | 20 |
| Generace | 50 |
| Trénovacích vzorků | 300 |

**Analýza feature importance (Random Forest):**

| Pořadí | Feature | Importance |
|--------|---------|------------|
| 1 | edge_prob_mass | 0.1425 |
| 2 | dist_gini | 0.0591 |
| 3 | quad_SE | 0.0504 |
| 4 | spread_x | 0.0500 |
| 5 | quad_NW | 0.0437 |
| 6 | hotspot | 0.0412 |
| 7 | dispersion | 0.0395 |
| 8 | skewness | 0.0375 |
| 9 | quadrant_balance | 0.0370 |
| 10 | quad_SW | 0.0369 |

**Korelace s fitness (top 10):**

| Feature | Korelace |
|---------|----------|
| edge_prob_mass | +0.449 |
| median_dist | -0.444 |
| dist_gini | -0.440 |
| throughput_ratio | +0.434 |
| weighted_avg_dist | -0.429 |
| expected_travel | -0.429 |
| avg_to_delivery | -0.429 |
| skewness | +0.401 |
| weighted_sq_dist | -0.394 |
| spread_x | +0.335 |

**Feature Selection - vliv na R²:**

| Počet features (K) | F-regression | Mutual Info | RF Importance |
|--------------------|--------------|-------------|---------------|
| 5 | 0.192 ± 0.026 | 0.166 ± 0.045 | 0.187 ± 0.032 |
| 10 | **0.198 ± 0.034** | 0.171 ± 0.042 | 0.145 ± 0.048 |
| 15 | 0.170 ± 0.031 | 0.151 ± 0.036 | 0.133 ± 0.048 |
| 20 | 0.167 ± 0.039 | 0.147 ± 0.033 | 0.120 ± 0.051 |
| 40 (všechny) | 0.072 ± 0.075 | 0.072 ± 0.075 | 0.072 ± 0.075 |

**Porovnání modelů (s vybranými features):**

| Model | Features | R² skóre |
|-------|----------|----------|
| **Voting Ensemble** | 10 | **0.166 ± 0.056** |
| Baseline GP | 40 | 0.152 ± 0.040 |
| Ridge (selected) | 10 | 0.145 ± 0.048 |
| Baseline RF | 40 | 0.145 ± 0.038 |
| RF (selected) | 10 | 0.136 ± 0.055 |
| Stacking Ensemble | 10 | 0.123 ± 0.019 |
| GBM (selected) | 10 | 0.084 ± 0.099 |

**Výsledky EA:**

| Metoda | Nejlepší fitness | Reálných eval. | Čas | Speedup |
|--------|------------------|----------------|-----|---------|
| Random (best) | 0.4800 | - | - | - |
| Random (mean) | 0.3399 | - | - | - |
| Greedy | 0.4400 | - | - | - |
| EA bez surrogate | **0.5500** | 1020 | 3321.8s | 1.00x |
| EA + GP surrogate | 0.5367 | 168 | 715.2s | 4.64x |
| EA + RF surrogate | 0.4867 | 168 | 689.0s | 4.82x |

- **Úspora evaluací**: 83.5%
- **Zrychlení**: 4.6-4.8x
- **Feature selection zlepšuje R²**: z 0.072 na 0.198 (2.75x zlepšení)
- **Optimální počet features**: 10 (F-regression)

---

## Klíčová zjištění

### 1. Efektivita náhradního modelu

| Experiment | Nejlepší R² | Rozdíl fitness | Zrychlení |
|------------|-------------|----------------|-----------|
| 9×9, 1 agent, SAT | 0.136 (RF) | +3.8% | 7.6x |
| 9×9, 2 agenti, Zipf 1.5 | 0.125 (RF) | -6.0% | 7.25x |
| 9×9, 4 agenti, Zipf 0.9 | 0.065 (GP) | -12.6% | 11.7x |
| 9×9, 4 agenti, Zipf 1.2 (feature sel.) | 0.198 (Ridge) | -2.4% | 4.64x |

### 2. Feature Selection

**Klíčové poznatky:**
- Redukce z 40 na 10 features **zlepšuje R² z 0.072 na 0.198** (2.75x)
- Optimální počet features: **5-10** (F-regression metoda)
- Příliš mnoho features vede k overfitting

**Nejdůležitější features:**
1. `edge_prob_mass` - podíl pravděpodobnostní hmoty na okraji (korelace +0.449)
2. `median_dist` - mediánová vzdálenost od okraje (korelace -0.444)
3. `dist_gini` - Gini koeficient vzdáleností (korelace -0.440)
4. `throughput_ratio` - teoretický poměr propustnosti (korelace +0.434)
5. `weighted_avg_dist` - vážená průměrná vzdálenost (korelace -0.429)

### 3. Ensemble modely

| Model | R² skóre | Poznámka |
|-------|----------|----------|
| Voting Ensemble | 0.166 ± 0.056 | Nejstabilnější |
| Stacking Ensemble | 0.123 ± 0.019 | Nízká variance |
| Jednotlivé modely | 0.08-0.15 | Vyšší variance |

**Závěr:** Voting ensemble (Ridge + RF + GBM) poskytuje nejlepší kompromis mezi přesností a stabilitou.

### 4. Faktory ovlivňující kvalitu surrogate

**Pozitivní faktory:**
- Strmější Zipf distribuce (1.5 > 0.9) → větší rozdíly ve fitness
- Více trénovacích vzorků (300+)
- Feature selection (redukce na 10 features)
- Random Forest nebo GP jako base model

**Negativní faktory:**
- Plochá Zipf distribuce → malé rozdíly ve fitness
- Vysoká stochasticita v MAPD simulaci
- Příliš málo nebo příliš mnoho agentů vzhledem k velikosti mapy
- Příliš mnoho features (overfitting)

### 5. Kompromisy

- **Zrychlení vs. kvalita**: Vyšší zrychlení často přichází s mírně nižší výslednou fitness
- **R² vs. výkon EA**: I nízké R² (0.06-0.14) poskytuje užitečné vedení pro EA
- **Feature selection**: Zlepšuje R², ale může ztratit informace důležité pro specifické konfigurace
- **Surrogate funguje nejlépe když**: Rozložení významně ovlivňuje propustnost

---

## Závěry

1. **EA s náhradním modelem konzistentně dosahuje 4-12x zrychlení** s 80-93% redukcí nákladných simulací

2. **Feature selection dramaticky zlepšuje kvalitu surrogate**: R² z 0.07 na 0.20 (2.75x zlepšení) při redukci z 40 na 10 features

3. **Nejdůležitější features** jsou vzdálenostní metriky (`edge_prob_mass`, `weighted_avg_dist`, `median_dist`)

4. **Voting ensemble** (Ridge + RF + GBM) poskytuje nejstabilnější predikce

5. **Random Forest a GP** jsou obecně nejlepší base modely pro surrogate

6. **Greedy baseline často selhává** oproti náhodnému prohledávání - jednoduché heuristiky nezachycují konflikty agentů

7. **Konfigurace je důležitá**:
   - Strmá Zipf (1.2-1.5) → lepší predikce surrogate
   - Přiměřený počet agentů vzhledem k velikosti mapy → rozložení má větší vliv
   - 5-10 vybraných features → optimální kompromis

---

## Vizualizace

Report obsahuje následující grafy (generované v `project_feature_selection_ensemble.ipynb`):

| Soubor | Popis |
|--------|-------|
| `ea_convergence_detailed.png` | Detailní konvergence (4 grafy): best fitness, avg fitness, relativní zlepšení, fitness vs. čas |
| `ea_comparison_bars.png` | Bar chart porovnání: fitness, čas, efektivita |
| `ea_tradeoff.png` | Trade-off analýza: fitness vs. evaluace, speedup vs. kvalita |
| `ea_distribution.png` | Box plot a violin plot distribuce fitness |
| `ea_radar_comparison.png` | Radar chart multikriteriálního porovnání |
| `ea_summary_table.png` | Souhrnná tabulka výsledků |
| `feature_selection_convergence.png` | Graf konvergence EA metod |

---

## Soubory

| Soubor | Popis |
|--------|-------|
| `warehouse.py` | Třídy Warehouse a OrderGenerator |
| `mapd_solver.py` | A* a SAT plánovače MAPF, MAPD simulátor |
| `surrogate.py` | Extrakce features (40 features), náhradní modely |
| `evolution.py` | EA založené na DEAP s hybridní fitness evaluací |
| `baselines.py` | Random, Greedy, Inverse Greedy baseline |
| `project_9x9_*.ipynb` | Experimenty s různými konfiguracemi |
| `project_feature_selection_ensemble.ipynb` | Feature selection a ensemble modely |
| `REPORT.md` | Tento report |
