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

## Klíčová zjištění

### 1. Efektivita náhradního modelu

| Experiment | Nejlepší R² | Rozdíl fitness | Zrychlení |
|------------|-------------|----------------|-----------|
| 9×9, 1 agent, SAT | 0.136 (RF) | +3.8% | 7.6x |
| 9×9, 2 agenti, Zipf 1.5 | 0.125 (RF) | -6.0% | 7.25x |
| 9×9, 4 agenti, Zipf 0.9 | 0.065 (GP) | -12.6% | 11.7x |

### 2. Faktory ovlivňující kvalitu surrogate

**Pozitivní faktory:**
- Strmější Zipf distribuce (1.5 > 0.9) → větší rozdíly ve fitness
- Více trénovacích vzorků (500+)
- Random Forest obecně funguje nejlépe

**Negativní faktory:**
- Plochá Zipf distribuce → malé rozdíly ve fitness
- Vysoká stochasticita v MAPD simulaci
- Příliš málo nebo příliš mnoho agentů vzhledem k velikosti mapy

### 3. Kompromisy

- **Zrychlení vs. kvalita**: Vyšší zrychlení často přichází s mírně nižší výslednou fitness
- **R² vs. výkon EA**: I nízké R² (0.06-0.14) poskytuje užitečné vedení pro EA
- **Surrogate funguje nejlépe když**: Rozložení významně ovlivňuje propustnost

---

## Závěry

1. **EA s náhradním modelem konzistentně dosahuje 7-12x zrychlení** s 90%+ redukcí nákladných simulací

2. **Kvalita surrogate (R²) je střední** (0.06-0.14), ale dostatečná pro vedení evoluce

3. **Random Forest** je obecně nejlepší náhradní model pro tento problém

4. **Greedy baseline často selhává** oproti náhodnému prohledávání - jednoduché heuristiky nezachycují konflikty agentů

5. **Konfigurace je důležitá**:
   - Strmá Zipf (1.5) → lepší predikce surrogate
   - Přiměřený počet agentů vzhledem k velikosti mapy → rozložení má větší vliv

---

## Soubory

| Soubor | Popis |
|--------|-------|
| `warehouse.py` | Třídy Warehouse a OrderGenerator |
| `mapd_solver.py` | A* a SAT plánovače MAPF, MAPD simulátor |
| `surrogate.py` | Extrakce features (40 features), náhradní modely |
| `evolution.py` | EA založené na DEAP s hybridní fitness evaluací |
| `baselines.py` | Random, Greedy, Inverse Greedy baseline |
| `project_*.ipynb` | Notebooky s experimenty |
