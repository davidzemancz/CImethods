# Warehouse Organization for MAPD

Optimalizace rozložení skladu pomocí Multi-Agent Pickup and Delivery (MAPD) s využitím surrogate modelů.

## Cíl

Najít **optimální rozmístění zboží** ve skladu tak, aby byla maximalizována **propustnost** (počet odbavených objednávek za čas).

## Model skladu

```
┌───┬───┬───┬───┬───┬───┬───┐
│ D │ D │ D │ D │ D │ D │ D │   D = Delivery point (okraj)
├───┼───┼───┼───┼───┼───┼───┤       Agenti zde startují a doručují
│ D │ A │ B │ C │ D │ E │ D │
├───┼───┼───┼───┼───┼───┼───┤   A,B,C,... = Druhy zboží
│ D │ F │ G │ H │ I │ J │ D │       Každá vnitřní buňka = 1 druh
├───┼───┼───┼───┼───┼───┼───┤
│ D │ K │ L │ M │ N │ O │ D │
├───┼───┼───┼───┼───┼───┼───┤
│ D │ P │ Q │ R │ S │ T │ D │
├───┼───┼───┼───┼───┼───┼───┤
│ D │ D │ D │ D │ D │ D │ D │
└───┴───┴───┴───┴───┴───┴───┘
```

### Předpoklady
- **Grid**: M×N buněk
- **Okraj**: všechny hraniční buňky = delivery points
- **Vnitřek**: (M-2)×(N-2) buněk, **každá obsahuje právě 1 druh zboží**
- **Počet druhů zboží** = počet vnitřních buněk
- **Bez překážek**: agenti se mohou pohybovat přes všechny buňky

### Objednávky
- Každá objednávka = požadavek na **1 konkrétní zboží**
- **Distribuce**: očekávaný počet objednávek pro každý druh (např. A: 100/h, B: 20/h)

### Vstupy a výstup
| Vstup | Popis |
|-------|-------|
| Rozložení zboží | Permutace: který druh zboží je na které vnitřní pozici |
| Distribuce objednávek | Očekávaný počet objednávek pro každý druh |
| Počet agentů | Kolik robotů operuje ve skladu |

**Výstup**: Propustnost = počet dokončených objednávek / čas

## MAPD Simulace

### Lifecycle agenta
1. **Start**: Agent čeká na delivery pointu (okraj)
2. **Pickup**: Dostane úkol → jde k pozici zboží
3. **Delivery**: Nese zboží k **nejbližšímu volnému okraji**
4. **Idle**: Zůstává na místě, kde doručil, čeká na další úkol

### Task assignment
- **Heuristika**: objednávku dostane **nejbližší volný agent**
- Agent je "volný" když je v IDLE stavu (čeká na okraji)

### Path planning (SAT)
Pro každý MAPF problém (agenti s cíli) řešíme SAT:
- **Proměnné**: `x[a][t][v]` = agent `a` je v čase `t` na pozici `v`
- **Klauzule**:
  - Každý agent v každém čase na právě jedné pozici
  - Pohyb jen na sousední buňku nebo stání
  - Vertex conflict: dva agenti nemohou být na stejném místě
  - Edge conflict: dva agenti si nemohou prohodit pozice
  - Start a cíl pro každého agenta

### SAT limity
- **Makespan limit**: max počet timestepů pro cestu (např. 2× Manhattan distance)
- **Timeout**: max 5 sekund na jeden SAT dotaz
- Pokud SAT neuspěje → agent čeká a zkusí znovu příští timestep

### Online replanning
Když přijde nová objednávka za běhu simulace:
1. **Vyber agenta** (nejbližší volný)
2. **Zmraz aktivní agenty** - jejich aktuální cesty se nemění
3. **Naplánuj cestu novému agentovi** pomocí SAT s "zmraženými" agenty jako pohyblivými překážkami
4. Pokud SAT nenajde řešení → agent **čeká 1 timestep** a zkusí znovu

### SAT solver
[python-sat](https://pysathq.github.io/) (pysat) - Python rozhraní pro Glucose/MiniSat

## Evoluční algoritmus

### Reprezentace
- **Genotyp**: Permutace délky K (počet druhů zboží = počet vnitřních buněk)
- **Gen[i]** = index pozice, kde je zboží i
- **Fenotyp**: Konkrétní mapa skladu

### Operátory
| Operátor | Metoda | Popis |
|----------|--------|-------|
| Selekce | Turnajová | Turnaj velikosti 3 |
| Křížení | Order Crossover (OX) | Zachovává relativní pořadí |
| Mutace | Swap | Prohodí 2 náhodné pozice |

### Fitness
```python
def fitness(individual, generation, surrogate):
    if generation % N == 0:  # každých N generací
        real_fitness = run_mapd_simulation(individual)
        surrogate.update(individual, real_fitness)  # přetrénování
        return real_fitness
    else:
        return surrogate.predict(individual)
```

## Surrogate model

### Vstupní features
Pro rozložení zboží s pozicemi a distribucí objednávek:

| Feature | Popis |
|---------|-------|
| `weighted_avg_dist` | Průměrná vzdálenost od okraje, vážená počtem objednávek |
| `weighted_std_dist` | Směrodatná odchylka vzdáleností (vážená) |
| `max_popular_dist` | Max vzdálenost pro top 10% nejpopulárnějšího zboží |
| `congestion_score` | Jak blízko u sebe je populární zboží (potenciální konflikty) |
| `n_agents` | Počet agentů |
| `grid_size` | M × N |

### Regresní modely
| Model | Implementace | Proč |
|-------|--------------|------|
| **Linear Regression** | `sklearn.linear_model.LinearRegression` | Jednoduchý baseline, interpretovatelný |
| **Gaussian Process** | `sklearn.gaussian_process.GaussianProcessRegressor` | Uncertainty odhad, sample-efficient |
| **Random Forest** | `sklearn.ensemble.RandomForestRegressor` | Rychlý, robustní |
| **XGBoost** | `xgboost.XGBRegressor` | Velmi přesný, dobře škáluje |

### Evaluace
- **Metriky**: RMSE, MAE, R²
- **Validace**: K-fold cross-validation

## Architektura řešení

```
                              ┌──────────────────┐
                         ┌───▶│  Surrogate model │───┐
                         │    │  (rychlá aprox.) │   │
                         │    └──────────────────┘   │
┌─────────────────┐      │                           ▼    ┌─────────────────┐
│  Evoluční       │──────┤       FITNESS            ├───▶│  Fitness        │
│  algoritmus     │      │                           │    │  (propustnost)  │
└─────────────────┘      │    ┌──────────────────┐   │    └─────────────────┘
        ▲                └───▶│  MAPD simulace   │───┘
        │                     │  (SAT solver)    │
        │                     └──────────────────┘
        │                     každých N generací
        │
        └─── nová generace (selekce, křížení, mutace)
```

## Parametry experimentu

| Parametr | Hodnota |
|----------|---------|
| Grid | 7×7 (25 vnitřních buněk = 25 druhů zboží) |
| Počet agentů | 3-5 |
| Délka simulace | 1000 timestepů |
| Distribuce objednávek | Zipf (několik populárních, většina vzácných) |
| Populace EA | 50 jedinců |
| Generace EA | 100 |
| Přetrénování surrogate | Každých 10 generací |

## Baseline

Pro porovnání evoluce použijeme dva baseline přístupy:

| Baseline | Popis |
|----------|-------|
| **Random** | Náhodná permutace zboží |
| **Greedy heuristic** | Populární zboží (dle distribuce) umístěno co nejblíže okraji |

## Metriky pro porovnání

### Kvalita řešení
- **Propustnost** nejlepšího nalezeného rozložení

### Efektivita surrogate
| Metrika | Popis |
|---------|-------|
| **Počet simulací** | Kolik reálných MAPD simulací bylo potřeba |
| **Wall-clock time** | Celkový čas optimalizace |
| **Simulace k cíli** | Počet simulací potřebných k dosažení X% optima |

### Porovnání
- Evoluce **s** surrogate vs. **bez** surrogate (pure EA)
- Při fixním počtu simulací: která metoda najde lepší řešení?
- Při fixním čase: která metoda najde lepší řešení?

## Implementační plán

### Fáze 1: Základní infrastruktura
- [ ] Třída `Warehouse` - grid, pozice zboží, agenti
- [ ] Třída `OrderGenerator` - generování objednávek dle distribuce

### Fáze 2: MAPD Solver
- [ ] SAT formulace pro MAPF
- [ ] Task assignment (nejbližší volný agent)
- [ ] Online replanning s "zmraženými" agenty
- [ ] Měření propustnosti

### Fáze 3: Surrogate modely
- [ ] Feature extraction
- [ ] GP, Random Forest, XGBoost
- [ ] Evaluace a výběr nejlepšího

### Fáze 4: Evoluční optimalizace
- [ ] DEAP setup (permutace, OX, swap)
- [ ] Hybridní fitness (surrogate + real)
- [ ] Experimenty

### Fáze 5: Analýza
- [ ] Porovnání: evoluce s/bez surrogate
- [ ] Vizualizace nejlepších rozložení
- [ ] Statistické vyhodnocení

## Závislosti

```
numpy>=1.20
scipy>=1.7
matplotlib>=3.4
scikit-learn>=1.0
xgboost>=1.5
deap>=1.3
python-sat>=0.1.8
```

## Literatura

- [SAT-based MAPF](https://ojs.aaai.org/index.php/ICAPS/article/view/13795) - Surynek et al., ICAPS 2019
- [Gaussian Processes for ML](https://gaussianprocess.org/gpml/) - Rasmussen & Williams
- [Surrogate-Assisted Evolutionary Optimization](https://link.springer.com/article/10.1007/s11047-018-9702-0) - Jin, 2011
