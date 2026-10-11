# Methodology

This document explains the modelling choices in `ninecat` and how each one
is tested. Results live in [`reports/results.md`](../reports/results.md).

## The decision problem

In a 9-category head-to-head league you play one opponent per week. Each of
FG%, FT%, 3PM, PTS, REB, AST, STL, BLK and TO is won or lost on that week's
totals, and the team that wins more categories wins the week. Every in-season
decision (who to stream, which trade to accept, whether to punt a category)
is a bet on **weekly** outcomes. That framing drives the whole design:

| Question | Tool | Objective |
|---|---|---|
| How good is a player for the rest of the season? | `project` + `category_scores` | per-game projection, G-score |
| Will I win this week? | `simulate_week` | P(win), P(win each category) |
| Who should I stream? | `streaming_moves` | change in this week's P(win) |
| Should I take this trade? | `evaluate_trade` | change in average weekly P(win) vs. the league |
| What should I punt? | `team_profile`, `suggest_punts` | category strength vs. the league |

## 1. Projections (`ninecat/projections.py`)

A player's expected line for a scheduled game factors into three pieces,
each with its own model:

```
E[stat | scheduled game] = availability × minutes per game × stat per minute
```

**Per-minute rates: Gamma–Poisson empirical Bayes.** For each stat and position
group we fit a Gamma(a, b) prior on per-minute rates by the method of moments.
The between-player variance is the observed variance of rates minus the
expected sampling variance, inflated for overdispersion. A player's posterior is

```
shape = a + Σ wᵢ xᵢ        rate = b + Σ wᵢ minᵢ        estimate = shape / rate
```

where `wᵢ` are optional recency weights. The prior strength `b` is measured in
minutes: it is how much playing time it takes before a player's own numbers
outweigh his position's. It differs by stat. Blocks vary a lot between
players, so they shrink little; steals are mostly noise, so they shrink a lot.
The posterior shape is kept and reused later as parameter uncertainty.

**Shooting percentages: Beta–Binomial.** Same idea on makes and attempts. A
guard who starts 9-for-10 from the line is not a 90% shooter yet.

**Minutes.** Exponentially weighted over the last games played (6-game
half-life), regressed toward the player's season average. Minutes change with
role (trades, injuries to teammates, demotions), so this is the piece where
recency belongs.

**Availability.** The recency-weighted share of the team's games a player
appeared in (8-game half-life), shrunk toward a league prior of 88%. A player
in the middle of an injury spell gets a low number that recovers as he plays.

**Overdispersion.** Box-score counts vary more than Poisson because of hot
nights and minutes swings. A pooled negative-binomial size `r` per stat
(variance = μ + μ²/r) is estimated by the method of moments and passed to the
simulator.

### What the backtest changed

I expected recency weighting of per-minute rates to help. On held-out seasons
it didn't: decaying old games threw away information faster than it tracked
real change, so the tuned default uses no decay on rates. Recency still helps
for **minutes**, where real role changes happen. The grid search also found
the best prior strength close to 1× the empirical-Bayes estimate, which is a
useful check that the method-of-moments prior is about right.

## 2. Valuation: Z-scores vs. G-scores (`ninecat/scoring.py`)

The standard approach standardises each category across the rostered pool:
`Z = (μ − μ_pool) / σ_pool`. Rosenof (2024, [arXiv:2307.02188](https://arxiv.org/abs/2307.02188))
points out that head-to-head categories are decided by *weekly* totals, so an
edge in a volatile category is worth less than the same edge in a stable one.
His G-score adds the average within-player weekly variance τ²:

```
G = (μ − μ_pool) / sqrt(σ_pool² + κ τ²),     κ = 2N / (2N − 1)
```

`ninecat` computes τ² from the same model the simulator uses (negative-binomial
game variance, plus the variance from missed games) and converts per-game
quantities to weekly ones. Between-player spread grows like `g` with games
played; within-player noise grows like `√g`.

Percentage categories use volume-weighted impact, `attempts × (pct −
pool_pct) / mean_attempts`, so volume matters. The within-player variance of
that impact includes both binomial make/miss noise and attempt variance.

The reference pool is the top `teams × roster_size` players and is re-estimated
iteratively. Punting a category sets its weight to zero in the total.

## 3. Matchup simulation (`ninecat/simulation.py`)

For every player and simulated week:

1. **Games played** ~ Binomial(scheduled games remaining, availability).
2. **True rates for the week** are drawn from the projection posterior:
   Gamma for each per-minute rate, Beta for each percentage, lognormal minutes
   shock. This keeps the simulator from being overconfident about players with
   small samples. In the calibration study its effect is small, because most
   rostered players already have large samples by week 3. Overdispersion
   matters much more.
3. **Weekly counts** ~ Gamma–Poisson over the games played. A sum of `n`
   i.i.d. NB(μ, r) variables is NB(nμ, nr), so this is one draw rather than `n`.
4. **Makes** ~ Binomial(attempts, pct), so FG% and FT% come from simulated
   volume, not from averaging percentages.

Stats already banked earlier in the week are added to the totals, which
makes mid-week projections exact about the past and uncertain only about
the future.

**Common random numbers.** All player-weeks are simulated once into an array
of shape `(simulations, players)`. A team total is a column sum, and a lineup
change is a subtract-and-add on the same simulated weeks. The streaming tool
compares hundreds of add/drop pairs this way in one vectorised pass.
Comparisons of nearly identical lineups have far lower variance than they
would with independent simulations.

**Known simplifications.** Stats within a game are simulated independently
given the player's rates (no shared hot-night factor across stats), and
3PM is not constrained to be at most FGM within a simulated week. Both are
easy extensions, and calibration (below) suggests neither matters much.

## 4. In-season tools (`ninecat/inseason.py`)

**Streaming.** Candidates are the free agents with the most value this week
(G-score × games remaining). Every (add, drop) pair is scored by the change in
this week's win probability, with ties counting half. Protected players are
never dropped.

**Trades.** Each team's roster is measured before and after against the same
field (every other team in the league), in a typical rest-of-season week
where each player plays his team's average remaining games per week.
Uneven trades are balanced the way a manager would balance them: the side
with too many players drops its least valuable one, and the side with an
open spot picks up the best free agent. The output is the change in average
weekly win probability, the change in total G-score, and the change in the
probability of winning each category. That last one is where a trade's real
effect usually shows up.

Win probability, not summed G-score, is the decision metric. That is also
what makes punting coherent: a team that has given up on FT% gains nothing
from FT% in the simulator, without any hand-set weights.

## 5. Evaluation (`ninecat/experiments.py`)

Real data shows only one realisation of each season, which makes it hard to
separate a good model from a lucky one. So every model is evaluated on
synthetic seasons (`ninecat/synthetic.py`) generated from a known process:
position-dependent and correlated talent, aging curves, drifting form, role
changes, injury spells, overdispersed and internally consistent box scores
(`PTS = 2·FGM + 3PM + FTM`).

To avoid fitting the evaluation:

* hyperparameters are tuned on seeds 101–103 and reported on seeds 1–5;
* every projection uses only games before the evaluation date;
* calibration compares simulated probabilities with the box scores that
  actually happened in that week.

| Experiment | Question | Metric |
|---|---|---|
| Projection backtest | Are rest-of-season projections better than simple baselines? | weighted MSE per stat |
| Tuning | How much recency and how much shrinkage? | mean MSE reduction vs. season average |
| Calibration | When the simulator says 70%, does it happen 70% of the time? | Brier score, log loss, ECE, reliability diagram |
| G vs. Z | Which ranking better orders players by simulated marginal value? | Spearman correlation |

**Limits.** Synthetic seasons are only as realistic as their generator. A
model can look good on them by matching the generator's assumptions, which
is a real risk for the G vs. Z experiment because the simulator and G-scores
share a variance model. The natural next step is to rerun the projection
backtest and calibration on real seasons with `scripts/fetch_nba_logs.py`.
The code path is the same; only the data changes.
