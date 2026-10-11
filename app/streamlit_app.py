"""ninecat dashboard: in-season tools for 9-cat head-to-head leagues.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ninecat as nc
from ninecat.categories import CATEGORIES, LABELS
from ninecat.data import DAYS_PER_WEEK
from ninecat.inseason import best_pickups
from ninecat.scoring import suggest_punts
from ninecat.simulation import swing_categories

st.set_page_config(page_title="ninecat", page_icon="🏀", layout="wide")

BLUE, ORANGE = "#2a78d6", "#eb6834"


# ---------------------------------------------------------------------------
# Cached model calls
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Generating league ...")
def load_league(seed: int):
    season, league, _ = nc.generate_season(nc.SyntheticConfig(seed=seed))
    return season, league


@st.cache_resource(show_spinner="Projecting players ...")
def load_projections(seed: int, day: int):
    season, _ = load_league(seed)
    return nc.project(season, day)


@st.cache_data(show_spinner=False)
def load_scores(seed: int, day: int, method: str, weights: tuple) -> pd.DataFrame:
    return nc.category_scores(load_projections(seed, day), method=method, weights=dict(weights))


def label(proj, pid) -> str:
    r = proj.table.loc[pid]
    return f"{r['name']} ({r['nba_team']}, {r['position']})"


def prob_chart(p: pd.Series, title: str) -> alt.Chart:
    df = pd.DataFrame({"category": [LABELS[c] for c in p.index], "p": p.to_numpy()})
    bars = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4, height=16).encode(
        x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=title),
        y=alt.Y("category:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
        tooltip=[alt.Tooltip("category:N"), alt.Tooltip("p:Q", format=".1%")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color="#898781").encode(x="x:Q")
    text = bars.mark_text(align="left", dx=4, color="#52514e").encode(text=alt.Text("p:Q", format=".0%"))
    return (bars + rule + text).properties(height=9 * 34)


def delta_chart(d: pd.Series, title: str) -> alt.Chart:
    df = pd.DataFrame({"category": [LABELS[c] for c in d.index], "delta": d.to_numpy()})
    df["direction"] = np.where(df["delta"] >= 0, "better", "worse")
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=16).encode(
        x=alt.X("delta:Q", axis=alt.Axis(format="+%"), title=title),
        y=alt.Y("category:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["better", "worse"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("category:N"), alt.Tooltip("delta:Q", format="+.1%")],
    ).properties(height=9 * 34)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("ninecat")
st.sidebar.caption("In-season analytics for 9-category head-to-head leagues.")
seed = st.sidebar.number_input("League seed", min_value=0, max_value=10_000, value=7, step=1,
                               help="Each seed generates a different synthetic season and league.")
season, league = load_league(int(seed))
week = st.sidebar.slider("Fantasy week", 2, season.n_weeks, 11) - 1  # 0-indexed internally
days_in = st.sidebar.slider("Days into the week", 0, DAYS_PER_WEEK - 1, 0,
                            help="Games before this day are already banked in the matchup.")
day = week * DAYS_PER_WEEK + days_in
team = st.sidebar.selectbox("My team", league.teams)
method = st.sidebar.radio("Valuation", ["G-score", "Z-score"], horizontal=True,
                          help="G-scores account for week-to-week volatility (Rosenof, 2024).")
punts = st.sidebar.multiselect("Punt categories", CATEGORIES, format_func=LABELS.get)
n_sims = st.sidebar.select_slider("Simulations", [1000, 2000, 5000, 10000], value=5000)

proj = load_projections(int(seed), day)
weights = tuple(sorted({c: (0.0 if c in punts else 1.0) for c in CATEGORIES}.items()))
scores = load_scores(int(seed), day, "g" if method == "G-score" else "z", weights)
ctx = nc.week_context(season, league, week, day)
opponent = league.opponent(team, week)

st.title(f"{team} · Week {week + 1}")
st.caption("Data is a synthetic season from `ninecat.synthetic`, so every number here can be "
           "checked against a known ground truth. See the repo README for real-data loading.")

tabs = st.tabs(["Matchup", "Streaming", "Trade analyzer", "Team profile", "Rankings"])

# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------
with tabs[0]:
    res = nc.simulate_week(proj, league, ctx, team, opponent, n_sims=n_sims, seed=week)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win", f"{res.win_prob:.0%}")
    c2.metric("Tie", f"{res.tie_prob:.0%}")
    c3.metric("Loss", f"{res.loss_prob:.0%}")
    c4.metric("Expected categories", f"{res.expected_categories:.2f} / 9")
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"vs {opponent}")
        st.altair_chart(prob_chart(res.category_win_prob, "P(win category)"), width="stretch")
    with right:
        swing = swing_categories(res)
        st.subheader("Swing categories")
        st.write(", ".join(LABELS[c] for c in swing) if swing else "None: every category is close to decided.")
        tbl = res.summary().rename(index=LABELS, columns={"team_a": team, "team_b": opponent,
                                                          "p_team_a_wins": "P(win)"})
        st.dataframe(tbl.style.format({team: "{:.3f}", opponent: "{:.3f}", "P(win)": "{:.0%}"}),
                     width="stretch")
        g_left = ctx.games_left.loc[league.rosters[team]].sum()
        g_opp = ctx.games_left.loc[league.rosters[opponent]].sum()
        st.caption(f"Games left this week: {team} {g_left:.0f}, {opponent} {g_opp:.0f}. "
                   "Weekly totals include stats already banked.")

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Best add/drop moves for this week")
    roster = league.rosters[team]
    n_protect = st.slider("Never drop my top N players", 0, 12, 6)
    protect = scores.loc[roster, "total"].nlargest(n_protect).index.tolist()
    with st.spinner("Simulating every add/drop pair ..."):
        moves = nc.streaming_moves(proj, league, ctx, team, scores, opponent, n_sims=min(n_sims, 4000),
                                   seed=week, protect=protect)
    st.caption(f"Current win probability (ties count half): {moves.attrs['baseline_win_prob']:.1%}. "
               "Every candidate is evaluated on the same simulated weeks, so small differences are real.")
    best_per_add = moves.drop_duplicates("add").head(15)
    show = best_per_add[["add_name", "add_games", "drop_name", "drop_games", "win_prob",
                         "delta_win_prob", "delta_categories"]].rename(columns={
        "add_name": "Add", "add_games": "G left", "drop_name": "Drop", "drop_games": "G left (drop)",
        "win_prob": "Win prob", "delta_win_prob": "Δ win prob", "delta_categories": "Δ exp. cats"})
    st.dataframe(show.style.format({"Win prob": "{:.1%}", "Δ win prob": "{:+.1%}", "Δ exp. cats": "{:+.2f}",
                                    "G left": "{:.0f}", "G left (drop)": "{:.0f}"}),
                 width="stretch", hide_index=True)

    st.subheader("Rest-of-season pickups")
    picks = best_pickups(scores, league, team)
    picks["Add"] = [label(proj, p) for p in picks["add"]]
    picks["Drop"] = [label(proj, p) for p in picks["drop"]]
    st.dataframe(picks[["Add", "add_value", "Drop", "gain"]].rename(
        columns={"add_value": f"{method} total", "gain": "Gain"}).style.format({f"{method} total": "{:.2f}",
                                                                               "Gain": "{:+.2f}"}),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Trade analyzer")
    partner = st.selectbox("Trade partner", [t for t in league.teams if t != team])
    c1, c2 = st.columns(2)
    send = c1.multiselect("I send", league.rosters[team], format_func=lambda p: label(proj, p))
    get = c2.multiselect("I receive", league.rosters[partner], format_func=lambda p: label(proj, p))
    if send or get:
        games = nc.typical_week_games(season, day)
        with st.spinner("Simulating both teams against the league ..."):
            mine, theirs = nc.evaluate_trade(proj, league, scores, games, team, send, partner, get,
                                             n_sims=min(n_sims, 3000), seed=week)
        for col, side in zip(st.columns(2), (mine, theirs)):
            with col:
                st.markdown(f"#### {side.team}")
                a, b = st.columns(2)
                a.metric("Weekly win rate vs league", f"{side.win_rate_after:.1%}",
                         f"{side.delta_win_rate:+.1%}")
                b.metric(f"Roster {method} total", f"{side.value_after:.2f}", f"{side.delta_value:+.2f}")
                st.altair_chart(delta_chart(side.categories_after - side.categories_before,
                                            "Change in P(win category)"), width="stretch")
                notes = [f"drops {label(proj, p)}" for p in side.auto_drops]
                notes += [f"picks up {label(proj, p)}" for p in side.auto_adds]
                if notes:
                    st.caption("To keep a full roster, " + side.team + " " + "; ".join(notes) + ".")
        verdict = ("Accept" if mine.delta_win_rate > 0.005 else "Decline" if mine.delta_win_rate < -0.005
                   else "Roughly neutral")
        st.info(f"**{verdict}** for {team}: weekly win rate {mine.win_rate_before:.1%} → {mine.win_rate_after:.1%}.")
    else:
        st.write("Pick players on either side to evaluate a trade.")

# ---------------------------------------------------------------------------
# Team profile
# ---------------------------------------------------------------------------
with tabs[3]:
    profile = nc.team_profile(scores, league.rosters)
    z = (profile - profile.mean()) / profile.std(ddof=0)
    st.subheader("Category strength by team (league standard deviations)")
    st.dataframe(z.rename(columns=LABELS).style.format("{:+.2f}").background_gradient(cmap="RdBu", vmin=-2, vmax=2),
                 width="stretch")
    suggestion = suggest_punts(profile, team)
    if suggestion:
        st.write(f"**{team}** is far behind the league in {', '.join(LABELS[c] for c in suggestion)}. "
                 "Consider punting: set them in the sidebar and rankings re-weight instantly.")
    else:
        st.write(f"**{team}** has no category far enough behind to be an obvious punt.")

# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------
with tabs[4]:
    fa_only = st.checkbox("Free agents only")
    r = nc.rankings(proj, method="g" if method == "G-score" else "z", weights=dict(weights))
    if fa_only:
        r = r[~r.index.isin(league.rostered())]
    owner = {p: t for t, ps in league.rosters.items() for p in ps}
    r.insert(2, "owner", [owner.get(p, "FA") for p in r.index])
    cols = ["rank", "name", "owner", "nba_team", "position", "gp", "min", "availability",
            "fg_pct", "ft_pct", "fg3m", "pts", "reb", "ast", "stl", "blk", "tov", "v_total"]
    st.dataframe(r[cols].head(250).rename(columns=LABELS | {"v_total": f"{method} total"}).style.format(precision=2),
                 width="stretch", hide_index=True)
