#!/usr/bin/env python3
"""
FPL expected-points model, Monte Carlo simulator and squad/transfer optimiser.

Quick start
    pip install -r requirements.txt
    python fpl_model.py                      # live FPL API: projections + best squad from scratch
    python fpl_model.py --entry 1234567      # + best transfers & captain for YOUR team
    python fpl_model.py --entry 1234567 --free-transfers 2 --max-hits 1
    python fpl_model.py --team-file my_team.txt --refresh --open   # your 15 + bank + selling prices, typed in

Pipeline
    1. Data       official FPL API (default) or a community GitHub mirror (--source mirror)
    2. Teams      attack / defence ratings from xG (75%) and goals (25%), shrunk to league average
    3. Players    per-90 rates (xG, xA, saves, defensive contributions, bonus, cards),
                  shrunk toward last season's rates
    4. Minutes    recency-weighted P(appear), P(60+), expected minutes, x injury flag
    5. Fixture    player rate x (team's expected goals in this match / team average)
    6. Scoring    current FPL rules -> expected points (xP) per player per gameweek
    7. Simulate   Monte Carlo -> floor, ceiling, haul and blank probabilities
    8. Backtest   refit on GW1..k-1, predict GW k, compare with recent form (and FPL's ep_next in mirror mode)
    9. Optimise   integer program: 15-man squad, XI and captain per GW, under budget / 3-per-club;
                  with --entry: best transfers from your current team, paying -4 per extra transfer

Outputs (in --out, default ./out):
    projections_gw<N>.csv   every player, xP per GW, floor/ceiling, haul %, etc.
    explain_gw<N>.csv       where each player's xP comes from (minutes, goals, CS, bonus...)
    squad_gw<N>.csv         recommended 15 (from scratch, or after transfers with --entry)
    backtest.csv            model accuracy on past gameweeks
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import sys
import time
import unicodedata
import webbrowser
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pulp
import requests

# ----------------------------------------------------------------------------- settings
FPL_API = "https://fantasy.premierleague.com/api"
MIRROR = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data"
MIRROR_FIXTURES = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

HOME_ADV = 1.10            # home side scores ~10% more, away ~9% less
TEAM_SHRINK_GAMES = 6      # pseudo-games of league average added to every team's record
PLAYER_PRIOR_MIN = 450     # pseudo-minutes of last-season rate added to every player's record
HORIZON_DECAY = 0.9        # GW+1 counts 90% of GW, GW+2 81%, ... (later weeks are less certain)
BENCH_WEIGHT = 0.1         # value of bench points in the optimiser (autosubs happen sometimes)
HIT_COST = 4               # FPL points per transfer beyond the free ones
RECENCY = [0.32, 0.24, 0.16, 0.11, 0.09, 0.08]   # minutes weights, most recent GW first

GOAL_PTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
DEFCON_THR = {"GKP": None, "DEF": 10, "MID": 12, "FWD": 12}   # 2 pts at this many CBI(T)(R)
POS_ID = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POS_NAME = {"Goalkeeper": "GKP", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}
SQUAD_SHAPE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_SHAPE = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}

STAT_COLS = ["minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
             "bonus", "expected_goals", "expected_assists", "expected_goals_conceded",
             "defensive_contribution", "yellow_cards", "starts", "total_points"]

RNG = np.random.default_rng(7)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (fpl-model)"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


ALLOWED_HOSTS = {"fantasy.premierleague.com", "raw.githubusercontent.com"}


def check_url(url):
    """Only ever talk to the FPL site (and the GitHub mirror) over HTTPS. Local http is allowed for tests."""
    u = urlparse(url)
    local = u.hostname in ("127.0.0.1", "localhost")
    if not ((u.scheme == "https" and u.hostname in ALLOWED_HOSTS) or (local and u.scheme == "http")):
        raise ValueError(f"Refusing to fetch {url}: not an allowed host")


def get_json(url, tries=4):
    check_url(url)
    for k in range(tries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(2 ** k)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    raise RuntimeError(f"Could not fetch {url}")


# ============================================================================ DATA
# Every loader returns the same Data dict:
#   players  : one row per player NOW  (player_id, player_code, web_name, pos, team, now_cost [£m],
#              status, chance_of_playing_next_round, selected_by_percent, ep_next, news)
#   per      : one row per player per match played this season (STAT_COLS + gw, match, team)
#   fixtures : event, team_h, team_a, kickoff_time
#   prior    : last season per-player totals keyed on player_code
#   teams    : id, short_name
#   history  : optional {gw: players-like snapshot as it was before that GW} for honest backtests
#   last_gw, next_gw

def load_api(cache_dir: Path, base=FPL_API, refresh=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"Fetching FPL API from {base} ...")
    boot = get_json(f"{base}/bootstrap-static/")
    fixtures = pd.DataFrame(get_json(f"{base}/fixtures/"))
    events = pd.DataFrame(boot["events"])
    nxt = events[events.is_next]
    next_gw = int(nxt.id.iloc[0]) if len(nxt) else int(events.id.max())
    finished = events[events.finished]
    last_gw = int(finished.id.max()) if len(finished) else 0

    teams = pd.DataFrame(boot["teams"])[["id", "short_name", "name"]]
    el = pd.DataFrame(boot["elements"])
    el = el[el.element_type.isin(POS_ID)]
    players = pd.DataFrame({
        "player_id": el.id, "player_code": el.code, "web_name": el.web_name,
        "pos": el.element_type.map(POS_ID), "team": el.team, "now_cost": el.now_cost / 10,
        "status": el.status, "chance_of_playing_next_round": el.chance_of_playing_next_round,
        "selected_by_percent": pd.to_numeric(el.selected_by_percent, errors="coerce"),
        "ep_next": pd.to_numeric(el.ep_next, errors="coerce"), "news": el.news.fillna(""),
        "full_name": el.first_name.fillna("") + " " + el.second_name.fillna(""),
    })
    dl = events.loc[events.id == next_gw, "deadline_time"]
    deadline = str(dl.iloc[0]) if len(dl) else None

    # per-player match history (one request per player, cached per finished GW so reruns are instant)
    cache = cache_dir / f"element_summaries_gw{last_gw}.json"
    summaries = {}
    if cache.exists() and not refresh:
        summaries = {int(k): v for k, v in json.loads(cache.read_text()).items()}
    todo = [int(i) for i in players.player_id if int(i) not in summaries]
    if todo:
        log(f"Downloading match history for {len(todo)} players (cached after the first run) ...")

        def one(pid):
            return pid, get_json(f"{base}/element-summary/{pid}/")
        with ThreadPoolExecutor(max_workers=6) as ex:
            for n, (pid, js) in enumerate(ex.map(one, todo), 1):
                summaries[pid] = {"history": js.get("history", []), "history_past": js.get("history_past", [])}
                if n % 100 == 0:
                    log(f"  {n}/{len(todo)}")
        cache.write_text(json.dumps(summaries))

    hist_rows, past_rows = [], []
    for pid, js in summaries.items():
        hist_rows += js["history"]
        if js["history_past"]:
            past_rows.append(js["history_past"][-1])          # most recent completed season
    per = pd.DataFrame(hist_rows)
    if per.empty:
        per = pd.DataFrame(columns=STAT_COLS + ["element", "round", "fixture", "was_home", "opponent_team"])
    for c in STAT_COLS:
        per[c] = pd.to_numeric(per.get(c, 0), errors="coerce").fillna(0)
    fx_team = fixtures.set_index("id")[["team_h", "team_a"]]
    per = per.rename(columns={"element": "player_id", "round": "gw", "fixture": "match"})
    per["team"] = np.where(per.was_home.astype(bool), per.match.map(fx_team.team_h), per.match.map(fx_team.team_a))
    per = per.merge(players[["player_id", "player_code", "web_name", "pos"]], on="player_id", how="inner")
    per = per[per.gw <= last_gw] if last_gw else per

    prior = pd.DataFrame(past_rows).rename(columns={"element_code": "player_code"})
    for c in ["minutes", "expected_goals", "expected_assists", "saves", "defensive_contribution", "bonus", "yellow_cards"]:
        prior[c] = pd.to_numeric(prior[c], errors="coerce") if c in prior else np.nan

    fixtures = fixtures[["event", "team_h", "team_a", "kickoff_time"]].dropna(subset=["event"])
    fixtures["event"] = fixtures.event.astype(int)
    return dict(players=players, per=per, fixtures=fixtures, prior=prior, teams=teams,
                history=None, last_gw=last_gw, next_gw=next_gw, deadline=deadline, source="FPL API (live)")


def load_mirror(cache_dir: Path, refresh=False, season="2026-2027", prev="2025-2026", fx_season="2026-27"):
    """Community mirror of weekly FPL API snapshots. Use when the official API is unreachable."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def csv(url, name, volatile=True):
        p = cache_dir / name
        if not p.exists() or (refresh and volatile):      # last season never changes; this season updates daily
            log(f"Downloading {url}")
            check_url(url)
            r = SESSION.get(url, timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
        return pd.read_csv(p)
    snaps = csv(f"{MIRROR}/{season}/playerstats.csv", f"m_{season}_playerstats.csv").rename(columns={"id": "player_id"})
    plist = csv(f"{MIRROR}/{season}/players.csv", f"m_{season}_players.csv")
    mteams = csv(f"{MIRROR}/{season}/teams.csv", f"m_{season}_teams.csv")
    prev_s = csv(f"{MIRROR}/{prev}/playerstats.csv", f"m_{prev}_playerstats.csv", volatile=False).rename(columns={"id": "pid_prev"})
    prev_p = csv(f"{MIRROR}/{prev}/players.csv", f"m_{prev}_players.csv", volatile=False).rename(columns={"player_id": "pid_prev"})
    fixtures = csv(f"{MIRROR_FIXTURES}/{fx_season}/fixtures.csv", f"v_{fx_season}_fixtures.csv")

    plist["pos"] = plist.position.map(POS_NAME)
    plist = plist.merge(mteams[["id", "code"]].rename(columns={"id": "team", "code": "team_code"}), on="team_code", how="left")
    snaps = snaps.sort_values(["player_id", "gw"])
    last_gw = int(snaps.gw.max())

    # snapshots are cumulative -> per-gameweek deltas
    g = snaps.groupby("player_id")
    per = snaps[["player_id", "gw"]].copy()
    for c in STAT_COLS:
        per[c] = snaps[c] - g[c].shift(1).fillna(0)
    per = per.merge(plist[["player_id", "player_code", "web_name", "pos", "team"]], on="player_id", how="left")
    per["match"] = per.gw                              # mirror has no fixture ids; doubles merge into one row

    def snap_players(gw):
        s = snaps[snaps.gw == gw]
        out = s[["player_id", "status", "chance_of_playing_next_round", "now_cost", "selected_by_percent",
                 "ep_next", "news"]].merge(plist[["player_id", "player_code", "web_name", "pos", "team",
                                                  "first_name", "second_name"]], on="player_id")
        out["full_name"] = out.first_name.fillna("") + " " + out.second_name.fillna("")
        out["news"] = out.news.fillna("")
        return out.drop(columns=["first_name", "second_name"])
    history = {gw: snap_players(gw) for gw in snaps.gw.unique()}

    prev_s = prev_s[prev_s.gw == prev_s.gw.max()].merge(prev_p[["player_code", "pid_prev"]], on="pid_prev")
    prior = prev_s[["player_code", "minutes", "expected_goals", "expected_assists", "saves",
                    "defensive_contribution", "bonus", "yellow_cards"]]
    teams = csv(f"{MIRROR_FIXTURES}/{fx_season}/teams.csv", f"v_{fx_season}_teams.csv")[["id", "short_name"]]
    fx = fixtures[["event", "team_h", "team_a", "kickoff_time"]].dropna(subset=["event"]).copy()
    fx["event"] = fx.event.astype(int)
    ko = pd.to_datetime(fx.loc[fx.event == last_gw + 1, "kickoff_time"], utc=True)
    deadline = (ko.min() - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ") if len(ko) else None
    return dict(players=history[last_gw], per=per, fixtures=fx, prior=prior, teams=teams, history=history,
                last_gw=last_gw, next_gw=last_gw + 1, deadline=deadline, source="GitHub mirror of the FPL API")


# ============================================================================ MODEL
def team_ratings(per, upto_gw):
    """Goals per game for and against, shrunk toward league average. Returns (table, league average)."""
    d = per[per.gw <= upto_gw]
    played = d.groupby(["team", "match"]).minutes.max().reset_index()
    played = played[played.minutes > 0][["team", "match"]]
    att = d.groupby(["team", "match"]).agg(xg=("expected_goals", "sum"), g=("goals_scored", "sum")).reset_index()
    gk = d[d.pos == "GKP"].sort_values("minutes").groupby(["team", "match"]).tail(1)
    gk = gk[["team", "match", "expected_goals_conceded", "goals_conceded"]].rename(
        columns={"expected_goals_conceded": "xga", "goals_conceded": "ga"})
    m = played.merge(att, on=["team", "match"], how="left").merge(gk, on=["team", "match"], how="left").fillna(0)
    t = m.groupby("team").agg(n=("match", "count"), xg=("xg", "sum"), g=("g", "sum"),
                              xga=("xga", "sum"), ga=("ga", "sum")).reset_index()
    t["for_raw"] = 0.75 * t.xg + 0.25 * t.g
    t["ag_raw"] = 0.75 * t.xga + 0.25 * t.ga
    lg = t.for_raw.sum() / t.n.sum()
    k = TEAM_SHRINK_GAMES
    t["att"] = (t.for_raw + k * lg) / (t.n + k)
    t["dfn"] = (t.ag_raw + k * lg) / (t.n + k)
    return t.set_index("team"), lg


def fixture_lambdas(tr, lg, home, away):
    """Expected goals for the home and away side in one match."""
    lh = tr.loc[home, "att"] * tr.loc[away, "dfn"] / lg * HOME_ADV
    la = tr.loc[away, "att"] * tr.loc[home, "dfn"] / lg / HOME_ADV
    return lh, la


def player_rates(per, status, prior, upto_gw, use_status=True):
    d = per[per.gw <= upto_gw].copy()
    tot = d.groupby("player_id").agg(
        mins=("minutes", "sum"), xg=("expected_goals", "sum"), xa=("expected_assists", "sum"),
        sv=("saves", "sum"), dc=("defensive_contribution", "sum"), bon=("bonus", "sum"),
        yc=("yellow_cards", "sum"), pts=("total_points", "sum")).reset_index()

    w = {upto_gw - i: wt for i, wt in enumerate(RECENCY)}
    d["w"] = d.gw.map(w).fillna(0)
    d["app"] = (d.minutes > 0).astype(float)
    d["m60"] = (d.minutes >= 60).astype(float)
    d["wm"], d["wa"], d["w6"] = d.w * d.minutes, d.w * d.app, d.w * d.m60
    mm = d.groupby("player_id")[["w", "wm", "wa", "w6"]].sum()
    den = mm.w.replace(0, np.nan)
    mm = pd.DataFrame({"exp_min": mm.wm / den, "p_app": mm.wa / den, "p60": mm.w6 / den}).fillna(0).reset_index()
    tot = tot.merge(mm, on="player_id")

    tot = tot.merge(status, on="player_id", how="inner")
    if use_status:
        avail = tot.chance_of_playing_next_round.fillna(100) / 100
        tot["avail"] = np.where(tot.status.isin(["u", "n"]), 0, avail)
    else:
        tot["avail"] = 1.0
    for c in ["exp_min", "p_app", "p60"]:
        tot[c] = tot[c] * tot.avail

    s = prior[prior.minutes >= 900].copy()
    for a, b in [("xg", "expected_goals"), ("xa", "expected_assists"), ("sv", "saves"),
                 ("dc", "defensive_contribution"), ("bon", "bonus"), ("yc", "yellow_cards")]:
        s[f"p_{a}"] = s[b] / s.minutes * 90
    tot = tot.merge(s[["player_code"] + [c for c in s.columns if c.startswith("p_")]], on="player_code", how="left")
    reg = tot[tot.mins >= 270]
    for a in ["xg", "xa", "sv", "dc", "bon", "yc"]:
        pos_med = (reg[a] / reg.mins * 90).groupby(reg.pos).median()
        pr = tot[f"p_{a}"].fillna(tot.pos.map(pos_med)).fillna(0)
        tot[f"r_{a}"] = (tot[a] + pr * PLAYER_PRIOR_MIN / 90) / ((tot.mins + PLAYER_PRIOR_MIN) / 90)
    return tot


def _pois_logpmf(lam, nmax):
    x = np.arange(nmax)[:, None]
    return x, -lam + x * np.log(np.maximum(lam, 1e-12)) - np.array([math.lgamma(i + 1) for i in range(nmax)])[:, None]


def poisson_floor_div_mean(lam, k, nmax=25):
    """E[floor(X/k)] for X ~ Poisson(lam): e.g. -1 per 2 goals conceded, +1 per 3 saves."""
    lam = np.asarray(lam, float)
    x, lp = _pois_logpmf(lam, nmax)
    return (np.exp(lp) * (x // k)).sum(0)


def poisson_sf(lam, thr):
    """P(X >= thr) for X ~ Poisson(lam)."""
    lam = np.asarray(lam, float)
    cdf, term = np.zeros_like(lam), np.exp(-lam)
    for i in range(thr):
        cdf += term
        term = term * lam / (i + 1)
    return 1 - cdf


def expected_points(pr, tr, lg, fixtures, gw):
    """One row per (player, fixture) in `gw`. A double gameweek gives two rows; a blank gives none."""
    rows = []
    for f in fixtures[fixtures.event == gw].itertuples():
        lh, la = fixture_lambdas(tr, lg, f.team_h, f.team_a)
        rows += [dict(team=f.team_h, opp=f.team_a, home=1, lam_for=lh, lam_ag=la),
                 dict(team=f.team_a, opp=f.team_h, home=0, lam_for=la, lam_ag=lh)]
    x = pr.merge(pd.DataFrame(rows), on="team", how="inner") if rows else pd.DataFrame()
    if x.empty:
        return x
    att_mult = x.lam_for / x.team.map(tr.att)     # this fixture vs the team's average
    opp_att = x.opp.map(tr.att)
    m90 = x.exp_min / 90
    gp, cp = x.pos.map(GOAL_PTS), x.pos.map(CS_PTS)

    x["e_goals"] = x.r_xg * m90 * att_mult
    x["e_ast"] = x.r_xa * m90 * att_mult
    x["p_cs"] = np.exp(-x.lam_ag) * x.p60
    gc_lam = x.lam_ag * np.clip(m90, 0, 1)
    x["pts_conceded"] = -poisson_floor_div_mean(gc_lam, 2) * x.pos.isin(["GKP", "DEF"])
    x["pts_saves"] = poisson_floor_div_mean(x.r_sv * m90 * (opp_att / lg), 3) * (x.pos == "GKP")
    x["p_defcon"] = 0.0
    for p, t in DEFCON_THR.items():
        if t:
            msk = x.pos == p
            per_app = 0.9 * x.loc[msk, "r_dc"] * m90[msk] / np.maximum(x.loc[msk, "p_app"], 1e-6)
            x.loc[msk, "p_defcon"] = poisson_sf(per_app, t) * x.loc[msk, "p_app"]
    x["pts_bonus"] = x.r_bon * m90 * (0.5 + 0.5 * att_mult)
    x["pts_cards"] = -x.r_yc * m90
    x["pts_minutes"] = (x.p_app - x.p60) * 1 + x.p60 * 2
    x["pts_goals"] = x.e_goals * gp
    x["pts_assists"] = x.e_ast * 3
    x["pts_cs"] = x.p_cs * cp
    x["pts_defcon"] = 2 * x.p_defcon
    x["xP"] = x[["pts_minutes", "pts_goals", "pts_assists", "pts_cs", "pts_conceded", "pts_saves",
                 "pts_defcon", "pts_bonus", "pts_cards"]].sum(axis=1)
    x["gw"] = gw
    return x


def simulate(rows, n=20000):
    """Monte Carlo of each player's gameweek score (doubles summed): p10, p90, P(10+), P(<=2)."""
    out = []
    for pid, grp in rows.groupby("player_id"):
        tot = np.zeros(n)
        for r in grp.itertuples():
            plays = RNG.random(n) < r.p_app
            full = plays & (RNG.random(n) < (r.p60 / r.p_app if r.p_app > 0 else 0))
            share = 1 / max(r.p_app, 1e-6)          # rates conditional on playing
            g = RNG.poisson(r.e_goals * share, n) * plays
            a = RNG.poisson(r.e_ast * share, n) * plays
            opp = RNG.poisson(r.lam_ag, n)
            s = plays * 1.0 + full * 1.0 + g * GOAL_PTS[r.pos] + a * 3 + ((opp == 0) & full) * CS_PTS[r.pos]
            if r.pos in ("GKP", "DEF"):
                s -= (opp // 2) * full
            s += 2 * ((RNG.random(n) < r.p_defcon * share) & plays)
            s += (np.minimum(RNG.poisson(r.pts_bonus * share, n), 3) - (RNG.random(n) < -r.pts_cards * share)) * plays
            s += r.pts_saves * share * plays
            tot += s
        out.append(dict(player_id=pid, p10=np.percentile(tot, 10), p90=np.percentile(tot, 90),
                        p_haul=(tot >= 10).mean(), p_blank=(tot <= 2).mean()))
    return pd.DataFrame(out)


# ============================================================================ BACKTEST
def backtest(D, gws):
    res = []
    for gw in gws:
        status = D["history"][gw - 1] if D["history"] else D["players"]
        st = status[["player_id", "player_code", "web_name", "pos", "team", "status",
                     "chance_of_playing_next_round", "now_cost", "selected_by_percent", "ep_next", "news"]]
        tr, lg = team_ratings(D["per"], gw - 1)
        pr = player_rates(D["per"], st, D["prior"], gw - 1, use_status=D["history"] is not None)
        xp = expected_points(pr, tr, lg, D["fixtures"], gw)
        if xp.empty:
            continue
        xp = xp.groupby("player_id").agg(xP=("xP", "sum"), ep_next=("ep_next", "first"),
                                         exp_min=("exp_min", "first")).reset_index()
        per = D["per"]
        xp["form"] = xp.player_id.map(per[(per.gw <= gw - 1) & (per.gw >= gw - 4)].groupby("player_id").total_points.mean())
        xp["actual"] = xp.player_id.map(per[per.gw == gw].groupby("player_id").total_points.sum())
        methods = ["xP", "form"] + (["ep_next"] if D["history"] is not None else [])
        xp = xp[xp.exp_min > 20].dropna(subset=["actual"] + methods)
        r = {"gw": int(gw), "n": int(len(xp))}
        for c in methods:
            r[f"mae_{c}"] = float((xp[c] - xp.actual).abs().mean())
            r[f"spearman_{c}"] = float(xp[c].rank().corr(xp.actual.rank()))  # Spearman without needing scipy
            r[f"top20_{c}"] = float(xp.nlargest(20, c).actual.mean())
        res.append(r)
    return pd.DataFrame(res)


# ============================================================================ OPTIMISER
def optimise(proj, budget, current=None, free_transfers=1, max_hits=None):
    """
    Maximise discounted xP of XI (+ captain bonus, + a little bench value) over the horizon.
    current: set of player_ids you own -> transfers into players outside it cost a slot;
             transfers beyond `free_transfers` cost HIT_COST points each.
    """
    gws = sorted(int(c[5:]) for c in proj.columns if c.startswith("xP_gw"))
    P = proj.reset_index(drop=True)
    I = list(P.index)
    xp = {(i, g): float(P.at[i, f"xP_gw{g}"]) for i in I for g in gws}
    w = {g: HORIZON_DECAY ** k for k, g in enumerate(gws)}
    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    sq = {i: pulp.LpVariable(f"sq_{i}", cat="Binary") for i in I}
    xi = {(i, g): pulp.LpVariable(f"xi_{i}_{g}", cat="Binary") for i in I for g in gws}
    cap = {(i, g): pulp.LpVariable(f"cap_{i}_{g}", cat="Binary") for i in I for g in gws}
    obj = pulp.lpSum(w[g] * xp[i, g] * (xi[i, g] + cap[i, g] + BENCH_WEIGHT * (sq[i] - xi[i, g]))
                     for i in I for g in gws)
    hits = None
    if current is not None:
        owned = P.player_id.isin(current)
        n_in = pulp.lpSum(sq[i] for i in I if not owned[i])
        hits = pulp.LpVariable("hits", lowBound=0, cat="Integer")
        prob += hits >= n_in - free_transfers
        if max_hits is not None:
            prob += hits <= max_hits
        obj -= HIT_COST * hits
    prob += obj
    prob += pulp.lpSum(P.at[i, "now_cost"] * sq[i] for i in I) <= budget + 1e-6
    for pos, n in SQUAD_SHAPE.items():
        prob += pulp.lpSum(sq[i] for i in I if P.at[i, "pos"] == pos) == n
    for t in P.team.unique():
        prob += pulp.lpSum(sq[i] for i in I if P.at[i, "team"] == t) <= 3
    for g in gws:
        prob += pulp.lpSum(xi[i, g] for i in I) == 11
        prob += pulp.lpSum(cap[i, g] for i in I) == 1
        for pos, (lo, hi) in XI_SHAPE.items():
            s = pulp.lpSum(xi[i, g] for i in I if P.at[i, "pos"] == pos)
            prob += s >= lo
            prob += s <= hi
        for i in I:
            prob += xi[i, g] <= sq[i]
            prob += cap[i, g] <= xi[i, g]
    solver = pulp.HiGHS(msg=False, timeLimit=120) if "HiGHS" in pulp.listSolvers(onlyAvailable=True) \
        else pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    pick = P[[sq[i].value() > 0.5 for i in I]].copy()
    pick["starts"] = [[g for g in gws if xi[i, g].value() > 0.5] for i in pick.index]
    pick["captain"] = [[g for g in gws if cap[i, g].value() > 0.5] for i in pick.index]
    return pick, status, (int(round(hits.value())) if hits is not None else 0)


# ============================================================================ DASHBOARD + PLAYER LOOKUP
COMP_COLS = ["pts_minutes", "pts_goals", "pts_assists", "pts_cs", "pts_conceded", "pts_saves",
             "pts_defcon", "pts_bonus", "pts_cards"]


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python, floats rounded."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 3)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _norm(txt):
    return "".join(c for c in unicodedata.normalize("NFKD", str(txt)) if not unicodedata.combining(c)).casefold().strip()


def find_player(proj, query):
    """Match a typed name ('saka', 'Bruno', 'Gross') to a player row, accent- and case-insensitive."""
    q = _norm(query)
    names = proj.web_name.map(_norm)
    full = proj.full_name.map(_norm)
    hit = proj[(names == q) | (full == q)]
    if hit.empty:
        hit = proj[names.str.contains(q, regex=False) | full.str.contains(q, regex=False)]
    if hit.empty:
        close = difflib.get_close_matches(q, list(names) + list(full), n=1, cutoff=0.6)
        if close:
            hit = proj[(names == close[0]) | (full == close[0])]
    return hit.sort_values("exp_min", ascending=False)


def build_dashboard_data(D, proj, allx, nxt, pr, tr, tname, squad, bt, next_gw, hcols, a, current, hits, budget):
    per, fx = D["per"], D["fixtures"]
    # opponent for every past match, from the fixture list
    opp_map = {}
    for f in fx.itertuples():
        opp_map[(f.event, f.team_h)] = (f.team_a, 1)
        opp_map[(f.event, f.team_a)] = (f.team_h, 0)
    season = per.groupby("player_id").agg(
        mins=("minutes", "sum"), pts=("total_points", "sum"), g=("goals_scored", "sum"), a=("assists", "sum"),
        xg=("expected_goals", "sum"), xa=("expected_assists", "sum"), cs=("clean_sheets", "sum"),
        bonus=("bonus", "sum"), dc=("defensive_contribution", "sum"), starts=("starts", "sum"))
    hist = {}
    for r in per.sort_values("gw").itertuples():
        o, h = opp_map.get((r.gw, r.team), (None, None))
        hist.setdefault(r.player_id, []).append(
            [int(r.gw), tname.get(o, "") if o else "", h, int(r.minutes), int(r.total_points),
             round(float(r.expected_goals), 2), round(float(r.expected_assists), 2)])
    comp = nxt.groupby("player_id")[COMP_COLS].sum()

    # per-90 rates and their percentile among regulars (270+ minutes) in the same position
    rates = pr.set_index("player_id")[["pos", "mins", "r_xg", "r_xa", "r_dc", "r_bon", "r_sv", "exp_min"]].copy()
    reg = rates[rates.mins >= 270]
    pct = {}
    for col in ["r_xg", "r_xa", "r_dc", "r_bon", "r_sv", "exp_min"]:
        for pos, grp in reg.groupby("pos"):
            vals = np.sort(grp[col].values)
            for pid in rates.index[rates.pos == pos]:
                pct.setdefault(pid, {})[col] = float(np.searchsorted(vals, rates.at[pid, col], side="right") / max(len(vals), 1))
    full = D["players"].set_index("player_id").full_name if "full_name" in D["players"] else pd.Series(dtype=str)

    keep = proj[(proj.exp_min > 0) | (proj.mins.fillna(0) > 0) |
                (proj.player_id.isin(current or [])) | proj.player_id.isin(squad.player_id)]
    players = []
    for r in keep.itertuples():
        pid = r.player_id
        s = season.loc[pid] if pid in season.index else None
        rt = rates.loc[pid] if pid in rates.index else None
        players.append({
            "id": pid, "n": r.web_name, "full": full.get(pid, r.web_name), "pos": r.pos, "t": r.team_s,
            "tid": r.team, "price": r.now_cost, "own": r.selected_by_percent, "status": r.status,
            "news": r.news or "", "chance": r.chance_of_playing_next_round, "fpl_ep": r.ep_next,
            "xP": r.xP, "opp": r.opp, "p10": r.p10, "p90": r.p90, "haul": r.p_haul, "blank": r.p_blank,
            "emin": r.exp_min, "p60": r.p60, "h": [getattr(r, c) for c in hcols], "xPh": r.xP_horizon,
            "season": None if s is None else {k: s[k] for k in season.columns},
            "rates": None if rt is None else {"xg": rt.r_xg, "xa": rt.r_xa, "dc": rt.r_dc, "bon": rt.r_bon, "sv": rt.r_sv},
            "pct": pct.get(pid),
            "comp": None if pid not in comp.index else {c[4:]: comp.at[pid, c] for c in COMP_COLS},
            "hist": hist.get(pid, []),
        })

    ticker = []
    for x in allx.drop_duplicates(["team", "gw", "opp"]).itertuples():
        ticker.append({"t": tname.get(x.team), "gw": x.gw, "opp": tname.get(x.opp), "home": x.home,
                       "for": x.lam_for, "ag": x.lam_ag, "cs": math.exp(-x.lam_ag)})
    teams = [{"t": tname.get(t), "n": int(r.n), "att": r.att, "dfn": r.dfn, "xg": r.xg, "g": r.g, "xga": r.xga, "ga": r.ga}
             for t, r in tr.iterrows()]
    first = f"xP_gw{next_gw}"
    starters = squad[squad.starts.apply(lambda v: next_gw in v)]
    vice = starters[~starters.captain.apply(lambda v: next_gw in v)].sort_values(first, ascending=False).player_id.iloc[0]
    sq = [{"id": r.player_id, "start": next_gw in r.starts, "cap": next_gw in r.captain, "vice": r.player_id == vice}
          for r in squad.itertuples()]
    transfers = []
    if current:
        outs = proj[proj.player_id.isin(set(current) - set(squad.player_id))].sort_values("pos")
        ins = squad[~squad.player_id.isin(current)].sort_values("pos")
        transfers = [{"out": o, "in": i} for o, i in zip(outs.player_id, ins.player_id)]
    return _clean({
        "meta": {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": D["source"],
                 "last_gw": next_gw - 1, "next_gw": next_gw, "gws": list(range(next_gw, next_gw + len(hcols))),
                 "deadline": D.get("deadline"), "mode": "team" if current else "scratch",
                 "free_transfers": a.free_transfers, "hits": hits, "budget": budget,
                 "bank": getattr(a, "bank_used", None), "sell": getattr(a, "sell_used", {}),
                 "rerun": rerun_command(a),
                 "league_goals": float(tr.for_raw.sum() / tr.n.sum())},
        "players": players, "squad": sq, "transfers": transfers, "ticker": ticker, "teams": teams,
        "backtest": [] if bt is None else bt.to_dict("records"),
    })


def open_in_browser(path: Path):
    """Open the dashboard. Under WSL, hand it to the Windows default browser via explorer.exe."""
    try:
        in_wsl = "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        in_wsl = False
    if in_wsl:
        import subprocess  # fixed argv, no shell: the path is never interpreted as a command
        try:
            win = subprocess.run(["wslpath", "-w", str(path.resolve())], capture_output=True,
                                 text=True, check=True).stdout.strip()
            subprocess.run(["explorer.exe", win], check=False)  # explorer returns 1 even on success
            return
        except (OSError, subprocess.CalledProcessError):
            log("Couldn't open a Windows browser from WSL; open the dashboard path above yourself.")
            return
    webbrowser.open(path.resolve().as_uri())


def write_dashboard(data, path: Path):
    tpl = Path(__file__).with_name("dashboard_template.html")
    if not tpl.exists():
        log(f"dashboard_template.html not found next to {Path(__file__).name}; skipping dashboard")
        return None
    # '<' escaped so no data value (e.g. injury news text) can ever close the <script> tag
    blob = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    path.write_text(tpl.read_text(encoding="utf-8").replace("__FPL_DATA__", blob), encoding="utf-8")
    return path


def print_profile(proj, row, nxt, next_gw, hcols):
    r = row
    pos_pool = proj[(proj.pos == r.pos) & (proj.exp_min > 45)]
    rank = int((pos_pool.xP_horizon > r.xP_horizon).sum()) + 1
    print(f"\n{r.web_name}  ({r.full_name.strip()})  {r.pos} · {r.team_s} · £{r.now_cost:.1f}m · owned {r.selected_by_percent:.1f}%")
    if r.news:
        print(f"  News: {r.news}")
    print(f"  Next GW{next_gw}: {r.opp}  xP {r.xP:.2f}   floor {r.p10 if pd.notna(r.p10) else 0:.0f} / "
          f"ceiling {r.p90 if pd.notna(r.p90) else 0:.0f}   haul (10+) {0 if pd.isna(r.p_haul) else r.p_haul:.0%}   "
          f"expected minutes {r.exp_min:.0f}")
    print("  Next gameweeks: " + "  ".join(f"GW{c[5:]} {getattr(r, c):.1f}" for c in hcols) +
          f"   = {r.xP_horizon:.1f} xP  (#{rank} of {len(pos_pool)} {r.pos}s)")
    c = nxt[nxt.player_id == r.player_id][COMP_COLS].sum()
    print("  Where GW points come from: " + ", ".join(f"{k[4:]} {v:+.2f}" for k, v in c.items() if abs(v) >= 0.05))
    alts = proj[(proj.pos == r.pos) & (proj.player_id != r.player_id) & (proj.now_cost <= r.now_cost + 0.5)
                & (proj.xP_horizon > r.xP_horizon) & (proj.exp_min > 45)].sort_values("xP_horizon", ascending=False).head(5)
    if len(alts):
        print("  Better value at or under his price (+£0.5m): " +
              ", ".join(f"{x.web_name} £{x.now_cost:.1f}m {x.xP_horizon:.1f}" for x in alts.itertuples()))
    else:
        print("  No player in his position at or under his price projects more over the horizon.")


# ============================================================================ MAIN
def fetch_entry(entry, gw, base=FPL_API):
    """Your public team: picks of the last finished GW + money in the bank (£m)."""
    js = get_json(f"{base}/entry/{entry}/event/{gw}/picks/")
    return [p["element"] for p in js["picks"]], js["entry_history"]["bank"] / 10


# ============================================================================ TEAM FILE
TEAM_KEYS = {"bank": float, "free_transfers": int, "entry": int, "max_hits": int}


def read_team_file(path):
    """
    Plain-text team description, one item per line ('#' starts a comment):
        bank = 0.1
        free_transfers = 2
        entry = 1234567            (optional: read the 15 from FPL instead of listing them)
        Haaland
        Calafiori = 5.9            (optional: your selling price, from the FPL Transfers page)
        Thomas (COV)               (club code in brackets when a name is ambiguous)
    Parsed as text only; nothing in the file is ever executed.
    """
    cfg = {"names": [], "sell": {}}
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, val = (s.strip() for s in line.split("=", 1)) if "=" in line else (line, None)
        k = key.lower().replace(" ", "_")
        try:
            if k in TEAM_KEYS:
                cfg[k] = TEAM_KEYS[k](val)
            else:
                cfg["names"].append(key)
                if val:
                    cfg["sell"][key] = float(val)
        except (TypeError, ValueError):
            sys.exit(f"{path}, line {n}: can't read '{raw.strip()}'. Use 'Name' or 'Name = 5.9'.")
    return cfg


def resolve_names(proj, names):
    """Typed names -> player_ids. 'Thomas (COV)' or 'Thomas DEF' pins down duplicates."""
    ids = {}
    for name in names:
        q, club, pos = name, None, None
        if "(" in q and q.endswith(")"):
            q, club = q[:-1].split("(", 1)
            q, club = q.strip(), club.strip().upper()
        parts = q.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].upper() in ("GKP", "GK", "DEF", "MID", "FWD"):
            q, pos = parts[0], {"GK": "GKP"}.get(parts[1].upper(), parts[1].upper())
        hit = find_player(proj, q)
        if club:
            hit = hit[hit.team_s.str.upper() == club]
        if pos:
            hit = hit[hit.pos == pos]
        if hit.empty:
            sys.exit(f"Can't find a player called '{name}'. Check the spelling (as shown in the FPL app).")
        if len(hit) > 1:
            pick = hit.iloc[0]
            log(f"'{name}' matches {len(hit)} players; using {pick.web_name} ({pick.team_s}, {pick.pos}). "
                f"Others: " + ", ".join(f"{x.web_name} ({x.team_s})" for x in hit.iloc[1:4].itertuples())
                + ". Add the club in brackets to be sure.")
        ids[name] = int(hit.iloc[0].player_id)
    return ids


def rerun_command(a):
    """The command that rebuilds this dashboard with fresh data (shown on the page)."""
    cmd = ["python", "fpl_model.py"]
    if getattr(a, "team_file", None):
        cmd += ["--team-file", Path(a.team_file).name]
    elif a.entry:
        cmd += ["--entry", str(a.entry), "--free-transfers", str(a.free_transfers)]
    # always the live-data command; update.command / update.bat fall back to the mirror on their own
    return " ".join(cmd + ["--refresh", "--open"])


def main():
    ap = argparse.ArgumentParser(description="FPL expected-points model and optimiser")
    ap.add_argument("--source", choices=["api", "mirror"], default="api",
                    help="api = official FPL site (live, default); mirror = weekly GitHub copy")
    ap.add_argument("--api-base", default=FPL_API, help=argparse.SUPPRESS)
    ap.add_argument("--entry", type=int, help="your FPL team ID (number in the URL of your Points page)")
    ap.add_argument("--team-file", metavar="PATH",
                    help="text file with your 15 players, bank, free transfers and selling prices (see my_team.txt)")
    ap.add_argument("--free-transfers", type=int)
    ap.add_argument("--max-hits", type=int, help="max -4 transfers allowed (default 0)")
    ap.add_argument("--bank", type=float, help="override money in the bank, £m")
    ap.add_argument("--horizon", type=int, default=5, help="gameweeks to plan over")
    ap.add_argument("--budget", type=float, default=100.0, help="budget for the from-scratch squad")
    ap.add_argument("--gw", type=int, help="gameweek to predict (default: the next one)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default=".fpl_cache")
    ap.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--player", action="append", metavar="NAME",
                    help="print a profile for a player, e.g. --player Saka (repeatable)")
    ap.add_argument("--open", action="store_true", help="open the dashboard in your browser when done")
    ap.add_argument("--no-dashboard", action="store_true")
    a = ap.parse_args()
    cfg = read_team_file(a.team_file) if a.team_file else {"names": [], "sell": {}}
    # command-line values win over the team file; the team file wins over defaults
    a.entry = a.entry or cfg.get("entry")
    a.bank = a.bank if a.bank is not None else cfg.get("bank")
    a.free_transfers = a.free_transfers if a.free_transfers is not None else cfg.get("free_transfers", 1)
    a.max_hits = a.max_hits if a.max_hits is not None else cfg.get("max_hits", 0)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load_api(Path(a.cache), a.api_base, a.refresh) if a.source == "api" else load_mirror(Path(a.cache), a.refresh)
    next_gw = a.gw or D["next_gw"]
    last_gw = next_gw - 1
    tname = D["teams"].set_index("id").short_name
    log(f"Data through GW{last_gw}; predicting GW{next_gw}-{next_gw + a.horizon - 1}")

    bt = None
    if not a.no_backtest and last_gw >= 3:
        bt = backtest(D, range(3, last_gw + 1))
        bt.to_csv(out / "backtest.csv", index=False)
        print("\nBACKTEST (lower MAE = better; top20 = avg actual points of each method's top 20 picks)")
        print(bt.round(2).to_string(index=False))

    # ---- fit on everything up to last_gw, project the horizon
    tr, lg = team_ratings(D["per"], last_gw)
    status = D["players"][["player_id", "player_code", "web_name", "pos", "team", "status",
                           "chance_of_playing_next_round", "now_cost", "selected_by_percent", "ep_next", "news"]]
    pr = player_rates(D["per"], status, D["prior"], last_gw)
    allx = pd.concat([e for g in range(next_gw, next_gw + a.horizon)
                      if not (e := expected_points(pr, tr, lg, D["fixtures"], g)).empty])
    allx["opp_s"] = allx.opp.map(tname) + np.where(allx.home == 1, " (H)", " (A)")

    nxt = allx[allx.gw == next_gw]
    sim = simulate(nxt[nxt.exp_min > 30])
    g1 = nxt.groupby("player_id").agg(xP=("xP", "sum"), opp=("opp_s", ", ".join), p_cs=("p_cs", "sum"),
                                      e_goals=("e_goals", "sum"), e_ast=("e_ast", "sum")).reset_index()
    wide = allx.groupby(["player_id", "gw"]).xP.sum().unstack()
    wide = wide.reindex(columns=range(next_gw, next_gw + a.horizon)).fillna(0)
    wide.columns = [f"xP_gw{c}" for c in wide.columns]
    base = status.merge(pr[["player_id", "exp_min", "p60", "mins", "pts"]], on="player_id", how="left")
    proj = base.merge(g1, on="player_id", how="left").merge(sim, on="player_id", how="left") \
               .merge(wide, on="player_id", how="left")
    hcols = list(wide.columns)
    proj[hcols] = proj[hcols].fillna(0)
    proj["xP"] = proj.xP.fillna(0)
    proj["exp_min"] = proj.exp_min.fillna(0)
    proj["team_s"] = proj.team.map(tname)
    proj["opp"] = proj.opp.fillna("no game")
    proj["xP_horizon"] = proj[hcols].sum(axis=1)
    proj["full_name"] = proj.player_id.map(D["players"].set_index("player_id").get("full_name", pd.Series(dtype=str)))
    proj["full_name"] = proj.full_name.fillna(proj.web_name)
    proj = proj.sort_values("xP", ascending=False)
    proj.to_csv(out / f"projections_gw{next_gw}.csv", index=False)

    comp = ["pts_minutes", "pts_goals", "pts_assists", "pts_cs", "pts_conceded", "pts_saves",
            "pts_defcon", "pts_bonus", "pts_cards"]
    expl = nxt.groupby(["player_id", "web_name", "pos"]).agg(
        opp=("opp_s", ", ".join), exp_min=("exp_min", "first"), xg_per90=("r_xg", "first"),
        xa_per90=("r_xa", "first"), team_goals=("lam_for", "sum"), opp_goals=("lam_ag", "sum"),
        **{c: (c, "sum") for c in comp}, xP=("xP", "sum")).reset_index().sort_values("xP", ascending=False)
    expl.round(3).to_csv(out / f"explain_gw{next_gw}.csv", index=False)

    # ---- optimise
    pool = proj[proj.exp_min > 15]
    current, budget = None, a.budget
    a.sell_used, a.bank_used = {}, None
    if a.entry or cfg["names"]:
        named = resolve_names(proj, cfg["names"]) if cfg["names"] else {}
        if cfg["names"] and not a.entry:
            current, bank = list(named.values()), (a.bank if a.bank is not None else 0.0)
            if len(set(current)) != 15:
                log(f"Note: the team file lists {len(set(current))} different players, not 15.")
        else:
            current, bank = fetch_entry(a.entry, last_gw, a.api_base)
            bank = a.bank if a.bank is not None else bank
        a.sell_used = {named[n]: p for n, p in cfg["sell"].items() if named.get(n) in current}
        pool = pd.concat([pool, proj[proj.player_id.isin(current)]]).drop_duplicates("player_id").copy()
        # owned players are worth their SELLING price to the optimiser; everyone else costs the market price
        sp = pool.player_id.map(a.sell_used)
        pool["now_cost"] = sp.fillna(pool.now_cost)
        mine = pool[pool.player_id.isin(current)]
        budget, a.bank_used = float(mine.now_cost.sum() + bank), float(bank)
        who = f"Team {a.entry}" if a.entry else f"Team from {Path(a.team_file).name}"
        log(f"{who}: squad value £{mine.now_cost.sum():.1f}m + bank £{bank:.1f}m, "
            f"{a.free_transfers} free transfer(s), selling prices given for {len(a.sell_used)} player(s)"
            + ("" if a.sell_used else " (current prices used; your real selling prices may be £0.1-0.3m lower)"))
    squad, st, hits = optimise(pool, budget, set(current) if current else None, a.free_transfers, a.max_hits)
    squad.to_csv(out / f"squad_gw{next_gw}.csv", index=False)

    # ---- report
    cols = ["web_name", "pos", "team_s", "now_cost", "opp", "xP", "p10", "p90", "p_haul", "ep_next", "exp_min", "xP_horizon"]
    print(f"\nTOP 20 FOR GW{next_gw}  (xP = expected points; p10/p90 = simulated floor/ceiling; p_haul = P(10+))")
    print(proj[cols].head(20).round(2).to_string(index=False))

    if current is not None:
        sell = pool[pool.player_id.isin(set(current) - set(squad.player_id))]
        buy = squad[~squad.player_id.isin(current)]
        left = budget - squad.now_cost.sum()
        print(f"\nTRANSFERS  ({st}; hits taken: {hits}, cost {hits * HIT_COST} pts; bank after: £{left:.1f}m)")
        if buy.empty:
            print("  Roll the transfer: no move beats keeping your team over the horizon.")
        for (_, o), (_, i) in zip(sell.sort_values("pos").iterrows(), buy.sort_values("pos").iterrows()):
            print(f"  OUT {o.web_name:<16} £{o.now_cost:>4.1f}m  {o.xP_horizon:5.1f} xP  ->  "
                  f"IN {i.web_name:<16} £{i.now_cost:>4.1f}m  {i.xP_horizon:5.1f} xP")
    else:
        print(f"\nBEST SQUAD FROM SCRATCH (£{squad.now_cost.sum():.1f}m of £{budget:.1f}m, {st})")

    first = f"xP_gw{next_gw}"
    sq = squad.assign(start=squad.starts.apply(lambda s: next_gw in s),
                      cap=squad.captain.apply(lambda s: next_gw in s))
    sq["pos_o"] = sq.pos.map({"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3})
    sq = sq.sort_values(["start", "pos_o", first], ascending=[False, True, False])
    print(f"\nGW{next_gw} LINEUP  (C = captain, * = vice)")
    vice = sq[sq.start & ~sq.cap].sort_values(first, ascending=False).web_name.iloc[0]
    for r in sq.itertuples():
        tag = "C" if r.cap else ("*" if r.web_name == vice else " ")
        bench = "" if r.start else "  (bench)"
        print(f"  {tag} {r.pos} {r.web_name:<16} {r.team_s} {r.opp or 'no game':<14} xP {getattr(r, first):4.1f}{bench}")

    for q in a.player or []:
        hit = find_player(proj, q)
        if hit.empty:
            print(f"\nNo player matching '{q}'.")
            continue
        if len(hit) > 1:
            print(f"\n'{q}' matches {len(hit)} players; showing the most-played. Others: " +
                  ", ".join(f"{x.web_name} ({x.team_s})" for x in hit.iloc[1:6].itertuples()))
        print_profile(proj, hit.iloc[0], nxt, next_gw, hcols)

    if not a.no_dashboard:
        data = build_dashboard_data(D, proj, allx, nxt, pr, tr, tname, squad, bt, next_gw, hcols, a,
                                    current, hits, budget)
        path = write_dashboard(data, out / "dashboard.html")
        if path:
            print(f"\nDashboard: {path.resolve()}")
            if a.open:
                open_in_browser(path)
    print(f"Files written to {out.resolve()}")


if __name__ == "__main__":
    main()
