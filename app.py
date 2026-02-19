import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ================================
# VOS CONFIGS (copie quasi identique)
# ================================
METRICS_BY_POSTE = {
    "avant-centre": [
        ("Finition", "score_finition"),
        ("Impact", "score_impact"),
        ("Dribble", "score_dribble"),
        ("Déplacement", "score_deplacement"),
        ("Passe", "score_passe"),
        ("Défense", "score_defense"),
    ],
    "ailier - offensif": [
        ("Dribble","score_dribble"),
        ("Passe","score_passe"),
        ("Finition","score_finition"),
        ("Présence","score_presence"),
        ("Impact","score_impact"),
        ("Défense","score_defense"),
    ],
    "milieu": [
        ("Présence","score_presence"),
        ("Projection","score_projection"),
        ("Impact","score_impact"),
        ("Récuperation","score_recuperation"),
        ("Danger","score_danger"),
        ("Technique","score_technique"),
    ],
    "piston": [
        ("Défense","score_defense"),
        ("Relance","score_relance"),
        ("Progression","score_progression"),
        ("Projection","score_projection"),
        ("Centre","score_centre"),
        ("Impact","score_impact"),
    ],
    "defenseur": [
        ("Duel sol","score_duel_def"),
        ("Intervention","score_intervention"),
        ("Relance","score_relance"),
        ("Duel aérien","score_impact"),
        ("Projection","score_projection"),
        ("Discipline","score_discipline"),
    ],
}

VAR_LABELS = {
 "pct_block_pass_p90" : "Passes contrées (90m)",
 "pct_block_shot_p90" : "Tirs contrés (90m)",
 "pct_carry_lost_pct" : "Conduites de balle perdues (%)",
 "pct_carry_prog_p90" : "Conduites progressives (90m)",
 "pct_challenge_att_p90" : "Duels défensifs disputés (90m)",
 "pct_challenge_pct" : "Duels défensifs gagnés (%)",
 "pct_degagement_p90" : "Dégagements (90m)",
 "pct_duel_pct" : "Duels aériens gagnés (%)",
 "pct_duel_play_p90" : "Duels aériens disputés (90m)",
 "pct_erreur_opp_shot_p90" : "Erreur menant à un tir adverse (90m)",
 "pct_faute_commise_p90" : "Fautes commises (90m)",
 "pct_interception_p90" : "Interceptions (90m)",
 "pct_pass_long_pct" : "Passes longues réussies (%)",
 "pct_pass_med_pct"	: "Passes moyennes réussies (%)",
 "pct_pass_prog_p90" : "Passes progressives (90m)",
 "pct_pass_short_pct" : "Passes courtes réussies (%)",
 "pct_tacle_p90" : "Tacles tentés (90m)",
 "pct_tacle_win_ball_pct" : "Tacles réussis (%)",
 "pct_yellow_card_p90" : "Carton jaune reçu (90m)",
 "pct_ballon_touche_surface_off_p90" : "Ballons touchés surface adverse (90m)",
 "pct_carry_in_surface_p90"	: "Conduite surface adverse (90m)",
 "pct_dribble_att_p90" : "Dribbles tentés (90m)",
 "pct_dribble_pct" : "Dribbles réussis (%)",
 "pct_faute_won_p90" : "Fautes provoquées (90m)",
 "pct_goals_no_pk_p90" : "Buts inscrits hors penalty (90m)",
 "pct_offside_p90" : "Hors-jeu (90m)",
 "pct_pass_profondeur_defense_p90" : "Passes dos de la défense (90m)",
 "pct_pass_prog_recu_p90" : "Passes progressives reçues (90m)",
 "pct_pass_shot_p90" : "Passes amenant à un tir (90m)",
 "pct_pass_surface_cmp_p90"	: "Passes réussies dans la surface adverse (90m)",
 "pct_passe_pct" : "Passes réussies (%)",
 "pct_shot_on_target_pct" : "Tirs cadrés (%)",
 "pct_xG_per_shot" : "xG par tir",
 "pct_xGot_per_shot_on_target" : "xGot par tir cadré",
 "pct_fautes_net_pct" : "Ratio fautes subies & commises",
 "pct_passe_att_p90" : "Passes tentées (90m)",
 "pct_shots_p90" : "Tirs tentés (90m)",
 "pct_touch_med_p90" : "Ballons touchés milieu (90m)",
 "pct_touch_off_p90" : "Ballons touchés offensif (90m)",
 "pct_centre_p90" : "Centres tentés (90m)",
 "pct_centre_surface_cmp_p90" : "Centres réussies dans la surface adverse (90m)",
 "pct_pass_transversal_p90" : "Passes transversales (90m)"
}

LOWER_IS_BETTER = {
    "pct_yellow_card_p90",
    "pct_faute_commise_p90",
    "pct_carry_lost_pct",
    "pct_erreur_opp_shot_p90",
    "pct_offside_p90",
}

# ================================
# STREAMLIT SETUP
# ================================
st.set_page_config(page_title="Performance & Profil", layout="wide")

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # normaliser poste
    if "poste" in df.columns:
        df["poste"] = df["poste"].astype(str).str.strip().str.lower()
    if "player" in df.columns:
        df["player"] = df["player"].astype(str).str.strip()
    if "minutes" in df.columns:
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    return df

df = load_csv("data/data.csv")

# ================================
# LOGIQUE RADAR (Plotly)
# ================================
def build_radar_values(row: pd.Series):
    poste = str(row.get("poste", "")).strip().lower()
    if poste not in METRICS_BY_POSTE:
        # fallback simple : essaye de matcher partiellement
        found = None
        for k in METRICS_BY_POSTE.keys():
            if k in poste:
                found = k
                break
        poste = found or "defenseur"

    metrics = METRICS_BY_POSTE[poste]
    labels = [lab for lab, _ in metrics]
    pct_cols = [col + "_pct" for _, col in metrics]

    values = []
    for c in pct_cols:
        v = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        values.append(float(v) if pd.notna(v) else 0.0)

    values = np.clip(values, 0, 100)
    return poste, labels, values

def radar_plotly(labels, values, title=None):
    # fermer le polygone
    theta = labels + labels[:1]
    r = list(values) + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        title=title
    )
    return fig

# ================================
# FORCES / FAIBLESSES
# ================================
def strengths_weaknesses_from_row(row: pd.Series, max_items=5, high_thr=10, low_thr=90):
    pct_cols = [c for c in row.index if str(c).startswith("pct_")]

    items = []
    for c in pct_cols:
        raw = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        if pd.isna(raw):
            continue
        # normaliser "plus haut = mieux"
        score = 100.0 - float(raw) if c in LOWER_IS_BETTER else float(raw)
        items.append((c, score, float(raw)))

    if not items:
        return [], []

    # forces = score élevé ; faiblesses = score faible
    forces = [x for x in items if x[1] >= high_thr]
    faiblesses = [x for x in items if x[1] <= low_thr]

    forces.sort(key=lambda x: x[1], reverse=True)
    faiblesses.sort(key=lambda x: x[1])

    def label(col):
        return VAR_LABELS.get(col, col.replace("pct_", "").replace("_", " "))

    forces_lbl = [f"{label(c)}" for c, _, _ in forces[:max_items]]
    faib_lbl = [f"{label(c)}" for c, _, _ in faiblesses[:max_items]]

    return forces_lbl, faib_lbl

# ================================
# UI
# ================================
st.title("Performance & profil – Ligue 1 – 2024/2025")

c1, c2 = st.columns([2, 1])
with c1:
    player = st.selectbox("Joueur", sorted(df["player"].unique()))
with c2:
    min_minutes = st.number_input("Minutes min", min_value=0, value=0, step=100)

dff = df[df["minutes"].fillna(0) >= min_minutes].copy()
row_df = dff[dff["player"] == player]
if row_df.empty:
    st.warning("Aucun joueur ne correspond au filtre minutes.")
    st.stop()

row = row_df.iloc[0]
poste, labels, values = build_radar_values(row)

left, right = st.columns([1.15, 1])

with left:
    st.subheader(poste)

    k1, k2 = st.columns(2)
    score_moyen = float(np.mean(values)) if len(values) else np.nan

    with k1:
        st.metric("Score moyen", f"{score_moyen:.2f}" if not np.isnan(score_moyen) else "—")
    with k2:
        if "global_rank" in row.index and "global_n" in row.index:
            try:
                st.metric("Classement", f"{int(row['global_rank'])} / {int(row['global_n'])}")
            except Exception:
                st.metric("Classement", "—")
        else:
            st.metric("Classement", "—")

    forces, faiblesses = strengths_weaknesses_from_row(row, max_items=5, high_thr=90, low_thr=10)

    st.markdown("### Forces")
    st.write("\n".join([f"• {x}" for x in forces]) if forces else "—")

    st.markdown("### Faiblesses")
    st.write("\n".join([f"• {x}" for x in faiblesses]) if faiblesses else "—")

with right:
    st.subheader("Radar")
    fig = radar_plotly(labels, values, title=None)
    st.plotly_chart(fig, use_container_width=True)

st.caption("Forces/Faiblesses : seuils par défaut = top ≥ 90, bottom ≤ 10 (après inversion des métriques où 'moins = mieux').")
