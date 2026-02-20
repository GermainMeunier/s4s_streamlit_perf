import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import textwrap

from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ================================
# CONFIGS (Power BI)
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
        ("Dribble", "score_dribble"),
        ("Passe", "score_passe"),
        ("Finition", "score_finition"),
        ("Présence", "score_presence"),
        ("Impact", "score_impact"),
        ("Défense", "score_defense"),
    ],
    "milieu": [
        ("Présence", "score_presence"),
        ("Projection", "score_projection"),
        ("Impact", "score_impact"),
        ("Récuperation", "score_recuperation"),
        ("Danger", "score_danger"),
        ("Technique", "score_technique"),
    ],
    "piston": [
        ("Défense", "score_defense"),
        ("Relance", "score_relance"),
        ("Progression", "score_progression"),
        ("Projection", "score_projection"),
        ("Centre", "score_centre"),
        ("Impact", "score_impact"),
    ],
    "defenseur": [
        ("Duel sol", "score_duel_def"),
        ("Intervention", "score_intervention"),
        ("Relance", "score_relance"),
        ("Duel aérien", "score_impact"),
        ("Projection", "score_projection"),
        ("Discipline", "score_discipline"),
    ],
}

LABEL_OVERRIDES_BY_POSTE = {
    "avant-centre": {
        "Finition": {"r": 110, "dtheta": 0, "ha": "center"},
        "Impact": {"r": 110, "dtheta": 0, "ha": "left"},
        "Dribble": {"r": 110, "dtheta": 0, "ha": "left"},
        "Déplacement": {"r": 110, "dtheta": 0, "ha": "center"},
        "Passe": {"r": 110, "dtheta": 0, "ha": "right"},
        "Défense": {"r": 110, "dtheta": 0, "ha": "right"},
    },
    "ailier - offensif": {
        "Dribble": {"r": 110, "dtheta": 0, "ha": "center"},
        "Passe": {"r": 110, "dtheta": 0, "ha": "left"},
        "Finition": {"r": 110, "dtheta": 0, "ha": "left"},
        "Présence": {"r": 110, "dtheta": 0, "ha": "center"},
        "Impact": {"r": 110, "dtheta": 0, "ha": "right"},
        "Défense": {"r": 110, "dtheta": 0, "ha": "right"},
    },
    "milieu": {
        "Présence": {"r": 110, "dtheta": 0, "ha": "center"},
        "Projection": {"r": 110, "dtheta": 0, "ha": "left"},
        "Impact": {"r": 110, "dtheta": 0, "ha": "left"},
        "Récuperation": {"r": 110, "dtheta": 0, "ha": "center"},
        "Danger": {"r": 110, "dtheta": 0, "ha": "right"},
        "Technique": {"r": 110, "dtheta": 0, "ha": "right"},
    },
    "piston": {
        "Défense": {"r": 110, "dtheta": 0, "ha": "center"},
        "Relance": {"r": 110, "dtheta": 0, "ha": "left"},
        "Progression": {"r": 110, "dtheta": 0, "ha": "left"},
        "Projection": {"r": 110, "dtheta": 0, "ha": "center"},
        "Centre": {"r": 110, "dtheta": 0, "ha": "right"},
        "Impact": {"r": 110, "dtheta": 0, "ha": "right"},
    },
    "defenseur": {
        "Duel sol": {"r": 110, "dtheta": 0, "ha": "center"},
        "Intervention": {"r": 110, "dtheta": 0, "ha": "left"},
        "Relance": {"r": 110, "dtheta": 0, "ha": "left"},
        "Duel aérien": {"r": 110, "dtheta": 0, "ha": "center"},
        "Projection": {"r": 110, "dtheta": 0, "ha": "right"},
        "Discipline": {"r": 110, "dtheta": 0, "ha": "right"},
    },
}

VAR_LABELS = {
    "pct_block_pass_p90": "Passes contrées (90m)",
    "pct_block_shot_p90": "Tirs contrés (90m)",
    "pct_carry_lost_pct": "Conduites de balle perdues (%)",
    "pct_carry_prog_p90": "Conduites progressives (90m)",
    "pct_challenge_att_p90": "Duels défensifs disputés (90m)",
    "pct_challenge_pct": "Duels défensifs gagnés (%)",
    "pct_degagement_p90": "Dégagements (90m)",
    "pct_duel_pct": "Duels aériens gagnés (%)",
    "pct_duel_play_p90": "Duels aériens disputés (90m)",
    "pct_erreur_opp_shot_p90": "Erreur menant à un tir adverse (90m)",
    "pct_faute_commise_p90": "Fautes commises (90m)",
    "pct_interception_p90": "Interceptions (90m)",
    "pct_pass_long_pct": "Passes longues réussies (%)",
    "pct_pass_med_pct": "Passes moyennes réussies (%)",
    "pct_pass_prog_p90": "Passes progressives (90m)",
    "pct_pass_short_pct": "Passes courtes réussies (%)",
    "pct_tacle_p90": "Tacles tentés (90m)",
    "pct_tacle_win_ball_pct": "Tacles réussis (%)",
    "pct_yellow_card_p90": "Carton jaune reçu (90m)",
    "pct_ballon_touche_surface_off_p90": "Ballons touchés surface adverse (90m)",
    "pct_carry_in_surface_p90": "Conduite surface adverse (90m)",
    "pct_dribble_att_p90": "Dribbles tentés (90m)",
    "pct_dribble_pct": "Dribbles réussis (%)",
    "pct_faute_won_p90": "Fautes provoquées (90m)",
    "pct_goals_no_pk_p90": "Buts inscrits hors penalty (90m)",
    "pct_offside_p90": "Hors-jeu (90m)",
    "pct_pass_profondeur_defense_p90": "Passes dos de la défense (90m)",
    "pct_pass_prog_recu_p90": "Passes progressives reçues (90m)",
    "pct_pass_shot_p90": "Passes amenant à un tir (90m)",
    "pct_pass_surface_cmp_p90": "Passes réussies dans la surface adverse (90m)",
    "pct_passe_pct": "Passes réussies (%)",
    "pct_shot_on_target_pct": "Tirs cadrés (%)",
    "pct_xG_per_shot": "xG par tir",
    "pct_xGot_per_shot_on_target": "xGot par tir cadré",
    "pct_fautes_net_pct": "Ratio fautes subies & commises",
    "pct_passe_att_p90": "Passes tentées (90m)",
    "pct_shots_p90": "Tirs tentés (90m)",
    "pct_touch_med_p90": "Ballons touchés milieu (90m)",
    "pct_touch_off_p90": "Ballons touchés offensif (90m)",
    "pct_centre_p90": "Centres tentés (90m)",
    "pct_centre_surface_cmp_p90": "Centres réussies dans la surface adverse (90m)",
    "pct_pass_transversal_p90": "Passes transversales (90m)",
}

# ================================
# APP SETUP
# ================================
st.set_page_config(page_title="Performance & Profil", layout="wide")

st.markdown(
    """
    <div style="background-color:#e6692e; padding:18px 12px; text-align:center;">
      <div style="color:white; font-size:34px; font-weight:700; line-height:1.1;">
        Performance & profil – Ligue 1 – 2024/2025
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["poste"] = df["poste"].astype(str).str.strip().str.lower()
    df["player"] = df["player"].astype(str).str.strip()
    return df

df = load_csv("data/data.csv")

def _poste_fallback(poste: str) -> str:
    poste = (poste or "").strip().lower()
    if poste in METRICS_BY_POSTE:
        return poste
    for k in METRICS_BY_POSTE.keys():
        if k in poste:
            return k
    return "defenseur"

def fig_to_png_bytes(fig) -> bytes:
    canvas = FigureCanvas(fig)
    buf = BytesIO()
    canvas.print_png(buf)  # pas de bbox_inches="tight"
    return buf.getvalue()

# ================================
# RADAR -> PNG (taille FIXE)
# ================================
def pizza_radar_by_poste(row: pd.Series):
    poste = _poste_fallback(str(row.get("poste", "")))
    metrics = METRICS_BY_POSTE[poste]
    label_overrides = LABEL_OVERRIDES_BY_POSTE.get(poste, {})

    labels = [lab for lab, _ in metrics]
    pct_cols = [col + "_pct" for _, col in metrics]

    values = np.array([pd.to_numeric(row.get(c, np.nan), errors="coerce") for c in pct_cols], dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    values = np.clip(values, 0, 100)

    N = len(values)
    angle_offset = np.deg2rad(-30)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) + angle_offset
    width = 2*np.pi / N
    mid_angles = angles + width / 2

    cmap = mpl.colormaps["RdYlGn"]
    colors = [cmap(v / 100) for v in values]

    # Taille pixel FIXE (ex: 900x900)
    fig = plt.figure(figsize=(6.0, 6.0), dpi=150)  # 900x900
    fig.set_constrained_layout(False)

    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    r_outer = 100
    ax.set_ylim(0, r_outer + 20)

    grey = "#9aa0a6"
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # axe plus petit pour laisser la place aux labels
    ax.set_position([0.14, 0.14, 0.72, 0.72])

    theta_dense = np.linspace(0, 2*np.pi, 800)
    for r in [25, 50, 75]:
        ax.plot(theta_dense, np.full_like(theta_dense, r), lw=1, alpha=0.25, color=grey)

    for th in np.append(angles, angles[0]):
        ax.plot([th, th], [0, r_outer + 3], color=grey, lw=1, alpha=0.35)

    ax.bar(angles, values, width=width, bottom=0, align="edge",
           color=colors, edgecolor=grey, linewidth=1.2)

    ax.plot(theta_dense, np.full_like(theta_dense, r_outer), color="#222222", lw=1.1)

    # valeurs
    for th, v in zip(mid_angles, values):
        ax.text(th, min(v + 3, r_outer - 3), f"{int(v)}",
                ha="center", va="center", fontsize=8)

    # labels
    base_r = r_outer + 10
    for lab, th in zip(labels, mid_angles):
        ov = label_overrides.get(lab, {})
        r_lab = ov.get("r", base_r)
        ax.text(th, r_lab, lab,
                ha=ov.get("ha", "center"), va="center",
                fontsize=12, clip_on=False)

    png = fig_to_png_bytes(fig)
    plt.close(fig)
    return poste, values, png

# ================================
# FORCES / FAIBLESSES (liste 5/5)
# ================================
def strengths_weaknesses_always5(row: pd.Series, max_items=5):
    pct_cols = [c for c in row.index if str(c).startswith("pct_") and not str(c).startswith("pct_score")]

    vals = []
    for c in pct_cols:
        v = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        if pd.isna(v):
            continue
        vals.append((c, float(v)))

    if not vals:
        return [], []

    vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)

    forces = vals_sorted[:max_items]
    faiblesses = list(reversed(vals_sorted[-max_items:]))

    def label(col):
        return VAR_LABELS.get(col, col.replace("pct_", "").replace("_", " "))

    return [label(c) for c, _ in forces], [label(c) for c, _ in faiblesses]

# ================================
# UI LAYOUT
# ================================
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.15, 1.35], vertical_alignment="top")

with col_left:
    player = st.selectbox("Joueur", sorted(df["player"].unique()))

    row_df = df[df["player"] == player]
    if row_df.empty:
        st.warning("Aucune donnée pour ce joueur.")
        st.stop()
    row = row_df.iloc[0]

    poste, radar_values, radar_png = pizza_radar_by_poste(row)

    st.markdown(f"<div style='font-size:30px; font-weight:700; margin-top:8px;'>{poste}</div>", unsafe_allow_html=True)

    score_moyen = float(np.mean(radar_values)) if len(radar_values) else np.nan
    rank_txt = "—"
    if "global_rank" in row.index and "global_n" in row.index:
        try:
            rank_txt = f"{int(row['global_rank'])} / {int(row['global_n'])}"
        except Exception:
            rank_txt = "—"

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("<div style='font-weight:700; text-decoration:underline;'>Score moyen</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:64px; font-weight:800;'>{score_moyen:.1f}</div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div style='font-weight:700; text-decoration:underline;'>Classement</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:64px; font-weight:800;'>{rank_txt}</div>", unsafe_allow_html=True)

    forces, faiblesses = strengths_weaknesses_always5(row, max_items=5)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='color:#0b7a3b; font-size:28px; font-weight:800;'>Forces :</div>", unsafe_allow_html=True)
    for item in forces:
        st.markdown(f"<div style='font-size:20px;'>• {item}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    st.markdown("<div style='color:#b22222; font-size:28px; font-weight:800;'>Faiblesses :</div>", unsafe_allow_html=True)
    for item in faiblesses:
        st.markdown(f"<div style='font-size:20px;'>• {item}</div>", unsafe_allow_html=True)

with col_right:
    st.image(radar_png, use_container_width=True)

    st.markdown(
        """
        <div style="text-align:right; font-weight:800; margin-top:10px;">
        Un score de 80 signifie que le joueur est meilleur que 80% des joueurs de même poste sur ce facteur de performance
        </div>
        """,
        unsafe_allow_html=True,
    )
