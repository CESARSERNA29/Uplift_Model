import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, mean_absolute_error, r2_score
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
import io

from pathlib import Path

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Maximización de Donaciones | ML Dashboard",
    #page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paleta de colores corporativa ───────────────────────────────────────────
COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86AB",
    "accent": "#A8DADC",
    "success": "#2D6A4F",
    "warning": "#E9C46A",
    "danger": "#E76F51",
    "light": "#F8F9FA",
    "dark": "#212529",
    "purple": "#6C3483",
    "teal": "#148F77",
}

SEGMENT_COLORS = {
    "persuadable": "#2D6A4F",
    "sure_thing": "#2E86AB",
    "sleeping_dog": "#E76F51",
    "lost_cause": "#95A5A6",
}

# ─── CSS personalizado ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F4F6F9; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1B4F72;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #A8DADC !important;
        font-weight: 600;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB !important;
        color: white !important;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 5px solid;
        margin-bottom: 10px;
    }
    .kpi-value { font-size: 2rem; font-weight: 800; margin: 0; }
    .kpi-label { font-size: 0.85rem; color: #6c757d; margin: 0; }
    .kpi-delta { font-size: 0.9rem; font-weight: 600; }
    .section-header {
        background: linear-gradient(135deg, #1B4F72, #2E86AB);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 20px 0 15px 0;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .info-box {
        background: #EBF5FB;
        border: 1px solid #2E86AB;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: #FEF9E7;
        border: 1px solid #E9C46A;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
    }
    .danger-box {
        background: #FDEDEC;
        border: 1px solid #E76F51;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
    }
    .success-box {
        background: #EAFAF1;
        border: 1px solid #2D6A4F;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B4F72 0%, #154360 100%);
    }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] .stSelectbox label { color: #A8DADC !important; }
    div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: white !important; }
    .donor-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 12px;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        background: white;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ─── Funciones utilitarias ───────────────────────────────────────────────────

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    # Leer desde carpeta local data_donors/ 
    data_path = Path("data/donors_dataset.csv") 
    if data_path.exists():
        return pd.read_csv(data_path)
    
    st.error("❌ No se encontró el archivo en 'data_donors/donors_dataset.csv'. "
             "Ejecuta primero generate_donors_dataset.py o sube el CSV desde el sidebar.")
    st.stop()


def prepare_features(df):
    le = LabelEncoder()
    df2 = df.copy()
    cat_cols = ["acquisition_channel", "cause", "gender", "socioeconomic_level"]
    for col in cat_cols:
        df2[col + "_enc"] = le.fit_transform(df2[col].astype(str))
    return df2

FEATURE_COLS = [
    "age", "months_active", "payment_success_rate", "consecutive_failed_payments",
    "recency_days", "email_open_rate", "email_click_rate", "last_email_open_days",
    "complaint_history", "pause_requests", "rfm_recency", "rfm_frequency", "rfm_monetary",
    "current_amount", "amount_variance", "previous_upgrade_requests",
    "upgrade_acceptance_rate", "acquisition_channel_enc", "cause_enc",
    "socioeconomic_level_enc", "spontaneous_upgrade"
]

@st.cache_resource
def train_churn_model(df):
    df2 = prepare_features(df)
    X = df2[FEATURE_COLS].fillna(0)
    y = df2["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.35).astype(int)
    auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return model, auc, fpr, tpr, cm, fi, cv_scores, X_test, y_test, y_pred_proba

@st.cache_resource
def train_uplift_model(df):
    df2 = prepare_features(df)
    treated = df2[df2["experiment_group"] == 1]
    control = df2[df2["experiment_group"] == 0]

    X_treated = treated[FEATURE_COLS].fillna(0)
    y_treated = treated["upgrade_response"]
    X_control = control[FEATURE_COLS].fillna(0)
    y_control = control["upgrade_response"]

    # ── Validación: ambos grupos necesitan al menos 2 clases ──
    if y_treated.nunique() < 2:
        st.error("❌ El grupo TRATAMIENTO solo tiene una clase en 'upgrade_response'. "
                 "Verifica que el CSV tenga respuestas positivas y negativas.")
        st.stop()

    if y_control.nunique() < 2:
        # El grupo control no tiene respuestas positivas — comportamiento normal
        # Solución: usar el grupo tratamiento como proxy del control
        st.warning("⚠️ El grupo control no tiene variación en 'upgrade_response'. "
                   "Se usará un modelo de control basado en probabilidad base.")
        m_t = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        m_t.fit(X_treated, y_treated)
        # Control: predice probabilidad constante (media del target)
        base_prob = float(y_treated.mean())
        X_all = df2[FEATURE_COLS].fillna(0)
        uplift_scores = m_t.predict_proba(X_all)[:, 1] - base_prob
        fi_t = pd.Series(m_t.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        fi_c = fi_t.copy()  # misma importancia como fallback
        return m_t, m_t, uplift_scores, fi_t, fi_c

    m_t = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    m_c = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    m_t.fit(X_treated, y_treated)
    m_c.fit(X_control, y_control)
    X_all = df2[FEATURE_COLS].fillna(0)
    uplift_scores = m_t.predict_proba(X_all)[:, 1] - m_c.predict_proba(X_all)[:, 1]
    fi_t = pd.Series(m_t.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    fi_c = pd.Series(m_c.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    return m_t, m_c, uplift_scores, fi_t, fi_c

@st.cache_resource
def train_timing_model(df):
    df2 = prepare_features(df)
    timing_features = ["payment_success_rate", "email_open_rate", "consecutive_failed_payments",
                       "complaint_history", "pause_requests", "months_active", "rfm_frequency",
                       "recency_days", "spontaneous_upgrade"]
    X = df2[timing_features].fillna(0)
    y = df2["optimal_intervention_month"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    fi = pd.Series(model.feature_importances_, index=timing_features).sort_values(ascending=False)
    return model, mae, r2, fi, timing_features, X_test, y_test, y_pred

@st.cache_resource
def train_amount_model(df):
    df2 = prepare_features(df)
    amount_features = ["current_amount", "payment_success_rate", "email_open_rate",
                       "months_active", "rfm_monetary", "socioeconomic_level_enc",
                       "consecutive_failed_payments", "upgrade_acceptance_rate",
                       "spontaneous_upgrade", "amount_variance"]
    X = df2[amount_features].fillna(0)
    y = df2["optimal_upgrade_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    fi = pd.Series(model.feature_importances_, index=amount_features).sort_values(ascending=False)
    return model, mae, r2, fi, amount_features, X_test, y_test, y_pred

def plot_feature_importance(fi, title, n=10, color=COLORS["secondary"]):
    fig, ax = plt.subplots(figsize=(8, 4))
    fi.head(n).plot(kind="barh", ax=ax, color=color)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Importancia")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig

def render_metric_card(label, value, delta=None, color=COLORS["secondary"], icon="📊"):
    delta_html = f'<p class="kpi-delta" style="color:{"#2D6A4F" if delta and "+" in str(delta) else "#E76F51"}">{delta}</p>' if delta else ""
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
        <p class="kpi-label">{icon} {label}</p>
        <p class="kpi-value" style="color:{color}">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Maximización de Donaciones")
    st.markdown("---")
    st.markdown("### 📂 Fuente de Datos")
    uploaded = st.file_uploader("Cargar CSV de donantes", type=["csv"])
    st.markdown("---")
    st.markdown("### ⚙️ Parámetros Globales")
    churn_threshold = st.slider("Umbral Riesgo Churn", 0.10, 0.70, 0.35, 0.05,
                                help="Donantes sobre este umbral entran en modo protección")
    uplift_threshold = st.slider("Umbral Uplift Mínimo", 0.00, 0.40, 0.10, 0.05,
                                 help="Uplift mínimo requerido para intervenir")
    st.markdown("---")
    st.markdown("### Navegación")
    st.markdown("""
     **1. Overview** — KPIs y portafolio  
     **2. Churn Model** — Predicción abandono  
     **3. Uplift Model** — Segmentación causal  
     **4. Timing Model** — Mes óptimo  
     **5. Amount Model** — Monto óptimo  
     **6. Pipeline** — Decisión integral  
     **7. Simulador** — Donante individual
    """)
    st.markdown("---")
    st.caption("v1.0 | Metodología ML Donaciones")

# ─── CARGA DE DATOS ──────────────────────────────────────────────────────────
df = load_data(uploaded)

N = len(df) 

df2 = prepare_features(df)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1B4F72,#2E86AB,#148F77);
            padding:24px 30px;border-radius:16px;margin-bottom:24px;color:white;">
    <h1 style="margin:0;font-size:2rem;"> Dashboard de Maximización de Donaciones</h1>
    <p style="margin:6px 0 0 0;opacity:0.85;font-size:1rem;">
        Sistema integral de ML para optimizar el valor de vida del donante con mínimo riesgo de churn
    </p>
</div>
""", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview", "🔴 Churn Model", "🎯 Uplift Model",
    "⏱ Timing Model", "💰 Amount Model", "⚙️ Pipeline Decisión", "🔍 Simulador"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">📊 Resumen del Portafolio de Donantes</div>', unsafe_allow_html=True)

    active_donors = len(df[df["churned"] == 0])
    total_monthly = df[df["churned"] == 0]["current_amount"].sum()
    churn_rate = df["churned"].mean()
    avg_donation = df[df["churned"] == 0]["current_amount"].mean()
    upgrade_candidates = df[
        (df["churned"] == 0) &
        (df["months_active"] >= 6) &
        (df["consecutive_failed_payments"] == 0)
    ]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_metric_card("Donantes Activos", f"{active_donors:,}", None, COLORS["primary"], "👥")
    with c2: render_metric_card("Recaudo Mensual", f"${total_monthly:,.0f}", None, COLORS["success"], "💰")
    with c3: render_metric_card("Tasa de Churn", f"{churn_rate:.1%}", None, COLORS["danger"], "⚠️")
    with c4: render_metric_card("Donación Promedio", f"${avg_donation:.1f}", None, COLORS["secondary"], "📈")
    with c5: render_metric_card("Candidatos Upgrade", f"{len(upgrade_candidates):,}", None, COLORS["teal"], "🎯")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Distribución por Canal de Adquisición**")
        channel_data = df.groupby("acquisition_channel").agg(
            donantes=("donor_id", "count"),
            promedio=("current_amount", "mean"),
            churn=("churned", "mean")
        ).reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(channel_data["acquisition_channel"], channel_data["donantes"],
                      color=[COLORS["secondary"], COLORS["teal"], COLORS["success"],
                             COLORS["warning"], COLORS["primary"]], edgecolor="white", linewidth=1.5)
        ax.set_xlabel("Canal"); ax.set_ylabel("Donantes")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, channel_data["donantes"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Distribución por Causa**")
        cause_counts = df["cause"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_pie = [COLORS["primary"], COLORS["secondary"], COLORS["teal"],
                      COLORS["success"], COLORS["warning"]]
        wedges, texts, autotexts = ax.pie(cause_counts.values, labels=cause_counts.index,
                                           autopct="%1.1f%%", colors=colors_pie,
                                           startangle=90, pctdistance=0.75)
        for at in autotexts: at.set_fontsize(8); at.set_fontweight("bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col3:
        st.markdown("**Distribución de Montos Actuales**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[df["current_amount"] < 150]["current_amount"], bins=40,
                color=COLORS["secondary"], edgecolor="white", alpha=0.85)
        ax.axvline(avg_donation, color=COLORS["danger"], linewidth=2, linestyle="--",
                   label=f"Promedio: ${avg_donation:.1f}")
        ax.set_xlabel("Monto Mensual"); ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Segmentos Uplift en el Portafolio**")
        seg_counts = df["uplift_segment"].value_counts()
        seg_labels = {"persuadable": "Persuadables 🎯", "sure_thing": "Sure Things ✅",
                      "sleeping_dog": "Sleeping Dogs 🐕", "lost_cause": "Lost Causes ❌"}
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh([seg_labels.get(s, s) for s in seg_counts.index], seg_counts.values,
                       color=[SEGMENT_COLORS.get(s, "#999") for s in seg_counts.index])
        for bar, val in zip(bars, seg_counts.values):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f"{val} ({val/N*100:.1f}%)", va="center", fontsize=9)
        ax.set_xlabel("Número de Donantes")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col5:
        st.markdown("**Churn Rate por Nivel Socioeconómico**")
        churn_by_level = df.groupby("socioeconomic_level")["churned"].mean().reset_index()
        order = ["bajo", "medio_bajo", "medio", "medio_alto", "alto"]
        churn_by_level["socioeconomic_level"] = pd.Categorical(
            churn_by_level["socioeconomic_level"], categories=order, ordered=True)
        churn_by_level = churn_by_level.sort_values("socioeconomic_level")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_bar = [COLORS["danger"] if v > churn_rate else COLORS["success"]
                      for v in churn_by_level["churned"]]
        bars = ax.bar(churn_by_level["socioeconomic_level"], churn_by_level["churned"],
                      color=colors_bar, edgecolor="white")
        ax.axhline(churn_rate, color=COLORS["dark"], linewidth=1.5, linestyle="--",
                   label=f"Promedio: {churn_rate:.1%}")
        ax.set_ylabel("Tasa de Churn"); ax.set_ylim(0, 0.25)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        for bar, val in zip(bars, churn_by_level["churned"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")
        ax.legend(fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("**Vista Previa del Dataset** (primeros 20 registros)")
    display_cols = ["donor_id", "acquisition_channel", "cause", "months_active",
                    "current_amount", "payment_success_rate", "email_open_rate",
                    "churned", "uplift_segment", "optimal_intervention_month"]
    st.dataframe(df[display_cols].head(20), use_container_width=True, height=280)

    # Download buttons
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV completo", csv_data,
                           "donors_dataset.csv", "text/csv", use_container_width=True)
    with col_d2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Donors", index=False)
        st.download_button("⬇️ Descargar Excel completo", excel_buffer.getvalue(),
                           "donors_dataset.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: CHURN MODEL
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">🔴 Modelo 1: Predicción de Churn (Guardia de Riesgo)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Objetivo:</b> Identificar donantes en riesgo de abandono <i>antes</i> de cualquier acción de maximización.
    Donantes con P(churn) &gt; umbral entran en <b>modo protección</b> y no serán contactados para upgrade.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Entrenando modelo de Churn (GBM)..."):
        churn_model, auc, fpr, tpr, cm, fi_churn, cv_scores, X_test_c, y_test_c, y_pred_proba_c = train_churn_model(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("AUC-ROC", f"{auc:.4f}", None, COLORS["primary"], "📈")
    with c2: render_metric_card("CV AUC (5-fold)", f"{cv_scores.mean():.4f} ±{cv_scores.std():.4f}", None, COLORS["secondary"], "🔄")
    at_risk = (y_pred_proba_c > churn_threshold).sum()
    with c3: render_metric_card("En Riesgo (test)", f"{at_risk} / {len(y_test_c)}", None, COLORS["danger"], "⚠️")
    precision_at_thresh = y_test_c[(y_pred_proba_c > churn_threshold) == 1].mean() if at_risk > 0 else 0
    with c4: render_metric_card("Precisión @ Umbral", f"{precision_at_thresh:.1%}", None, COLORS["warning"], "🎯")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Curva ROC**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color=COLORS["secondary"], linewidth=2.5, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS["secondary"])
        ax.set_xlabel("FPR (Falsos Positivos)"); ax.set_ylabel("TPR (Verdaderos Positivos)")
        ax.set_title("Curva ROC - Churn Model", fontweight="bold")
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Matriz de Confusión**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
        ax.set_title("Matriz de Confusión", fontweight="bold")
        ax.set_ylabel("Real"); ax.set_xlabel("Predicho")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col3:
        st.markdown("**Importancia de Variables**")
        fig = plot_feature_importance(fi_churn, "Top Features — Churn Model", color=COLORS["danger"])
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Distribución de Probabilidad de Churn**")
        churn_probs_all = churn_model.predict_proba(df2[FEATURE_COLS].fillna(0))[:, 1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(churn_probs_all[df["churned"] == 0], bins=40, alpha=0.7,
                label="No abandonó", color=COLORS["success"])
        ax.hist(churn_probs_all[df["churned"] == 1], bins=40, alpha=0.7,
                label="Abandonó", color=COLORS["danger"])
        ax.axvline(churn_threshold, color="black", linewidth=2, linestyle="--",
                   label=f"Umbral: {churn_threshold:.2f}")
        ax.set_xlabel("P(Churn)"); ax.set_ylabel("Frecuencia")
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col5:
        st.markdown("**Validación Cruzada (5-Fold)**")
        fig, ax = plt.subplots(figsize=(7, 4))
        fold_names = [f"Fold {i+1}" for i in range(5)]
        colors_cv = [COLORS["success"] if s > 0.75 else COLORS["warning"] for s in cv_scores]
        bars = ax.bar(fold_names, cv_scores, color=colors_cv, edgecolor="white")
        ax.axhline(cv_scores.mean(), color=COLORS["danger"], linewidth=2, linestyle="--",
                   label=f"Media: {cv_scores.mean():.4f}")
        ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.5, 1.0)
        for bar, val in zip(bars, cv_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Donantes con Mayor Riesgo de Churn**")
    df2["churn_prob"] = churn_model.predict_proba(df2[FEATURE_COLS].fillna(0))[:, 1]
    high_risk = df2[df2["churn_prob"] > churn_threshold].sort_values("churn_prob", ascending=False)
    display_risk = high_risk[["donor_id", "months_active", "current_amount",
                               "payment_success_rate", "consecutive_failed_payments",
                               "complaint_history", "churn_prob"]].head(15)
    display_risk["churn_prob"] = display_risk["churn_prob"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_risk, use_container_width=True, height=300)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: UPLIFT MODEL
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">🎯 Modelo 2: Uplift Modeling (Segmentación Causal)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Objetivo:</b> Medir el efecto causal de la solicitud de upgrade sobre cada donante.
    Identifica <i>Persuadables</i>, <i>Sure Things</i>, <i>Lost Causes</i> y los críticos <b>Sleeping Dogs 🐕</b>.
    <br><b>Método:</b> Two-Model Approach — entrena un modelo para el grupo tratamiento y otro para el control.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Entrenando modelos Uplift (Two-Model Approach)..."):
        m_t, m_c, uplift_scores, fi_t, fi_c = train_uplift_model(df)

    df2["uplift_score"] = uplift_scores

    # Segmentación por uplift score
    def classify_uplift(row):
        if row["churn_prob"] > churn_threshold and row["uplift_score"] < 0:
            return "sleeping_dog"
        elif row["uplift_score"] >= uplift_threshold and row["churn_prob"] <= churn_threshold:
            return "persuadable"
        elif row["spontaneous_upgrade"] == 1:
            return "sure_thing"
        elif row["uplift_score"] < 0:
            return "lost_cause"
        else:
            return "persuadable"

    df2["uplift_segment_pred"] = df2.apply(classify_uplift, axis=1)
    seg_counts_pred = df2["uplift_segment_pred"].value_counts()

    c1, c2, c3, c4 = st.columns(4)
    persuadables = (df2["uplift_segment_pred"] == "persuadable").sum()
    sure_things = (df2["uplift_segment_pred"] == "sure_thing").sum()
    sleeping_dogs = (df2["uplift_segment_pred"] == "sleeping_dog").sum()
    lost_causes = (df2["uplift_segment_pred"] == "lost_cause").sum()
    with c1: render_metric_card("Persuadables 🎯", f"{persuadables:,}", f"{persuadables/N*100:.1f}% del total", COLORS["success"], "✅")
    with c2: render_metric_card("Sure Things ⚡", f"{sure_things:,}", f"{sure_things/N*100:.1f}% del total", COLORS["secondary"], "📈")
    with c3: render_metric_card("Sleeping Dogs 🐕", f"{sleeping_dogs:,}", "NO intervenir", COLORS["danger"], "🚫")
    with c4: render_metric_card("Lost Causes ❌", f"{lost_causes:,}", f"{lost_causes/N*100:.1f}% del total", COLORS["dark"], "💤")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribución de Uplift Scores**")
        fig, ax = plt.subplots(figsize=(7, 4))
        for seg, color in SEGMENT_COLORS.items():
            mask = df2["uplift_segment_pred"] == seg
            if mask.sum() > 0:
                ax.hist(df2[mask]["uplift_score"], bins=30, alpha=0.7,
                        label=seg.replace("_", " ").title(), color=color)
        ax.axvline(uplift_threshold, color="black", linewidth=2, linestyle="--",
                   label=f"Umbral Uplift: {uplift_threshold:.2f}")
        ax.axvline(0, color="gray", linewidth=1, linestyle=":", alpha=0.7)
        ax.set_xlabel("Uplift Score"); ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Uplift Score vs Probabilidad de Churn**")
        fig, ax = plt.subplots(figsize=(7, 4))
        for seg, color in SEGMENT_COLORS.items():
            mask = df2["uplift_segment_pred"] == seg
            if mask.sum() > 0:
                ax.scatter(df2[mask]["uplift_score"], df2[mask]["churn_prob"],
                           alpha=0.3, color=color, s=15,
                           label=seg.replace("_", " ").title())
        ax.axvline(uplift_threshold, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.axhline(churn_threshold, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_xlabel("Uplift Score"); ax.set_ylabel("P(Churn)")
        ax.legend(fontsize=8, markerscale=2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Importancia Variables — Modelo Tratamiento**")
        fig = plot_feature_importance(fi_t, "Top Features (Tratamiento)", color=COLORS["success"])
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**Importancia Variables — Modelo Control**")
        fig = plot_feature_importance(fi_c, "Top Features (Control)", color=COLORS["secondary"])
        st.pyplot(fig); plt.close()

    st.markdown("**Impacto Económico por Segmento**")
    seg_economics = df2.groupby("uplift_segment_pred").agg(
        donantes=("donor_id", "count"),
        monto_promedio=("current_amount", "mean"),
        uplift_promedio=("uplift_score", "mean"),
        churn_promedio=("churn_prob", "mean"),
        recaudo_total=("current_amount", "sum")
    ).round(3)
    seg_economics["recaudo_total"] = seg_economics["recaudo_total"].apply(lambda x: f"${x:,.0f}")
    seg_economics["monto_promedio"] = seg_economics["monto_promedio"].apply(lambda x: f"${x:.1f}")
    seg_economics["uplift_promedio"] = seg_economics["uplift_promedio"].apply(lambda x: f"{x:.3f}")
    seg_economics["churn_promedio"] = seg_economics["churn_promedio"].apply(lambda x: f"{x:.1%}")
    st.dataframe(seg_economics, use_container_width=True)
    
    st.caption("El uplift score mide el efecto causal incremental de hacer la solicitud de upgrade:  Uplift = P(acepta | se le pide) - P(acepta | NO se le pide)")
    st.caption("Uplift > 0 :  La intervención ayuda — el donante es más probable que acepte si se le pide")
    st.caption("Uplift = 0 :  La intervención no cambia nada")
    st.caption("Uplift < 0 :  La intervención perjudica — el donante tiene MENOS probabilidad de aceptar, y puede abandonar, si se le contacta")
    


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: TIMING MODEL
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">⏱ Modelo 3: Timing Óptimo de Intervención</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Objetivo:</b> Predecir el mes óptimo e individual para solicitar el upgrade a cada donante,
    reemplazando los intervalos fijos (6, 12, 24 meses) por una predicción personalizada basada
    en señales de madurez y engagement del donante.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Entrenando modelo de Timing (GBM Regressor)..."):
        timing_model, mae_t, r2_t, fi_timing, timing_features, X_test_t, y_test_t, y_pred_t = train_timing_model(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("MAE (meses)", f"{mae_t:.2f}", None, COLORS["primary"], "📅")
    with c2: render_metric_card("R² Score", f"{r2_t:.4f}", None, COLORS["secondary"], "📊")
    timing_preds_all = timing_model.predict(df2[timing_features].fillna(0))
    with c3: render_metric_card("Mes Óptimo Promedio", f"{timing_preds_all.mean():.1f}", None, COLORS["teal"], "⏰")
    early_intervention = (timing_preds_all < 6).sum()
    with c4: render_metric_card("Intervención Temprana (<6m)", f"{early_intervention:,}", None, COLORS["success"], "⚡")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Predicho vs Real — Mes Óptimo**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test_t, y_pred_t, alpha=0.3, color=COLORS["secondary"], s=15)
        lims = [min(y_test_t.min(), y_pred_t.min()) - 1, max(y_test_t.max(), y_pred_t.max()) + 1]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
        ax.set_xlabel("Mes Óptimo Real"); ax.set_ylabel("Mes Óptimo Predicho")
        ax.set_title(f"MAE = {mae_t:.2f} meses | R² = {r2_t:.4f}", fontweight="bold")
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Importancia de Variables — Timing**")
        fig = plot_feature_importance(fi_timing, "Top Features — Timing Model",
                                     n=len(timing_features), color=COLORS["purple"])
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Distribución del Mes Óptimo Predicho**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(timing_preds_all, bins=20, color=COLORS["teal"], edgecolor="white", alpha=0.85)
        ax.axvline(6, color=COLORS["danger"], linewidth=2, linestyle="--", label="Actual: Mes 6")
        ax.axvline(12, color=COLORS["warning"], linewidth=2, linestyle="--", label="Actual: Mes 12")
        ax.axvline(24, color=COLORS["dark"], linewidth=2, linestyle="--", label="Actual: Mes 24")
        ax.axvline(timing_preds_all.mean(), color=COLORS["primary"], linewidth=2,
                   label=f"Media ML: {timing_preds_all.mean():.1f}")
        ax.set_xlabel("Mes Óptimo de Intervención"); ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**Mes Óptimo por Canal de Adquisición**")
        df2["timing_pred"] = timing_preds_all
        timing_by_channel = df2.groupby("acquisition_channel")["timing_pred"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(timing_by_channel["acquisition_channel"], timing_by_channel["mean"],
                      yerr=timing_by_channel["std"], capsize=5,
                      color=COLORS["secondary"], edgecolor="white", error_kw={"linewidth": 2})
        ax.axhline(6, color=COLORS["danger"], linewidth=1.5, linestyle="--",
                   label="Regla fija: Mes 6")
        ax.set_ylabel("Mes Óptimo (promedio)"); ax.set_ylim(0, 16)
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for bar, mean in zip(bars, timing_by_channel["mean"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{mean:.1f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Calendario de Próximas Intervenciones Sugeridas**")
    df2["timing_pred_rounded"] = timing_preds_all.round(0).astype(int)
    intervention_schedule = df2[
        (df2["uplift_segment_pred"] == "persuadable") &
        (df2["churn_prob"] <= churn_threshold)
    ].sort_values("timing_pred_rounded")[
        ["donor_id", "months_active", "current_amount", "timing_pred_rounded",
         "uplift_score", "churn_prob"]
    ].head(20)
    intervention_schedule["uplift_score"] = intervention_schedule["uplift_score"].apply(lambda x: f"{x:.3f}")
    intervention_schedule["churn_prob"] = intervention_schedule["churn_prob"].apply(lambda x: f"{x:.1%}")
    intervention_schedule["current_amount"] = intervention_schedule["current_amount"].apply(lambda x: f"${x:.1f}")
    intervention_schedule.columns = ["ID Donante", "Meses Activo", "Monto Actual",
                                      "Mes Intervención", "Uplift Score", "P(Churn)"]
    st.dataframe(intervention_schedule, use_container_width=True, height=300)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5: AMOUNT MODEL
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">💰 Modelo 4: Monto Óptimo de Solicitud</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Objetivo:</b> Predecir el monto de incremento óptimo para cada donante, basado en su elasticidad,
    capacidad económica estimada y comportamiento histórico. No todos los donantes tienen la misma tolerancia al aumento.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Entrenando modelo de Monto Óptimo (GBM Regressor)..."):
        amount_model, mae_a, r2_a, fi_amount, amount_features, X_test_a, y_test_a, y_pred_a = train_amount_model(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("MAE Monto", f"${mae_a:.2f}", None, COLORS["primary"], "💵")
    with c2: render_metric_card("R² Score", f"{r2_a:.4f}", None, COLORS["secondary"], "📊")
    amount_preds_all = amount_model.predict(df2[amount_features].fillna(0))
    avg_opt_amount = amount_preds_all.mean()
    with c3: render_metric_card("Incremento Promedio Sugerido", f"${avg_opt_amount:.1f}", None, COLORS["success"], "📈")
    pct_increase = (avg_opt_amount / df["current_amount"].mean() * 100)
    with c4: render_metric_card("% Incremento Promedio", f"{pct_increase:.1f}%", None, COLORS["teal"], "🎯")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Predicho vs Real — Monto Óptimo**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test_a, y_pred_a, alpha=0.3, color=COLORS["success"], s=15)
        lims = [0, max(y_test_a.max(), y_pred_a.max()) + 2]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
        ax.set_xlabel("Monto Óptimo Real"); ax.set_ylabel("Monto Óptimo Predicho")
        ax.set_title(f"MAE = ${mae_a:.2f} | R² = {r2_a:.4f}", fontweight="bold")
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Importancia de Variables — Amount Model**")
        fig = plot_feature_importance(fi_amount, "Top Features — Amount Model",
                                     n=len(amount_features), color=COLORS["teal"])
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Distribución de Incrementos Sugeridos**")
        df2["amount_pred"] = amount_preds_all
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(amount_preds_all[amount_preds_all < 50], bins=35,
                color=COLORS["success"], edgecolor="white", alpha=0.85)
        ax.axvline(avg_opt_amount, color=COLORS["danger"], linewidth=2, linestyle="--",
                   label=f"Promedio: ${avg_opt_amount:.1f}")
        ax.set_xlabel("Monto de Incremento Sugerido ($)"); ax.set_ylabel("Frecuencia")
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**Incremento Sugerido por Nivel Socioeconómico**")
        df2["amount_pred"] = amount_preds_all
        amount_by_level = df2.groupby("socioeconomic_level")["amount_pred"].mean().reset_index()
        order = ["bajo", "medio_bajo", "medio", "medio_alto", "alto"]
        amount_by_level["socioeconomic_level"] = pd.Categorical(
            amount_by_level["socioeconomic_level"], categories=order, ordered=True)
        amount_by_level = amount_by_level.sort_values("socioeconomic_level")
        fig, ax = plt.subplots(figsize=(7, 4))
        color_list = [COLORS["secondary"], COLORS["teal"], COLORS["success"],
                      COLORS["warning"], COLORS["danger"]]
        bars = ax.bar(amount_by_level["socioeconomic_level"], amount_by_level["amount_pred"],
                      color=color_list, edgecolor="white")
        ax.set_ylabel("Incremento Promedio Sugerido ($)")
        for bar, val in zip(bars, amount_by_level["amount_pred"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"${val:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Top 20 Donantes con Potencial de Incremento**")
    df2["amount_pred"] = amount_preds_all
    top_potential = df2[
        (df2["uplift_segment_pred"] == "persuadable") &
        (df2["churn_prob"] <= churn_threshold)
    ].sort_values("amount_pred", ascending=False)[
        ["donor_id", "current_amount", "amount_pred", "uplift_score", "churn_prob",
         "socioeconomic_level", "months_active"]
    ].head(20)
    top_potential["revenue_gain_annual"] = ((top_potential["amount_pred"]) * 12).round(2)
    display_tp = top_potential.copy()
    for col in ["current_amount", "amount_pred", "revenue_gain_annual"]:
        display_tp[col] = display_tp[col].apply(lambda x: f"${x:.2f}")
    display_tp["uplift_score"] = display_tp["uplift_score"].apply(lambda x: f"{x:.3f}")
    display_tp["churn_prob"] = display_tp["churn_prob"].apply(lambda x: f"{x:.1%}")
    display_tp.columns = ["ID", "Monto Actual", "Incremento Sugerido", "Uplift",
                           "P(Churn)", "Nivel SE", "Meses Activo", "Ganancia Anual Est."]
    st.dataframe(display_tp, use_container_width=True, height=320)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6: PIPELINE DE DECISIÓN
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">⚙️ Pipeline Integral de Decisión</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Este pipeline aplica los 4 modelos en cascada sobre <b>todos los donantes activos</b> y genera
    la acción recomendada para cada uno: intervenir, proteger o monitorear.
    </div>
    """, unsafe_allow_html=True)

    # Aplicar pipeline completo
    df2["churn_prob"] = churn_model.predict_proba(df2[FEATURE_COLS].fillna(0))[:, 1]
    df2["uplift_score"] = uplift_scores
    df2["timing_pred"] = timing_model.predict(df2[timing_features].fillna(0)).round(0).astype(int)
    df2["amount_pred"] = amount_model.predict(df2[amount_features].fillna(0)).round(2)
    df2["uplift_segment_pred"] = df2.apply(classify_uplift, axis=1)

    def pipeline_decision(row):
        if row["churned"] == 1:
            return "❌ Donante Inactivo"
        if row["churn_prob"] > churn_threshold:
            return "🛡️ Modo Protección (alto riesgo churn)"
        if row["uplift_segment_pred"] == "sleeping_dog":
            return "🐕 No Intervenir (Sleeping Dog)"
        if row["uplift_segment_pred"] == "lost_cause":
            return "⏸️ No Intervenir (Lost Cause)"
        if row["uplift_segment_pred"] == "sure_thing":
            return "✅ Monitorear (Sure Thing — subirá solo)"
        if row["months_active"] < row["timing_pred"]:
            return f"⏳ Esperar hasta mes {row['timing_pred']}"
        if row["uplift_score"] >= uplift_threshold:
            return f"🚀 INTERVENIR — Solicitar +${row['amount_pred']:.1f}"
        return "⏸️ En observación"

    df2["decision"] = df2.apply(pipeline_decision, axis=1)

    decision_counts = df2["decision"].value_counts()
    intervene_now = df2["decision"].str.contains("INTERVENIR").sum()
    protection = df2["decision"].str.contains("Protección").sum()
    wait = df2["decision"].str.contains("Esperar").sum()
    sleeping = df2["decision"].str.contains("Sleeping").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_metric_card("🚀 Intervenir Ahora", f"{intervene_now:,}", f"{intervene_now/N*100:.1f}%", COLORS["success"], "🚀")
    with c2: render_metric_card("🛡️ Modo Protección", f"{protection:,}", f"{protection/N*100:.1f}%", COLORS["danger"], "🛡️")
    with c3: render_metric_card("⏳ En Espera", f"{wait:,}", f"{wait/N*100:.1f}%", COLORS["warning"], "⏳")
    with c4: render_metric_card("🐕 Sleeping Dogs", f"{sleeping:,}", "Nunca intervenir", COLORS["primary"], "🚫")
    revenue_potential = df2[df2["decision"].str.contains("INTERVENIR")]["amount_pred"].sum()
    with c5: render_metric_card("💰 Potencial Mensual", f"${revenue_potential:,.0f}", "si todos aceptan", COLORS["teal"], "💰")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribución de Decisiones del Pipeline**")
        decision_summary = df2.groupby("decision").agg(
            count=("donor_id", "count")
        ).reset_index().sort_values("count", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors_dec = []
        for d in decision_summary["decision"]:
            if "INTERVENIR" in d: colors_dec.append(COLORS["success"])
            elif "Protección" in d: colors_dec.append(COLORS["danger"])
            elif "Sleeping" in d: colors_dec.append(COLORS["primary"])
            elif "Esperar" in d: colors_dec.append(COLORS["warning"])
            elif "Sure" in d: colors_dec.append(COLORS["secondary"])
            elif "Lost" in d: colors_dec.append(COLORS["dark"])
            else: colors_dec.append(COLORS["accent"])

        bars = ax.barh(decision_summary["decision"], decision_summary["count"],
                       color=colors_dec, edgecolor="white")
        for bar, val in zip(bars, decision_summary["count"]):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f"{val:,} ({val/N*100:.1f}%)", va="center", fontsize=8)
        ax.set_xlabel("Número de Donantes")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Uplift vs Riesgo Churn — Mapa de Decisiones**")
        fig, ax = plt.subplots(figsize=(8, 5))
        decision_color_map = {
            "🚀 INTERVENIR": COLORS["success"],
            "🛡️ Modo Protección": COLORS["danger"],
            "⏳ En Espera": COLORS["warning"],
            "🐕 No Intervenir": COLORS["primary"],
            "✅ Monitorear": COLORS["secondary"],
            "❌ Donante": "#ccc",
            "⏸️": COLORS["dark"],
        }
        for seg, color in SEGMENT_COLORS.items():
            mask = df2["uplift_segment_pred"] == seg
            if mask.sum() > 0:
                ax.scatter(df2[mask]["uplift_score"], df2[mask]["churn_prob"],
                           alpha=0.25, color=color, s=12,
                           label=seg.replace("_", " ").title())
        ax.axvline(uplift_threshold, color="black", linewidth=2, linestyle="--",
                   alpha=0.7, label=f"Umbral Uplift {uplift_threshold:.2f}")
        ax.axhline(churn_threshold, color="red", linewidth=2, linestyle="--",
                   alpha=0.7, label=f"Umbral Churn {churn_threshold:.2f}")
        ax.fill_between([uplift_threshold, df2["uplift_score"].max()], 0, churn_threshold,
                        alpha=0.05, color=COLORS["success"])
        ax.set_xlabel("Uplift Score"); ax.set_ylabel("P(Churn)")
        ax.legend(fontsize=8, markerscale=2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Lista de Intervenciones Prioritarias (Pipeline Output)**")
    pipeline_output = df2[df2["decision"].str.contains("INTERVENIR")].sort_values(
        "amount_pred", ascending=False
    )[["donor_id", "acquisition_channel", "cause", "months_active", "current_amount",
       "amount_pred", "uplift_score", "churn_prob", "timing_pred", "decision"]].head(30)

    display_pipeline = pipeline_output.copy()
    display_pipeline["current_amount"] = display_pipeline["current_amount"].apply(lambda x: f"${x:.1f}")
    display_pipeline["amount_pred"] = display_pipeline["amount_pred"].apply(lambda x: f"+${x:.1f}")
    display_pipeline["uplift_score"] = display_pipeline["uplift_score"].apply(lambda x: f"{x:.3f}")
    display_pipeline["churn_prob"] = display_pipeline["churn_prob"].apply(lambda x: f"{x:.1%}")
    display_pipeline.columns = ["ID", "Canal", "Causa", "Meses", "Monto Actual",
                                  "Incremento", "Uplift", "P(Churn)", "Mes Óptimo", "Decisión"]
    st.dataframe(display_pipeline, use_container_width=True, height=400)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df2[["donor_id", "acquisition_channel", "cause", "months_active", "current_amount",
             "amount_pred", "uplift_score", "churn_prob", "timing_pred",
             "uplift_segment_pred", "decision"]].to_excel(writer, sheet_name="Pipeline_Output", index=False)
    st.download_button("⬇️ Descargar Pipeline Output (Excel)",
                       excel_buffer.getvalue(), "pipeline_decisions.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ════════════════════════════════════════════════════════════════════════════
# TAB 7: SIMULADOR INDIVIDUAL
# ════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">🔍 Simulador de Donante Individual</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Ingresa los datos de un donante específico o selecciona uno del portafolio para ver
    la recomendación completa del pipeline: riesgo de churn, segmento uplift, timing y monto óptimo.
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Modo de entrada", ["Seleccionar donante existente", "Ingresar datos manualmente"],
                    horizontal=True)

    if mode == "Seleccionar donante existente":
        donor_selected = st.selectbox("Seleccionar donante", df["donor_id"].tolist())
        row = df[df["donor_id"] == donor_selected].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            age_sim = int(row["age"])
            months_active_sim = int(row["months_active"])
            current_amount_sim = float(row["current_amount"])
            payment_success_rate_sim = float(row["payment_success_rate"])
            consecutive_failed_sim = int(row["consecutive_failed_payments"])
        with col2:
            email_open_sim = float(row["email_open_rate"])
            email_click_sim = float(row["email_click_rate"])
            last_email_days_sim = int(row["last_email_open_days"])
            complaint_sim = int(row["complaint_history"])
            pause_sim = int(row["pause_requests"])
        with col3:
            prev_upgrades_sim = int(row["previous_upgrade_requests"])
            upgrade_acc_rate_sim = float(row["upgrade_acceptance_rate"])
            spontaneous_sim = int(row["spontaneous_upgrade"])
            acq_channel_sim = str(row["acquisition_channel"])
            cause_sim = str(row["cause"])
            level_sim = str(row["socioeconomic_level"])
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Perfil del Donante**")
            age_sim = st.slider("Edad", 18, 80, 42)
            months_active_sim = st.slider("Meses activo", 1, 36, 12)
            current_amount_sim = st.number_input("Monto actual ($)", 1.0, 500.0, 20.0, 1.0)
            acq_channel_sim = st.selectbox("Canal adquisición", ["calle", "online", "evento", "referido", "telefono"])
            cause_sim = st.selectbox("Causa", ["infancia", "medio_ambiente", "salud", "educacion", "emergencias"])
            level_sim = st.selectbox("Nivel socioeconómico", ["bajo", "medio_bajo", "medio", "medio_alto", "alto"])
        with col2:
            st.markdown("**Historial de Pagos**")
            payment_success_rate_sim = st.slider("Tasa pago exitoso", 0.0, 1.0, 0.90, 0.01)
            consecutive_failed_sim = st.slider("Pagos fallidos consecutivos", 0, 5, 0)
        with col3:
            st.markdown("**Engagement**")
            email_open_sim = st.slider("Tasa apertura email", 0.0, 1.0, 0.35, 0.01)
            email_click_sim = st.slider("Tasa clic email", 0.0, 1.0, 0.10, 0.01)
            last_email_days_sim = st.slider("Días desde último email abierto", 0, 180, 15)
            complaint_sim = st.slider("Quejas registradas", 0, 3, 0)
            pause_sim = st.slider("Solicitudes de pausa", 0, 3, 0)
            prev_upgrades_sim = st.slider("Upgrades solicitados previos", 0, 5, 0)
            upgrade_acc_rate_sim = st.slider("Tasa aceptación upgrades previos", 0.0, 1.0, 0.0, 0.1)
            spontaneous_sim = st.selectbox("¿Subió monto espontáneamente?", [0, 1])

    if st.button("🔍 Analizar Donante", type="primary", use_container_width=True):
        le_sim = LabelEncoder()
        channel_list = ["calle", "online", "evento", "referido", "telefono"]
        cause_list = ["infancia", "medio_ambiente", "salud", "educacion", "emergencias"]
        level_list = ["bajo", "medio_bajo", "medio", "medio_alto", "alto"]

        ch_enc = channel_list.index(acq_channel_sim)
        ca_enc = cause_list.index(cause_sim)
        le_enc = level_list.index(level_sim)

        rfm_r = last_email_days_sim
        rfm_f = round(payment_success_rate_sim * 10, 1)
        rfm_m = round(np.log1p(current_amount_sim * months_active_sim), 4)
        amount_var = current_amount_sim * 0.05

        feature_vector = pd.DataFrame([{
            "age": age_sim, "months_active": months_active_sim,
            "payment_success_rate": payment_success_rate_sim,
            "consecutive_failed_payments": consecutive_failed_sim,
            "recency_days": last_email_days_sim, "email_open_rate": email_open_sim,
            "email_click_rate": email_click_sim, "last_email_open_days": last_email_days_sim,
            "complaint_history": complaint_sim, "pause_requests": pause_sim,
            "rfm_recency": rfm_r, "rfm_frequency": rfm_f, "rfm_monetary": rfm_m,
            "current_amount": current_amount_sim, "amount_variance": amount_var,
            "previous_upgrade_requests": prev_upgrades_sim,
            "upgrade_acceptance_rate": upgrade_acc_rate_sim,
            "acquisition_channel_enc": ch_enc, "cause_enc": ca_enc,
            "socioeconomic_level_enc": le_enc, "spontaneous_upgrade": spontaneous_sim
        }])

        # Predicciones
        p_churn_sim = churn_model.predict_proba(feature_vector[FEATURE_COLS])[:, 1][0]

        timing_fv = feature_vector[timing_features].fillna(0)
        timing_sim = int(round(timing_model.predict(timing_fv)[0]))

        amount_fv = feature_vector[amount_features].fillna(0)
        amount_sim = round(amount_model.predict(amount_fv)[0], 2)

        # Uplift
        p_treat = m_t.predict_proba(feature_vector[FEATURE_COLS])[:, 1][0]
        p_ctrl = m_c.predict_proba(feature_vector[FEATURE_COLS])[:, 1][0]
        uplift_sim = p_treat - p_ctrl

        # Segmento
        if p_churn_sim > churn_threshold and uplift_sim < 0:
            seg_sim = "sleeping_dog"
        elif uplift_sim >= uplift_threshold and p_churn_sim <= churn_threshold:
            seg_sim = "persuadable"
        elif spontaneous_sim == 1:
            seg_sim = "sure_thing"
        elif uplift_sim < 0:
            seg_sim = "lost_cause"
        else:
            seg_sim = "persuadable"

        # Decisión
        if p_churn_sim > churn_threshold:
            dec_sim = "🛡️ MODO PROTECCIÓN — No solicitar upgrade. Prioridad: retención."
            dec_color = COLORS["danger"]
        elif seg_sim == "sleeping_dog":
            dec_sim = "🐕 SLEEPING DOG — NUNCA intervenir. Alta sensibilidad al contacto."
            dec_color = COLORS["primary"]
        elif seg_sim == "lost_cause":
            dec_sim = "❌ LOST CAUSE — No intervenir. Baja probabilidad de respuesta."
            dec_color = COLORS["dark"]
        elif seg_sim == "sure_thing":
            dec_sim = "✅ SURE THING — Monitorear. El donante subirá su monto espontáneamente."
            dec_color = COLORS["secondary"]
        elif months_active_sim < timing_sim:
            dec_sim = f"⏳ ESPERAR — Intervenir en el mes {timing_sim} (faltan {timing_sim - months_active_sim} meses)."
            dec_color = COLORS["warning"]
        else:
            dec_sim = f"🚀 INTERVENIR AHORA — Solicitar incremento de ${current_amount_sim:.1f} → ${current_amount_sim + amount_sim:.1f} (+${amount_sim:.1f})"
            dec_color = COLORS["success"]

        st.markdown("---")
        st.markdown("### 📋 Resultado del Análisis")

        c1, c2, c3, c4 = st.columns(4)
        with c1: render_metric_card("P(Churn)", f"{p_churn_sim:.1%}",
                                     "⚠️ ALTO RIESGO" if p_churn_sim > churn_threshold else "✅ Bajo riesgo",
                                     COLORS["danger"] if p_churn_sim > churn_threshold else COLORS["success"], "🔴")
        with c2: render_metric_card("Uplift Score", f"{uplift_sim:+.3f}",
                                     "Favorable" if uplift_sim > 0 else "Desfavorable",
                                     COLORS["success"] if uplift_sim > 0 else COLORS["danger"], "🎯")
        with c3: render_metric_card("Mes Óptimo Intervención", f"Mes {timing_sim}",
                                     f"{'LISTO' if months_active_sim >= timing_sim else f'En {timing_sim - months_active_sim} meses'}",
                                     COLORS["teal"], "📅")
        with c4: render_metric_card("Incremento Sugerido", f"+${amount_sim:.2f}",
                                     f"${current_amount_sim:.1f} → ${current_amount_sim + amount_sim:.1f}",
                                     COLORS["success"], "💰")

        seg_color = SEGMENT_COLORS.get(seg_sim, "#999")
        seg_emoji = {"persuadable": "🎯", "sure_thing": "✅", "sleeping_dog": "🐕", "lost_cause": "❌"}.get(seg_sim, "❓")
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{seg_color}">
            <p class="kpi-label">Segmento Uplift</p>
            <p class="kpi-value" style="color:{seg_color}">{seg_emoji} {seg_sim.replace('_', ' ').upper()}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{dec_color};color:white;padding:20px 24px;border-radius:12px;
                    margin-top:12px;font-size:1.15rem;font-weight:700;">
            📌 DECISIÓN DEL PIPELINE: {dec_sim}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Pasos del Pipeline para este Donante**")
        steps = [
            (f"1️⃣ Guardia Churn", f"P(churn) = {p_churn_sim:.1%} | Umbral = {churn_threshold:.0%}",
             "✅ Pasa" if p_churn_sim <= churn_threshold else "❌ BLOQUEADO — Modo Protección",
             COLORS["success"] if p_churn_sim <= churn_threshold else COLORS["danger"]),
            (f"2️⃣ Timing Model", f"Mes actual = {months_active_sim} | Mes óptimo = {timing_sim}",
             "✅ Momento óptimo" if months_active_sim >= timing_sim else f"⏳ Esperar {timing_sim - months_active_sim} meses",
             COLORS["success"] if months_active_sim >= timing_sim else COLORS["warning"]),
            (f"3️⃣ Uplift Model", f"Uplift = {uplift_sim:+.3f} | Umbral = {uplift_threshold:.2f} | Segmento: {seg_sim}",
             "✅ Intervenir" if seg_sim == "persuadable" else f"⚠️ {seg_sim.replace('_', ' ').title()}",
             COLORS["success"] if seg_sim == "persuadable" else SEGMENT_COLORS.get(seg_sim, "#999")),
            (f"4️⃣ Amount Model", f"Monto actual = ${current_amount_sim:.1f} | Incremento sugerido = +${amount_sim:.2f}",
             f"✅ Solicitar ${current_amount_sim + amount_sim:.1f}/mes",
             COLORS["teal"]),
        ]
        for title, detail, result, color in steps:
            st.markdown(f"""
            <div style="background:white;border-left:4px solid {color};border-radius:8px;
                        padding:12px 16px;margin:6px 0;box-shadow:0 1px 4px rgba(0,0,0,0.08)">
                <b>{title}</b><br>
                <span style="color:#6c757d;font-size:0.9rem">{detail}</span><br>
                <span style="color:{color};font-weight:600">{result}</span>
            </div>
            """, unsafe_allow_html=True)





# =================================================
# EJECUTARLO EN BASH:
# =================================================
# OJO, EL PYTHON 3.11.11 INSTALADO EN ESTE SPYDER, ESTÁ UBICADO AQUÍ:
# "C:\Users\cesar\AppData\Local\spyder-6\envs\spyder-runtime\python.exe"
    
# ENTONCES LA EJECUCIÓN DE ESTE STREAMLIT, SE DEBE HACER DE LA SIGUIENTE MANERA:
    
# cd C:\Users\cesar\OneDrive\Escritorio\UNICEF\UPLIFT\UPFLIT_1
# "C:\Users\cesar\AppData\Local\spyder-6\envs\spyder-runtime\python.exe" -m streamlit run donation_maximization_app.py












