import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 2000

# ─── Perfil base del donante ────────────────────────────────────────────────
donor_id = [f"DON-{i:04d}" for i in range(1, N + 1)]

acquisition_channel = np.random.choice(
    ["calle", "online", "evento", "referido", "telefono"],
    N, p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

cause = np.random.choice(
    ["infancia", "medio_ambiente", "salud", "educacion", "emergencias"],
    N, p=[0.30, 0.20, 0.20, 0.20, 0.10]
)

age = np.random.normal(42, 13, N).clip(18, 80).astype(int)

gender = np.random.choice(["M", "F", "otro"], N, p=[0.45, 0.50, 0.05])

socioeconomic_level = np.random.choice(
    ["bajo", "medio_bajo", "medio", "medio_alto", "alto"],
    N, p=[0.10, 0.20, 0.35, 0.25, 0.10]
)

# ─── Monto base de donación ─────────────────────────────────────────────────
base_amount_map = {
    "bajo": (3, 8), "medio_bajo": (5, 15), "medio": (10, 30),
    "medio_alto": (25, 60), "alto": (50, 200)
}
initial_amount = np.array([
    np.random.uniform(*base_amount_map[sl]) for sl in socioeconomic_level
]).round(2)

# ─── Historial de pagos ─────────────────────────────────────────────────────
months_active = np.random.randint(3, 37, N)

# Probabilidad de pago exitoso según canal
payment_success_prob_map = {
    "calle": 0.88, "online": 0.95, "evento": 0.90,
    "referido": 0.93, "telefono": 0.85
}
base_payment_prob = np.array([payment_success_prob_map[c] for c in acquisition_channel])

successful_payments = np.array([
    np.random.binomial(months_active[i], base_payment_prob[i])
    for i in range(N)
])
failed_payments = months_active - successful_payments
payment_success_rate = (successful_payments / months_active.clip(min=1)).round(4)

# ─── Varianza y tendencia del monto ─────────────────────────────────────────
amount_variance = np.random.exponential(initial_amount * 0.05, N).round(2)
amount_trend = np.random.normal(0.01, 0.03, N)  # % de cambio mensual promedio

current_amount = (initial_amount * (1 + amount_trend * months_active)).clip(min=1).round(2)
spontaneous_upgrade = (amount_trend > 0.02).astype(int)  # Subió solo sin que se le pidiera

# ─── Engagement / comunicaciones ────────────────────────────────────────────
emails_sent = (months_active * np.random.uniform(0.8, 1.2, N)).astype(int)
email_open_rate = np.random.beta(3, 5, N).round(4)
email_click_rate = (email_open_rate * np.random.beta(2, 6, N)).round(4)
emails_opened = (emails_sent * email_open_rate).astype(int)
emails_clicked = (emails_sent * email_click_rate).astype(int)

last_email_open_days = np.random.exponential(30, N).clip(0, 180).astype(int)
last_non_transactional_interaction_days = np.random.exponential(45, N).clip(0, 365).astype(int)

# ─── Historial de maximizaciones previas ────────────────────────────────────
previous_upgrade_requests = np.where(months_active >= 6,
    np.random.randint(0, 4, N), 0)
upgrade_accepted = np.array([
    np.random.binomial(previous_upgrade_requests[i],
                       0.35 + email_open_rate[i] * 0.3)
    for i in range(N)
])
upgrade_acceptance_rate = np.where(
    previous_upgrade_requests > 0,
    (upgrade_accepted / previous_upgrade_requests.clip(min=1)).round(4),
    np.nan
)

months_since_last_upgrade_request = np.where(
    previous_upgrade_requests > 0,
    np.random.randint(1, months_active + 1, N).clip(max=months_active),
    np.nan
)

# ─── RFM adaptado ───────────────────────────────────────────────────────────
recency_days = np.random.exponential(20, N).clip(0, 90).astype(int)  # días desde último pago
frequency_score = (payment_success_rate * 10).round(1)
monetary_score = np.log1p(current_amount * months_active).round(4)

# ─── Señales de riesgo ──────────────────────────────────────────────────────
consecutive_failed_payments = np.random.choice([0, 1, 2, 3], N, p=[0.70, 0.18, 0.08, 0.04])
complaint_history = np.random.choice([0, 1, 2], N, p=[0.85, 0.12, 0.03])
pause_requests = np.random.choice([0, 1, 2], N, p=[0.80, 0.15, 0.05])

# ─── Variables del experimento (tratamiento/control) ────────────────────────
# Grupo 0 = control (no recibió solicitud), Grupo 1 = tratamiento
experiment_group = np.random.choice([0, 1], N, p=[0.40, 0.60])

# ─── Label: Respuesta al tratamiento (para Uplift) ──────────────────────────
# Factores que incrementan probabilidad de aceptar upgrade
p_accept_base = 0.30
p_accept = (
    p_accept_base
    + email_open_rate * 0.25
    + payment_success_rate * 0.20
    - consecutive_failed_payments * 0.08
    - complaint_history * 0.10
    - pause_requests * 0.07
    + (cause == "infancia").astype(float) * 0.05
    + (age.clip(25, 60) - 25) / 35 * 0.05
).clip(0.02, 0.92)

upgrade_response = np.where(
    experiment_group == 1,
    np.random.binomial(1, p_accept),
    0  # control no recibe tratamiento
)

# ─── Label: Churn (abandono) ────────────────────────────────────────────────
p_churn = (
    0.10
    + consecutive_failed_payments * 0.12
    + complaint_history * 0.10
    + pause_requests * 0.08
    - payment_success_rate * 0.10
    - email_open_rate * 0.08
    + (recency_days > 45).astype(float) * 0.10
    + (experiment_group == 1) * (1 - p_accept) * 0.05  # Sleeping dogs
).clip(0.01, 0.90)

churned = np.random.binomial(1, p_churn)

# ─── Label: Monto óptimo de solicitud de upgrade ────────────────────────────
elasticity = (
    0.15
    + (socioeconomic_level == "alto").astype(float) * 0.20
    + (socioeconomic_level == "medio_alto").astype(float) * 0.12
    + email_open_rate * 0.10
    + payment_success_rate * 0.08
    - consecutive_failed_payments * 0.03
).clip(0.05, 0.60)

optimal_upgrade_amount = (current_amount * elasticity).round(2)

# ─── Timing: mes óptimo estimado para intervención ─────────────────────────
optimal_intervention_month = (
    6
    + (payment_success_rate > 0.90).astype(int) * (-1)
    + (email_open_rate > 0.40).astype(int) * (-1)
    + (consecutive_failed_payments > 0).astype(int) * 3
    + (complaint_history > 0).astype(int) * 4
    + np.random.randint(-1, 2, N)
).clip(3, 24)

# ─── Segmento Uplift (ground truth simulado) ─────────────────────────────────
def assign_uplift_segment(i):
    if p_churn[i] > 0.45 and experiment_group[i] == 1:
        return "sleeping_dog"
    elif p_accept[i] > 0.55 and upgrade_response[i] == 1:
        return "persuadable"
    elif spontaneous_upgrade[i] == 1:
        return "sure_thing"
    elif p_accept[i] < 0.20:
        return "lost_cause"
    else:
        return "persuadable"

uplift_segment = [assign_uplift_segment(i) for i in range(N)]

# ─── Ensamblado del DataFrame ────────────────────────────────────────────────
df = pd.DataFrame({
    # Identificadores
    "donor_id": donor_id,
    "acquisition_channel": acquisition_channel,
    "cause": cause,
    "gender": gender,
    "age": age,
    "socioeconomic_level": socioeconomic_level,

    # Financiero
    "initial_amount": initial_amount,
    "current_amount": current_amount,
    "amount_variance": amount_variance,
    "amount_trend_monthly_pct": amount_trend.round(4),
    "spontaneous_upgrade": spontaneous_upgrade,
    "optimal_upgrade_amount": optimal_upgrade_amount,
    "elasticity_score": elasticity.round(4),

    # Historial de pagos
    "months_active": months_active,
    "successful_payments": successful_payments,
    "failed_payments": failed_payments,
    "payment_success_rate": payment_success_rate,
    "consecutive_failed_payments": consecutive_failed_payments,
    "recency_days": recency_days,

    # Engagement
    "emails_sent": emails_sent,
    "emails_opened": emails_opened,
    "emails_clicked": emails_clicked,
    "email_open_rate": email_open_rate,
    "email_click_rate": email_click_rate,
    "last_email_open_days": last_email_open_days,
    "last_non_transactional_interaction_days": last_non_transactional_interaction_days,

    # RFM
    "rfm_recency": recency_days,
    "rfm_frequency": frequency_score,
    "rfm_monetary": monetary_score,

    # Señales de riesgo
    "complaint_history": complaint_history,
    "pause_requests": pause_requests,

    # Maximizaciones previas
    "previous_upgrade_requests": previous_upgrade_requests,
    "upgrade_accepted": upgrade_accepted,
    "upgrade_acceptance_rate": upgrade_acceptance_rate,
    "months_since_last_upgrade_request": months_since_last_upgrade_request,

    # Experimento
    "experiment_group": experiment_group,  # 0=control, 1=tratamiento
    "optimal_intervention_month": optimal_intervention_month,

    # Labels / Targets
    "upgrade_response": upgrade_response,       # Para Uplift Model
    "churned": churned,                         # Para Churn Model
    "uplift_segment": uplift_segment,           # Segmento Uplift
})

# ─── Exportar ─────────────────────────────────────────────────────────────────
df.to_csv("/mnt/user-data/outputs/donors_dataset.csv", index=False)

with pd.ExcelWriter("/mnt/user-data/outputs/donors_dataset.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Donors", index=False)

    # Hoja de diccionario de variables
    dict_data = {
        "Variable": [
            "donor_id", "acquisition_channel", "cause", "gender", "age",
            "socioeconomic_level", "initial_amount", "current_amount",
            "amount_variance", "amount_trend_monthly_pct", "spontaneous_upgrade",
            "optimal_upgrade_amount", "elasticity_score", "months_active",
            "successful_payments", "failed_payments", "payment_success_rate",
            "consecutive_failed_payments", "recency_days", "emails_sent",
            "emails_opened", "emails_clicked", "email_open_rate", "email_click_rate",
            "last_email_open_days", "last_non_transactional_interaction_days",
            "rfm_recency", "rfm_frequency", "rfm_monetary",
            "complaint_history", "pause_requests", "previous_upgrade_requests",
            "upgrade_accepted", "upgrade_acceptance_rate",
            "months_since_last_upgrade_request", "experiment_group",
            "optimal_intervention_month", "upgrade_response", "churned", "uplift_segment"
        ],
        "Tipo": [
            "ID", "Categórica", "Categórica", "Categórica", "Numérica",
            "Categórica", "Numérica", "Numérica", "Numérica", "Numérica", "Binaria",
            "Numérica", "Numérica", "Numérica", "Numérica", "Numérica", "Numérica",
            "Numérica", "Numérica", "Numérica", "Numérica", "Numérica",
            "Numérica", "Numérica", "Numérica", "Numérica",
            "Numérica", "Numérica", "Numérica",
            "Numérica", "Numérica", "Numérica", "Numérica", "Numérica",
            "Numérica", "Binaria", "Numérica", "Binaria", "Binaria", "Categórica"
        ],
        "Descripción": [
            "Identificador único del donante",
            "Canal por el que se adquirió al donante",
            "Causa a la que dona",
            "Género del donante",
            "Edad del donante",
            "Nivel socioeconómico estimado",
            "Monto inicial de donación mensual (unidades monetarias)",
            "Monto actual de donación mensual",
            "Varianza histórica del monto donado",
            "Tendencia promedio de cambio mensual del monto (%)",
            "1 si el donante subió su monto espontáneamente",
            "Monto óptimo estimado para solicitar como incremento",
            "Score de elasticidad al incremento (0-1)",
            "Meses que lleva activo como donante",
            "Número de pagos exitosos",
            "Número de pagos fallidos",
            "Tasa de pagos exitosos sobre el total",
            "Pagos fallidos consecutivos más recientes",
            "Días desde el último pago registrado",
            "Emails enviados al donante",
            "Emails abiertos",
            "Emails en los que hizo clic",
            "Tasa de apertura de emails",
            "Tasa de clics en emails",
            "Días desde la última apertura de email",
            "Días desde la última interacción no transaccional",
            "RFM: Recencia (días desde último pago)",
            "RFM: Frecuencia (score 0-10)",
            "RFM: Monetario (log del valor total acumulado)",
            "Número de quejas registradas",
            "Número de solicitudes de pausa",
            "Número de veces que se solicitó upgrade previo",
            "Número de upgrades aceptados",
            "Tasa de aceptación de upgrades anteriores",
            "Meses desde la última solicitud de upgrade",
            "Grupo experimental (0=control, 1=tratamiento)",
            "Mes óptimo estimado para intervención",
            "TARGET: 1 si aceptó el upgrade (grupo tratamiento)",
            "TARGET: 1 si abandonó (churn)",
            "Segmento Uplift: persuadable/sure_thing/lost_cause/sleeping_dog"
        ],
        "Modelo": [
            "—", "Todos", "Todos", "Todos", "Todos",
            "Todos", "Todos", "Todos", "Churn/Uplift", "Timing/Monto", "Timing",
            "Monto", "Monto", "Todos", "Churn/Uplift", "Churn/Uplift", "Churn/Uplift",
            "Churn", "Churn/Uplift", "Uplift/Churn", "Uplift/Churn", "Uplift/Churn",
            "Uplift/Churn", "Uplift/Churn", "Churn", "Churn",
            "Churn", "Churn", "Todos",
            "Churn", "Churn", "Uplift", "Uplift", "Uplift",
            "Timing", "Uplift", "Timing", "Uplift (target)", "Churn (target)", "Uplift (target)"
        ]
    }
    pd.DataFrame(dict_data).to_excel(writer, sheet_name="Diccionario", index=False)

print(f"✅ Dataset generado: {N} donantes, {df.shape[1]} variables")
print(f"   Churn rate: {df['churned'].mean():.1%}")
print(f"   Upgrade acceptance (tratamiento): {df[df['experiment_group']==1]['upgrade_response'].mean():.1%}")
print(f"   Distribución uplift_segment:\n{df['uplift_segment'].value_counts().to_string()}")
print(f"\n📁 Archivos guardados en /mnt/user-data/outputs/")
