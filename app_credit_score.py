import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf

# ── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Credit Score",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Directorio del script (para cargar artefactos de forma robusta)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Rangos de entrenamiento (estadísticas del dataset riesgo.xlsx) ────────────
# Estos valores delimitan el espacio de entrada válido para el modelo.
# Fuente: df.describe() sobre el dataset de entrenamiento (12 500 filas).
TRAINING_RANGES = {
    "Age":                     {"min": 14.00,    "max": 56.00,      "default": 33.0,    "step": 0.5,    "fmt": "%.2f"},
    "Annual_Income":           {"min": 7005.93,  "max": 179987.28,  "default": 50505.0, "step": 500.0,  "fmt": "%.2f"},
    "Monthly_Inhand_Salary":   {"min": 303.65,   "max": 15204.63,   "default": 4198.0,  "step": 50.0,   "fmt": "%.2f"},
    "Num_Bank_Accounts":       {"min": 0.0,      "max": 10.5,       "default": 5.0,     "step": 0.5,    "fmt": "%.1f"},
    "Num_Credit_Card":         {"min": 0.5,      "max": 10.88,      "default": 5.0,     "step": 0.5,    "fmt": "%.1f"},
    "Interest_Rate":           {"min": 1,        "max": 34,         "default": 14,      "step": 1,      "fmt": "%d"},
    "Num_of_Loan":             {"min": 0,        "max": 9,          "default": 3,       "step": 1,      "fmt": "%d"},
    "Delay_from_due_date":     {"min": -2.0,     "max": 63.25,      "default": 21.0,    "step": 0.25,   "fmt": "%.2f"},
    "Num_of_Delayed_Payment":  {"min": 0.0,      "max": 26.38,      "default": 13.0,    "step": 0.5,    "fmt": "%.2f"},
    "Changed_Credit_Limit":    {"min": 0.5,      "max": 31.12,      "default": 10.0,    "step": 0.5,    "fmt": "%.2f"},
    "Num_Credit_Inquiries":    {"min": 0.0,      "max": 17.0,       "default": 5.0,     "step": 1.0,    "fmt": "%.1f"},
    "Outstanding_Debt":        {"min": 0.0,      "max": 4998.07,    "default": 1426.0,  "step": 50.0,   "fmt": "%.2f"},
    "Credit_Utilization_Ratio":{"min": 20.0,     "max": 50.0,       "default": 32.0,    "step": 0.5,    "fmt": "%.2f"},
    "Credit_History_Age":      {"min": 0.0,      "max": 403.0,      "default": 180.0,   "step": 1.0,    "fmt": "%.1f"},
    "Total_EMI_per_month":     {"min": 0.0,      "max": 1779.0,     "default": 145.0,   "step": 10.0,   "fmt": "%.2f"},
    "Amount_invested_monthly": {"min": 0.0,      "max": 10000.0,    "default": 200.0,   "step": 10.0,   "fmt": "%.2f"},
    "Monthly_Balance":         {"min": 0.0,      "max": 1500.0,     "default": 300.0,   "step": 10.0,   "fmt": "%.2f"},
}

# Opciones categóricas exactas del dataset de entrenamiento
OCCUPATION_OPTIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]
CREDIT_MIX_OPTIONS    = ["Bad", "Standard", "Good"]
PAYMENT_MIN_OPTIONS   = ["Yes", "No", "NM"]
PAYMENT_BEHAV_OPTIONS = [
    "High_spent_Large_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments",
    "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Small_value_payments",
]
# Tipos de préstamo disponibles en el dataset (para construir Type_of_Loan)
LOAN_TYPE_OPTIONS = [
    "Auto Loan",
    "Credit-Builder Loan",
    "Debt Consolidation Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Not Specified",
    "Payday Loan",
    "Personal Loan",
    "Student Loan",
]

# ── Carga de artefactos (cacheados para no recargar en cada interacción) ──────
@st.cache_resource
def load_artifacts():
    model         = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "modelo_credit_score.keras")
    )
    preprocesador = joblib.load(
        os.path.join(BASE_DIR, "preprocesador_credit_score.jb")
    )
    pca           = joblib.load(
        os.path.join(BASE_DIR, "pca_credit_score.jb")
    )
    return model, preprocesador, pca


# ── Helpers ───────────────────────────────────────────────────────────────────
def num_input(label, key, help_text=None):
    """Crea un number_input con los rangos exactos de entrenamiento."""
    cfg = TRAINING_RANGES[key]
    min_v, max_v = cfg["min"], cfg["max"]
    # Streamlit number_input no acepta formato entero con float min/max,
    # así que siempre usamos float para uniformidad.
    val = st.number_input(
        label=label,
        min_value=float(min_v),
        max_value=float(max_v),
        value=float(cfg["default"]),
        step=float(cfg["step"]),
        help=help_text or f"Rango de entrenamiento: [{min_v} – {max_v}]",
        key=key,
    )
    return val


def validate_inputs(inputs: dict) -> list:
    """Devuelve lista de advertencias si algún valor está fuera del rango."""
    warnings = []
    for col, val in inputs.items():
        if col in TRAINING_RANGES:
            rng = TRAINING_RANGES[col]
            if val < rng["min"] or val > rng["max"]:
                warnings.append(
                    f"**{col}** = {val} está fuera del rango de entrenamiento "
                    f"[{rng['min']} – {rng['max']}]."
                )
    return warnings


# ── Cabecera ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background: linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
                padding:28px 32px; border-radius:14px; margin-bottom:24px;">
        <h1 style="color:#e2e8f0; margin:0; font-size:2rem;">💳 Predictor de Puntaje de Crédito</h1>
        <p style="color:#94a3b8; margin:8px 0 0 0; font-size:1rem;">
            Red Neuronal Artificial · Clasificación Multiclase · Universidad Andrés Bello
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Completa los datos del cliente para predecir su categoría crediticia: "
    "**Poor** (Malo), **Standard** (Regular) o **Good** (Bueno).  \n"
    "Los campos tienen los **rangos exactos del dataset de entrenamiento** para garantizar "
    "predicciones fiables."
)

# ── Carga de artefactos ───────────────────────────────────────────────────────
try:
    model, preprocesador, pca = load_artifacts()
    artifacts_ok = True
except Exception as e:
    st.error(
        "No se pudieron cargar los artefactos del modelo. "
        "Asegúrate de ejecutar primero **EntrenamientoModelo.ipynb** (sección *Guardar artefactos*) "
        "y de que los archivos `.keras` y `.jb` estén en el mismo directorio que esta app.\n\n"
        f"**Error:** {e}"
    )
    artifacts_ok = False

if artifacts_ok:
    st.divider()

    # ── Formulario de entrada ─────────────────────────────────────────────────
    st.subheader("Datos del Cliente")

    col1, col2, col3 = st.columns(3)

    # ── Columna 1 · Información Personal y Financiera ─────────────────────────
    with col1:
        st.markdown("#### Información Personal")

        age = num_input("Edad (años)", "Age",
                        "Edad del cliente. Rango del dataset: 14 – 56.")

        occupation = st.selectbox(
            "Ocupación",
            options=OCCUPATION_OPTIONS,
            index=OCCUPATION_OPTIONS.index("Engineer"),
            help="Ocupación laboral del cliente.",
        )

        annual_income = num_input("Ingreso Anual (USD)", "Annual_Income",
                                  "Ingreso bruto anual declarado.")

        monthly_salary = num_input("Salario Mensual Neto (USD)", "Monthly_Inhand_Salary",
                                   "Salario líquido mensual recibido.")

        num_bank = num_input("N.º de Cuentas Bancarias", "Num_Bank_Accounts",
                             "Cantidad de cuentas bancarias activas.")

        num_card = num_input("N.º de Tarjetas de Crédito", "Num_Credit_Card",
                             "Cantidad de tarjetas de crédito activas.")

    # ── Columna 2 · Historial de Crédito y Pagos ─────────────────────────────
    with col2:
        st.markdown("#### Historial de Crédito y Pagos")

        interest_rate = num_input("Tasa de Interés (%)", "Interest_Rate",
                                  "Tasa de interés promedio de los créditos vigentes.")

        num_loan = num_input("N.º de Préstamos Activos", "Num_of_Loan",
                             "Cantidad de préstamos activos en el momento.")

        delay = num_input("Días Promedio de Retraso en Pagos", "Delay_from_due_date",
                          "Promedio de días de retraso respecto a la fecha de vencimiento.")

        num_delayed = num_input("N.º de Pagos Atrasados", "Num_of_Delayed_Payment",
                                "Número total de pagos realizados con atraso.")

        changed_limit = num_input("Cambio en Límite de Crédito (%)", "Changed_Credit_Limit",
                                  "Variación porcentual del límite de crédito en el último período.")

        num_inquiries = num_input("N.º de Consultas de Crédito", "Num_Credit_Inquiries",
                                  "Número de consultas de crédito realizadas.")

        credit_mix = st.selectbox(
            "Mezcla de Crédito (Credit Mix)",
            options=CREDIT_MIX_OPTIONS,
            index=1,
            help="Calidad de la combinación de tipos de crédito: Bad, Standard o Good.",
        )

    # ── Columna 3 · Deuda, Comportamiento y Tipos de Préstamo ────────────────
    with col3:
        st.markdown("#### Deuda, Comportamiento e Inversión")

        outstanding_debt = num_input("Deuda Pendiente (USD)", "Outstanding_Debt",
                                     "Total de deuda pendiente de pago.")

        util_ratio = num_input("Ratio de Utilización de Crédito (%)", "Credit_Utilization_Ratio",
                               "Porcentaje del crédito disponible que está siendo utilizado.")

        credit_hist_age = num_input("Antigüedad del Historial Crediticio (meses)", "Credit_History_Age",
                                    "Tiempo en meses desde la apertura de la primera cuenta de crédito.")

        payment_min = st.selectbox(
            "¿Paga el Mínimo Mensual?",
            options=PAYMENT_MIN_OPTIONS,
            index=0,
            help="Yes = siempre paga el mínimo, No = no lo paga, NM = no corresponde.",
        )

        total_emi = num_input("EMI Total por Mes (USD)", "Total_EMI_per_month",
                              "Suma de todas las cuotas mensuales de préstamos (EMI).")

        amount_invested = num_input("Inversión Mensual (USD)", "Amount_invested_monthly",
                                    "Monto promedio invertido mensualmente.")

        payment_behaviour = st.selectbox(
            "Comportamiento de Pago",
            options=PAYMENT_BEHAV_OPTIONS,
            index=1,
            help="Patrón de comportamiento en los pagos del cliente.",
        )

        monthly_balance = num_input("Balance Mensual (USD)", "Monthly_Balance",
                                    "Saldo disponible promedio al final del mes.")

    # ── Tipos de préstamo (multiselect) ───────────────────────────────────────
    st.markdown("#### Tipos de Préstamo")
    loan_types_selected = st.multiselect(
        "Selecciona los tipos de préstamo que posee el cliente",
        options=LOAN_TYPE_OPTIONS,
        default=[],
        help=(
            "Selecciona todos los tipos de préstamo vigentes. "
            "El modelo usa el conteo de tipos seleccionados como característica."
        ),
    )
    # Construir Type_of_Loan como string separado por comas (igual que en el dataset)
    type_of_loan_str = ", ".join(loan_types_selected) if loan_types_selected else None

    st.divider()

    # ── Validación en tiempo real ─────────────────────────────────────────────
    numeric_inputs = {
        "Age":                     float(age),
        "Annual_Income":           float(annual_income),
        "Monthly_Inhand_Salary":   float(monthly_salary),
        "Num_Bank_Accounts":       float(num_bank),
        "Num_Credit_Card":         float(num_card),
        "Interest_Rate":           float(interest_rate),
        "Num_of_Loan":             float(num_loan),
        "Delay_from_due_date":     float(delay),
        "Num_of_Delayed_Payment":  float(num_delayed),
        "Changed_Credit_Limit":    float(changed_limit),
        "Num_Credit_Inquiries":    float(num_inquiries),
        "Outstanding_Debt":        float(outstanding_debt),
        "Credit_Utilization_Ratio":float(util_ratio),
        "Credit_History_Age":      float(credit_hist_age),
        "Total_EMI_per_month":     float(total_emi),
        "Amount_invested_monthly": float(amount_invested),
        "Monthly_Balance":         float(monthly_balance),
    }
    range_warnings = validate_inputs(numeric_inputs)
    if range_warnings:
        with st.expander("⚠️ Advertencias de rango (los valores están fuera del rango de entrenamiento)", expanded=True):
            for w in range_warnings:
                st.warning(w)

    # ── Botón de predicción ───────────────────────────────────────────────────
    predict_btn = st.button(
        "🔍 Predecir Credit Score",
        type="primary",
        use_container_width=True,
        disabled=bool(range_warnings),  # Deshabilitar si hay valores fuera de rango
    )

    if range_warnings:
        st.caption(
            "El botón de predicción está deshabilitado porque uno o más valores "
            "están fuera del rango de entrenamiento. Ajusta los valores para continuar."
        )

    # ── Predicción ────────────────────────────────────────────────────────────
    if predict_btn:
        # Construir DataFrame exactamente igual que en el notebook
        nuevo_cliente = pd.DataFrame([{
            "Age":                     float(age),
            "Occupation":              occupation,
            "Annual_Income":           float(annual_income),
            "Monthly_Inhand_Salary":   float(monthly_salary),
            "Num_Bank_Accounts":       float(num_bank),
            "Num_Credit_Card":         float(num_card),
            "Interest_Rate":           int(interest_rate),
            "Num_of_Loan":             int(num_loan),
            "Type_of_Loan":            type_of_loan_str,     # puede ser None (→ NaN)
            "Delay_from_due_date":     float(delay),
            "Num_of_Delayed_Payment":  float(num_delayed),
            "Changed_Credit_Limit":    float(changed_limit),
            "Num_Credit_Inquiries":    float(num_inquiries),
            "Credit_Mix":              credit_mix,
            "Outstanding_Debt":        float(outstanding_debt),
            "Credit_Utilization_Ratio":float(util_ratio),
            "Credit_History_Age":      float(credit_hist_age),
            "Payment_of_Min_Amount":   payment_min,
            "Total_EMI_per_month":     float(total_emi),
            "Amount_invested_monthly": float(amount_invested),
            "Payment_Behaviour":       payment_behaviour,
            "Monthly_Balance":         float(monthly_balance),
        }])

        try:
            with st.spinner("Procesando datos y generando predicción…"):
                # Pipeline idéntico al notebook
                X_prep = preprocesador.transform(nuevo_cliente)
                X_pca  = pca.transform(X_prep)
                proba  = model.predict(X_pca, verbose=0)[0]
                pred   = int(np.argmax(proba))

            # ── Resultado ──────────────────────────────────────────────────────
            label_map   = {0: "Poor",    1: "Standard", 2: "Good"}
            label_es    = {0: "Malo",    1: "Regular",  2: "Bueno"}
            emoji_map   = {0: "🔴",      1: "🟡",        2: "🟢"}
            color_map   = {0: "#e74c3c", 1: "#f39c12",  2: "#2ecc71"}
            bg_map      = {0: "#fdf2f2", 1: "#fffbf0",  2: "#f0fdf4"}
            message_map = {
                0: (
                    "El cliente presenta un **riesgo crediticio alto**. "
                    "Se recomienda revisar el historial de pagos, reducir deudas "
                    "pendientes y evitar nuevas consultas de crédito antes de "
                    "otorgar nuevos productos financieros."
                ),
                1: (
                    "El cliente tiene un **perfil de crédito estándar**. "
                    "Puede acceder a productos crediticios con condiciones normales "
                    "de mercado. Se sugiere mejorar la puntualidad en los pagos "
                    "para escalar a la categoría Good."
                ),
                2: (
                    "El cliente tiene un **excelente historial crediticio**. "
                    "Es candidato ideal para créditos con tasas preferenciales, "
                    "mayores límites y condiciones diferenciadas."
                ),
            }

            st.markdown("## Resultado de la Predicción")

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.markdown(
                    f"<div style='"
                    f"text-align:center; padding:28px 20px; border-radius:14px; "
                    f"background-color:{bg_map[pred]}; border: 2.5px solid {color_map[pred]};'>"
                    f"<span style='font-size:72px; line-height:1;'>{emoji_map[pred]}</span><br><br>"
                    f"<span style='font-size:26px; font-weight:700; color:{color_map[pred]};'>"
                    f"{label_map[pred]}</span><br>"
                    f"<span style='font-size:18px; color:#666;'>{label_es[pred]}</span><br>"
                    f"<span style='font-size:13px; color:#888; margin-top:6px; display:block;'>"
                    f"Clase {pred}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Confianza de la predicción
                confianza = float(proba[pred]) * 100
                st.markdown(
                    f"<p style='text-align:center; margin-top:12px; color:#555;'>"
                    f"Confianza: <strong>{confianza:.1f}%</strong></p>",
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.info(message_map[pred])

                st.markdown("**Probabilidades por clase:**")
                prob_df = pd.DataFrame({
                    "Clase": [
                        f"{emoji_map[i]} {label_map[i]} ({label_es[i]})"
                        for i in range(3)
                    ],
                    "Probabilidad": proba.astype(float),
                }).set_index("Clase")
                st.bar_chart(prob_df, height=220)

                st.markdown("**Detalle de probabilidades:**")
                for i in range(3):
                    pct = float(proba[i])
                    st.write(
                        f"{emoji_map[i]} **{label_map[i]}** ({label_es[i]}): "
                        f"{pct * 100:.2f}%"
                    )
                    st.progress(pct)

        except Exception as e:
            st.error(
                f"Ocurrió un error al realizar la predicción.\n\n**Error:** {e}"
            )

    # ── Información del modelo (pie de página) ────────────────────────────────
    st.divider()
    with st.expander("ℹ️ Información del Modelo", expanded=False):
        st.markdown(
            """
            **Arquitectura:** Red Neuronal Secuencial (Keras)
            - Entrada: 36 componentes principales (PCA, 95 % varianza explicada)
            - Capa 1: Dense(128, ReLU) + Dropout(0.3)
            - Capa 2: Dense(64, ReLU) + Dropout(0.2)
            - Capa 3: Dense(32, ReLU)
            - Salida: Dense(3, Softmax) → Poor / Standard / Good

            **Pipeline de preprocesamiento:**
            1. Columnas numéricas: imputación por mediana + StandardScaler
            2. Columnas categóricas: imputación por moda + OneHotEncoder
            3. PCA con n_components=0.95 → 36 componentes

            **Dataset de entrenamiento:** 12 500 muestras · 22 características originales  
            **Accuracy en test (20 %):** ≈ 74.6 %  
            **Clases objetivo:** 0 = Poor (33 %) · 1 = Standard (49 %) · 2 = Good (18 %)
            """
        )
