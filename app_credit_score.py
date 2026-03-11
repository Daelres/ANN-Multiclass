"""
Aplicación Streamlit: Predicción de Credit Score
Universidad Andrés Bello — Ciencia de Datos

Uso:
    streamlit run app_credit_score.py

Requiere los siguientes artefactos en el mismo directorio:
    - modelo_credit_score.keras
    - scaler_credit_score.jb
    - pca_credit_score.jb
    - le_occupation.jb
    - le_payment_behaviour.jb
    - feature_columns.jb
"""

import os
import numpy as np
import joblib
import streamlit as st
import tensorflow as tf

# ── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Credit Score",
    page_icon="💳",
    layout="wide"
)

# Directorio del script (para cargar artefactos de forma robusta)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Carga de artefactos (cacheados para no recargar en cada interacción) ──────
@st.cache_resource
def load_artifacts():
    model    = tf.keras.models.load_model(os.path.join(BASE_DIR, 'modelo_credit_score.keras'))
    scaler   = joblib.load(os.path.join(BASE_DIR, 'scaler_credit_score.jb'))
    pca      = joblib.load(os.path.join(BASE_DIR, 'pca_credit_score.jb'))
    le_occ   = joblib.load(os.path.join(BASE_DIR, 'le_occupation.jb'))
    le_pb    = joblib.load(os.path.join(BASE_DIR, 'le_payment_behaviour.jb'))
    feat_cols = joblib.load(os.path.join(BASE_DIR, 'feature_columns.jb'))
    return model, scaler, pca, le_occ, le_pb, feat_cols


# ── Interfaz principal ────────────────────────────────────────────────────────
st.title("💳 Predictor de Puntaje de Crédito")
st.markdown(
    "Completa los datos del cliente para predecir su categoría de crédito: "
    "**Poor** (Malo), **Standard** (Regular) o **Good** (Bueno)."
)
st.divider()

# Cargar artefactos
try:
    model, scaler, pca, le_occ, le_pb, feat_cols = load_artifacts()
    artifacts_ok = True
except Exception as e:
    st.error(
        f"No se pudieron cargar los artefactos del modelo. "
        f"Asegúrate de ejecutar primero el notebook y de que los archivos "
        f"`.keras` y `.jb` se encuentren en el mismo directorio que esta app.\n\n"
        f"**Error:** {e}"
    )
    artifacts_ok = False

if artifacts_ok:
    # ── Opciones para selectboxes ─────────────────────────────────────────────
    occupation_options       = list(le_occ.classes_)
    payment_behaviour_options = list(le_pb.classes_)
    credit_mix_options       = ["Bad", "Standard", "Good"]

    # ── Formulario de entrada ─────────────────────────────────────────────────
    st.subheader("Datos del Cliente")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Información Personal**")
        age = st.number_input("Edad", min_value=18, max_value=100, value=30, step=1)
        occupation = st.selectbox("Ocupación", options=occupation_options)
        annual_income = st.number_input(
            "Ingreso Anual (USD)", min_value=0.0, value=45000.0, step=1000.0, format="%.2f"
        )
        monthly_inhand_salary = st.number_input(
            "Salario Mensual Neto (USD)", min_value=0.0, value=3500.0, step=100.0, format="%.2f"
        )
        num_bank_accounts = st.number_input(
            "Nº de Cuentas Bancarias", min_value=0, max_value=20, value=3, step=1
        )
        num_credit_card = st.number_input(
            "Nº de Tarjetas de Crédito", min_value=0, max_value=20, value=4, step=1
        )
        interest_rate = st.number_input(
            "Tasa de Interés (%)", min_value=0.0, max_value=50.0, value=14.0, step=0.5, format="%.1f"
        )
        num_of_loan = st.number_input(
            "Nº de Préstamos Activos", min_value=0, max_value=20, value=3, step=1
        )

    with col2:
        st.markdown("**Historial de Pagos**")
        delay_from_due_date = st.number_input(
            "Días Promedio de Retraso en Pagos", min_value=0, max_value=60, value=15, step=1
        )
        num_of_delayed_payment = st.number_input(
            "Nº de Pagos Atrasados", min_value=0, max_value=30, value=7, step=1
        )
        changed_credit_limit = st.number_input(
            "Cambio en Límite de Crédito (%)", min_value=-50.0, max_value=50.0, value=10.0, step=0.5, format="%.1f"
        )
        num_credit_inquiries = st.number_input(
            "Nº de Consultas de Crédito", min_value=0, max_value=20, value=5, step=1
        )
        credit_mix = st.selectbox("Mezcla de Crédito (Credit Mix)", options=credit_mix_options)
        outstanding_debt = st.number_input(
            "Deuda Pendiente (USD)", min_value=0.0, value=1500.0, step=100.0, format="%.2f"
        )
        credit_utilization_ratio = st.number_input(
            "Ratio de Utilización de Crédito (%)", min_value=0.0, max_value=100.0, value=28.0, step=1.0, format="%.1f"
        )
        payment_of_min_amount = st.selectbox(
            "¿Paga el Mínimo Mensual?", options=["Sí", "No"], index=0
        )

    with col3:
        st.markdown("**Gastos y Comportamiento**")
        total_emi_per_month = st.number_input(
            "EMI Total por Mes (USD)", min_value=0.0, value=200.0, step=10.0, format="%.2f"
        )
        amount_invested_monthly = st.number_input(
            "Inversión Mensual (USD)", min_value=0.0, value=80.0, step=10.0, format="%.2f"
        )
        payment_behaviour = st.selectbox("Comportamiento de Pago", options=payment_behaviour_options)
        monthly_balance = st.number_input(
            "Balance Mensual (USD)", min_value=0.0, value=300.0, step=50.0, format="%.2f"
        )
        credit_history_age_months = st.number_input(
            "Antigüedad del Historial Crediticio (meses)", min_value=0, max_value=500, value=180, step=1
        )
        num_loan_types = st.number_input(
            "Nº de Tipos de Préstamo", min_value=0, max_value=10, value=2, step=1
        )

    st.divider()

    # ── Botón de predicción ───────────────────────────────────────────────────
    predict_btn = st.button("🔍 Predecir Credit Score", type="primary", use_container_width=True)

    if predict_btn:
        # Codificar inputs categóricos
        occupation_encoded        = le_occ.transform([occupation])[0]
        payment_behaviour_encoded = le_pb.transform([payment_behaviour])[0]
        credit_mix_encoded        = credit_mix_options.index(credit_mix)        # Bad=0, Standard=1, Good=2
        payment_min_encoded       = 1 if payment_of_min_amount == "Sí" else 0

        # Construir diccionario con todos los features en el orden correcto
        input_dict = {
            'Age':                      age,
            'Occupation':               occupation_encoded,
            'Annual_Income':            annual_income,
            'Monthly_Inhand_Salary':    monthly_inhand_salary,
            'Num_Bank_Accounts':        num_bank_accounts,
            'Num_Credit_Card':          num_credit_card,
            'Interest_Rate':            interest_rate,
            'Num_of_Loan':              num_of_loan,
            'Delay_from_due_date':      delay_from_due_date,
            'Num_of_Delayed_Payment':   num_of_delayed_payment,
            'Changed_Credit_Limit':     changed_credit_limit,
            'Num_Credit_Inquiries':     num_credit_inquiries,
            'Credit_Mix':               credit_mix_encoded,
            'Outstanding_Debt':         outstanding_debt,
            'Credit_Utilization_Ratio': credit_utilization_ratio,
            'Payment_of_Min_Amount':    payment_min_encoded,
            'Total_EMI_per_month':      total_emi_per_month,
            'Amount_invested_monthly':  amount_invested_monthly,
            'Payment_Behaviour':        payment_behaviour_encoded,
            'Monthly_Balance':          monthly_balance,
            'Credit_History_Age_Months': credit_history_age_months,
            'Num_Loan_Types':           num_loan_types,
        }

        # Crear array en el orden exacto de FEATURE_COLUMNS
        input_values = np.array([[input_dict[col] for col in feat_cols]], dtype=np.float64)

        # Pipeline de transformación
        input_scaled = scaler.transform(input_values)
        input_pca    = pca.transform(input_scaled)

        # Predicción
        proba = model.predict(input_pca, verbose=0)[0]
        pred  = int(np.argmax(proba))

        # ── Resultado ────────────────────────────────────────────────────────
        label_map   = {0: "Poor",     1: "Standard",  2: "Good"}
        emoji_map   = {0: "🔴",       1: "🟡",         2: "🟢"}
        color_map   = {0: "#e74c3c",  1: "#f39c12",   2: "#2ecc71"}
        message_map = {
            0: "El cliente presenta un **riesgo crediticio alto**. Se recomienda revisar su historial de pagos y reducir deudas pendientes antes de otorgar nuevos créditos.",
            1: "El cliente tiene un **perfil de crédito estándar**. Puede acceder a productos crediticios con condiciones normales de mercado.",
            2: "El cliente tiene un **excelente historial crediticio**. Es candidato ideal para créditos con tasas preferenciales.",
        }

        st.markdown("## Resultado de la Predicción")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(
                f"<div style='text-align:center; padding:20px; border-radius:12px; "
                f"background-color:{color_map[pred]}22; border: 2px solid {color_map[pred]};'>"
                f"<span style='font-size:64px;'>{emoji_map[pred]}</span><br>"
                f"<span style='font-size:28px; font-weight:bold; color:{color_map[pred]};'>"
                f"{label_map[pred]}</span><br>"
                f"<span style='font-size:16px; color:#555;'>Clase {pred}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        with res_col2:
            st.info(message_map[pred])
            st.markdown("**Probabilidades por clase:**")

            import pandas as pd
            prob_df = pd.DataFrame({
                "Clase": [f"{emoji_map[i]} {label_map[i]}" for i in range(3)],
                "Probabilidad": proba
            }).set_index("Clase")
            st.bar_chart(prob_df, height=200)

            st.markdown("**Detalle de probabilidades:**")
            for i in range(3):
                bar_pct = int(proba[i] * 100)
                st.write(f"{emoji_map[i]} **{label_map[i]}**: {proba[i]*100:.2f}%")
                st.progress(float(proba[i]))
