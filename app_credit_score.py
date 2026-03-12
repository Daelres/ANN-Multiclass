import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Predicción de Credit Score",
    page_icon="💳",
    layout="wide"
)

# =========================
# FUNCIONES AUXILIARES DE METADATA
# =========================
def completar_metadata(metadata):
    """
    Completa llaves faltantes en metadata para evitar errores
    cuando el archivo metadata_credit_score.jb no contiene
    'default_values' o 'dtypes'.
    """
    numeric_columns = metadata.get("numeric_columns", [])
    categorical_columns = metadata.get("categorical_columns", [])
    numeric_ranges = metadata.get("numeric_ranges", {})
    categorical_options = metadata.get("categorical_options", {})

    # Completar default_values si no existe
    if "default_values" not in metadata:
        metadata["default_values"] = {}

    # Completar dtypes si no existe
    if "dtypes" not in metadata:
        metadata["dtypes"] = {}

    # Valores por defecto y tipos para numéricas
    for col in numeric_columns:
        if col not in metadata["default_values"]:
            min_val = numeric_ranges[col]["min"]
            max_val = numeric_ranges[col]["max"]
            metadata["default_values"][col] = (min_val + max_val) / 2

        if col not in metadata["dtypes"]:
            min_val = numeric_ranges[col]["min"]
            max_val = numeric_ranges[col]["max"]

            # Si min y max son enteros exactos, tratamos como int
            if float(min_val).is_integer() and float(max_val).is_integer():
                metadata["dtypes"][col] = "int"
            else:
                metadata["dtypes"][col] = "float"

    # Valores por defecto y tipos para categóricas
    for col in categorical_columns:
        if col not in metadata["default_values"]:
            opciones = categorical_options.get(col, [])
            metadata["default_values"][col] = opciones[0] if len(opciones) > 0 else ""

        if col not in metadata["dtypes"]:
            metadata["dtypes"][col] = "str"

    return metadata

# =========================
# CARGA DE ARTEFACTOS
# =========================
@st.cache_resource
def cargar_modelo_y_componentes():
    modelo = tf.keras.models.load_model("modelo_credit_score.keras")
    preprocesador = joblib.load("preprocesador_credit_score.jb")
    pca = joblib.load("pca_credit_score.jb")
    metadata = joblib.load("metadata_credit_score.jb")

    # Completar metadata si faltan campos
    metadata = completar_metadata(metadata)

    return modelo, preprocesador, pca, metadata

modelo, preprocesador, pca, metadata = cargar_modelo_y_componentes()

feature_order = metadata["feature_order"]
numeric_columns = metadata["numeric_columns"]
categorical_columns = metadata["categorical_columns"]
numeric_ranges = metadata["numeric_ranges"]
categorical_options = metadata["categorical_options"]
default_values = metadata["default_values"]
dtypes = metadata["dtypes"]

# =========================
# FUNCIONES AUXILIARES
# =========================
def formatear_prediccion(clase):
    etiquetas = {
        0: "Credit Score = 0",
        1: "Credit Score = 1",
        2: "Credit Score = 2"
    }
    return etiquetas.get(clase, f"Clase desconocida: {clase}")

def validar_rangos(datos_entrada):
    errores = []

    for col in numeric_columns:
        valor = datos_entrada[col]
        min_val = numeric_ranges[col]["min"]
        max_val = numeric_ranges[col]["max"]

        if valor < min_val or valor > max_val:
            errores.append(
                f"La variable '{col}' debe estar entre {min_val:.4f} y {max_val:.4f}."
            )

    for col in categorical_columns:
        if datos_entrada[col] not in categorical_options[col]:
            errores.append(
                f"La variable categórica '{col}' contiene un valor no visto en entrenamiento."
            )

    return errores

def construir_dataframe_entrada(valores_usuario):
    df_input = pd.DataFrame([valores_usuario])
    df_input = df_input[feature_order]
    return df_input

# =========================
# INTERFAZ
# =========================
st.title("💳 Predicción de Credit Score")
st.markdown(
    """
    Esta aplicación permite predecir la categoría de **Credit Score** (`0`, `1` o `2`)
    utilizando el modelo entrenado previamente con:
    - preprocesamiento de variables
    - codificación de categóricas
    - normalización
    - reducción de dimensionalidad con PCA
    - red neuronal en TensorFlow/Keras
    """
)

st.info(
    "Los valores permitidos en la interfaz fueron restringidos con base en los rangos y categorías observados durante el entrenamiento."
)

with st.form("formulario_credit_score"):
    st.subheader("Ingrese la información del cliente")

    valores_usuario = {}

    # -------------------------
    # NUMÉRICAS
    # -------------------------
    st.markdown("### Variables numéricas")
    col1, col2 = st.columns(2)

    mitad_num = len(numeric_columns) // 2 + len(numeric_columns) % 2
    numeric_left = numeric_columns[:mitad_num]
    numeric_right = numeric_columns[mitad_num:]

    with col1:
        for col in numeric_left:
            min_val = numeric_ranges[col]["min"]
            max_val = numeric_ranges[col]["max"]
            default_val = default_values[col]

            if dtypes[col] == "int":
                valores_usuario[col] = st.number_input(
                    label=col,
                    min_value=int(np.floor(min_val)),
                    max_value=int(np.ceil(max_val)),
                    value=int(round(default_val)),
                    step=1,
                    format="%d"
                )
            else:
                step_val = max((float(max_val) - float(min_val)) / 1000, 0.01)
                valores_usuario[col] = st.number_input(
                    label=col,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=float(step_val),
                    format="%.4f"
                )

    with col2:
        for col in numeric_right:
            min_val = numeric_ranges[col]["min"]
            max_val = numeric_ranges[col]["max"]
            default_val = default_values[col]

            if dtypes[col] == "int":
                valores_usuario[col] = st.number_input(
                    label=col,
                    min_value=int(np.floor(min_val)),
                    max_value=int(np.ceil(max_val)),
                    value=int(round(default_val)),
                    step=1,
                    format="%d"
                )
            else:
                step_val = max((float(max_val) - float(min_val)) / 1000, 0.01)
                valores_usuario[col] = st.number_input(
                    label=col,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=float(step_val),
                    format="%.4f"
                )

    # -------------------------
    # CATEGÓRICAS
    # -------------------------
    st.markdown("### Variables categóricas")
    col3, col4 = st.columns(2)

    mitad_cat = len(categorical_columns) // 2 + len(categorical_columns) % 2
    cat_left = categorical_columns[:mitad_cat]
    cat_right = categorical_columns[mitad_cat:]

    with col3:
        for col in cat_left:
            opciones = categorical_options[col]
            default_val = default_values[col]
            default_index = opciones.index(default_val) if default_val in opciones else 0

            valores_usuario[col] = st.selectbox(
                label=col,
                options=opciones,
                index=default_index
            )

    with col4:
        for col in cat_right:
            opciones = categorical_options[col]
            default_val = default_values[col]
            default_index = opciones.index(default_val) if default_val in opciones else 0

            valores_usuario[col] = st.selectbox(
                label=col,
                options=opciones,
                index=default_index
            )

    boton_predecir = st.form_submit_button("Predecir Credit Score")

# =========================
# PREDICCIÓN
# =========================
if boton_predecir:
    errores = validar_rangos(valores_usuario)

    if errores:
        for error in errores:
            st.error(error)
    else:
        try:
            df_input = construir_dataframe_entrada(valores_usuario)

            input_prep = preprocesador.transform(df_input)
            input_pca = pca.transform(input_prep)

            probabilidades = modelo.predict(input_pca, verbose=0)[0]
            clase_predicha = int(np.argmax(probabilidades))

            st.success("Predicción realizada correctamente.")

            st.subheader("Resultado de la predicción")
            st.write(f"**Clase predicha:** {clase_predicha}")
            st.write(f"**Etiqueta:** {formatear_prediccion(clase_predicha)}")

            st.subheader("Probabilidades por clase")
            df_prob = pd.DataFrame({
                "Clase": [0, 1, 2],
                "Probabilidad": probabilidades
            })
            st.dataframe(df_prob, use_container_width=True)
            st.bar_chart(df_prob.set_index("Clase"))

            with st.expander("Ver datos enviados al modelo"):
                st.dataframe(df_input, use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error al realizar la predicción: {str(e)}")