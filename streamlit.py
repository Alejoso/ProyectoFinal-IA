import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from rag.retriever import build_retriever, retrieve

# Categorical mappings from the German credit preprocessing pipeline
ORDINAL_CATEGORIES = {
    "estado_cuenta": [
        "saldo_negativo",
        "saldo_0_a_200",
        "saldo_mayor_200",
        "sin_cuenta_corriente",
    ],
    "ahorros": [
        "menos_100",
        "desconocido_sin_ahorros",
        "100_a_500",
        "500_a_1000",
        "mayor_1000",
    ],
    "empleo_actual": [
        "desempleado",
        "menos_1_ano",
        "1_a_4_anos",
        "4_a_7_anos",
        "mayor_7_anos",
    ],
    "trabajo": [
        "no_calificado_no_residente",
        "no_calificado_residente",
        "calificado",
        "altamente_calificado",
    ],
}

NOMINAL_CATEGORIES = {
    "historial_credito": sorted([
        "sin_creditos_o_pagados",
        "todos_pagados_aqui",
        "creditos_al_dia",
        "retraso_en_pasado",
        "cuenta_critica_otros_bancos",
    ]),
    "proposito": sorted([
        "carro_nuevo",
        "carro_usado",
        "muebles_equipos",
        "radio_television",
        "electrodomesticos",
        "reparaciones",
        "educacion",
        "vacaciones",
        "reentrenamiento",
        "negocio",
        "otros",
    ]),
    "estado_civil_sexo": sorted([
        "hombre_divorciado",
        "mujer_divorciada_casada",
        "hombre_soltero",
        "hombre_casado_viudo",
        "mujer_soltera",
    ]),
    "deudores_garantes": sorted([
        "ninguno",
        "codeudor",
        "garante",
    ]),
    "propiedad": sorted([
        "bienes_raices",
        "seguro_o_ahorro_construccion",
        "carro_u_otro",
        "sin_propiedad",
    ]),
    "otros_planes_pago": sorted([
        "banco",
        "tiendas",
        "ninguno",
    ]),
    "vivienda": sorted([
        "alquilada",
        "propia",
        "gratis",
    ]),
    "telefono": sorted([
        "sin_telefono",
        "con_telefono",
    ]),
    "trabajador_extranjero": sorted([
        "si",
        "no",
    ]),
}

FEATURE_NAMES = [
    "duracion_meses", "monto_credito", "tasa_pago_pct_ingreso",
    "residencia_actual_anos", "edad", "creditos_existentes", "personas_a_cargo",
    "estado_cuenta", "ahorros", "empleo_actual", "trabajo",
    "historial_credito_cuenta_critica_otros_bancos",
    "historial_credito_retraso_en_pasado",
    "historial_credito_sin_creditos_o_pagados",
    "historial_credito_todos_pagados_aqui",
    "proposito_carro_usado", "proposito_educacion", "proposito_electrodomesticos",
    "proposito_muebles_equipos", "proposito_negocio", "proposito_otros",
    "proposito_radio_television", "proposito_reentrenamiento",
    "proposito_reparaciones",
    "estado_civil_sexo_hombre_divorciado", "estado_civil_sexo_hombre_soltero",
    "estado_civil_sexo_mujer_divorciada_casada",
    "deudores_garantes_garante", "deudores_garantes_ninguno",
    "propiedad_carro_u_otro", "propiedad_seguro_o_ahorro_construccion",
    "propiedad_sin_propiedad", "otros_planes_pago_ninguno",
    "otros_planes_pago_tiendas", "vivienda_gratis", "vivienda_propia",
    "telefono_sin_telefono", "trabajador_extranjero_si",
]

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    layout="wide"
)

st.title("Credit Risk Prediction Dashboard")
st.markdown("""
Random Forest model trained on German Credit Data.  
Includes performance metrics, feature importance, and live prediction.
""")

# =========================
# LOAD MODEL + ARTIFACTS
# =========================
@st.cache_resource
def load_model():
    return joblib.load("data/processed/modelo_final.joblib")  # your trained pipeline

model = load_model()

@st.cache_data
def load_test_data():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_test, y_test

X_test, y_test = load_test_data()

@st.cache_data
def evaluate_model_pipeline(_model, X_test, y_test):
    y_pred = _model.predict(X_test)
    y_proba = _model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return cm, metrics

test_cm, test_metrics = evaluate_model_pipeline(model, X_test, y_test)

@st.cache_resource
def load_rag_retriever():
    return build_retriever("rag/corpus")

retriever = load_rag_retriever()

# Mapping helpers for the sampled German credit preprocessing logic
ESTADO_CUENTA_OPTIONS = [
    ("Saldo negativo (A11)", "saldo_negativo"),
    ("Saldo 0-200 (A12)", "saldo_0_a_200"),
    ("Saldo mayor 200 (A13)", "saldo_mayor_200"),
    ("Sin cuenta corriente (A14)", "sin_cuenta_corriente"),
]
HISTORIAL_CREDITO_OPTIONS = [
    ("Sin créditos o pagados (A30)", "sin_creditos_o_pagados"),
    ("Todos pagados aquí (A31)", "todos_pagados_aqui"),
    ("Créditos al día (A32)", "creditos_al_dia"),
    ("Retraso en el pasado (A33)", "retraso_en_pasado"),
    ("Cuenta crítica / otros bancos (A34)", "cuenta_critica_otros_bancos"),
]
PROPOSITO_OPTIONS = [
    ("Carro nuevo (A40)", "carro_nuevo"),
    ("Carro usado (A41)", "carro_usado"),
    ("Muebles / equipos (A42)", "muebles_equipos"),
    ("Radio / televisión (A43)", "radio_television"),
    ("Electrodomésticos (A44)", "electrodomesticos"),
    ("Reparaciones (A45)", "reparaciones"),
    ("Educación (A46)", "educacion"),
    ("Vacaciones (A47)", "vacaciones"),
    ("Reentrenamiento (A48)", "reentrenamiento"),
    ("Negocio (A49)", "negocio"),
    ("Otros (A410)", "otros"),
]
AHORROS_OPTIONS = [
    ("Menos 100 (A61)", "menos_100"),
    ("Desconocido / sin ahorros (A65)", "desconocido_sin_ahorros"),
    ("100 a 500 (A62)", "100_a_500"),
    ("500 a 1000 (A63)", "500_a_1000"),
    ("Mayor 1000 (A64)", "mayor_1000"),
]
EMPLEO_ACTUAL_OPTIONS = [
    ("Desempleado (A71)", "desempleado"),
    ("Menos 1 año (A72)", "menos_1_ano"),
    ("1 a 4 años (A73)", "1_a_4_anos"),
    ("4 a 7 años (A74)", "4_a_7_anos"),
    ("Mayor 7 años (A75)", "mayor_7_anos"),
]
ESTADO_CIVIL_SEXO_OPTIONS = [
    ("Hombre divorciado (A91)", "hombre_divorciado"),
    ("Mujer divorciada/casada (A92)", "mujer_divorciada_casada"),
    ("Hombre soltero (A93)", "hombre_soltero"),
    ("Hombre casado/viudo (A94)", "hombre_casado_viudo"),
    ("Mujer soltera (A95)", "mujer_soltera"),
]
DEUDORES_GARANTES_OPTIONS = [
    ("Ninguno (A101)", "ninguno"),
    ("Codeudor (A102)", "codeudor"),
    ("Garante (A103)", "garante"),
]
PROPIEDAD_OPTIONS = [
    ("Bienes raíces (A121)", "bienes_raices"),
    ("Seguro / ahorro / construcción (A122)", "seguro_o_ahorro_construccion"),
    ("Carro u otro (A123)", "carro_u_otro"),
    ("Sin propiedad (A124)", "sin_propiedad"),
]
OTROS_PLANES_PAGO_OPTIONS = [
    ("Banco (A141)", "banco"),
    ("Tiendas (A142)", "tiendas"),
    ("Ninguno (A143)", "ninguno"),
]
VIVIENDA_OPTIONS = [
    ("Alquilada (A151)", "alquilada"),
    ("Propia (A152)", "propia"),
    ("Gratis (A153)", "gratis"),
]
TRABAJO_OPTIONS = [
    ("No calificado / no residente (A171)", "no_calificado_no_residente"),
    ("No calificado / residente (A172)", "no_calificado_residente"),
    ("Calificado (A173)", "calificado"),
    ("Altamente calificado (A174)", "altamente_calificado"),
]
TELEFONO_OPTIONS = [
    ("Sin teléfono (A191)", "sin_telefono"),
    ("Con teléfono (A192)", "con_telefono"),
]
TRABAJADOR_EXTRANJERO_OPTIONS = [
    ("Sí (A201)", "si"),
    ("No (A202)", "no"),
]


def _get_option_value(options, label):
    return dict(options)[label]


def _encode_ordinal(feature, value):
    return ORDINAL_CATEGORIES[feature].index(value)


def _set_one_hot(data, feature, value):
    categories = NOMINAL_CATEGORIES[feature]
    column_name = f"{feature}_{value}"
    if value != categories[0] and column_name in data:
        data[column_name] = 1

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Input Applicant Data")

def user_input():
    estado_cuenta = _get_option_value(
        ESTADO_CUENTA_OPTIONS,
        st.sidebar.selectbox(
            "Account Status",
            [label for label, _ in ESTADO_CUENTA_OPTIONS],
        ),
    )
    historial_credito = _get_option_value(
        HISTORIAL_CREDITO_OPTIONS,
        st.sidebar.selectbox(
            "Credit History",
            [label for label, _ in HISTORIAL_CREDITO_OPTIONS],
        ),
    )
    proposito = _get_option_value(
        PROPOSITO_OPTIONS,
        st.sidebar.selectbox(
            "Purpose",
            [label for label, _ in PROPOSITO_OPTIONS],
        ),
    )
    ahorros = _get_option_value(
        AHORROS_OPTIONS,
        st.sidebar.selectbox(
            "Savings",
            [label for label, _ in AHORROS_OPTIONS],
        ),
    )
    empleo_actual = _get_option_value(
        EMPLEO_ACTUAL_OPTIONS,
        st.sidebar.selectbox(
            "Current Employment",
            [label for label, _ in EMPLEO_ACTUAL_OPTIONS],
        ),
    )
    estado_civil_sexo = _get_option_value(
        ESTADO_CIVIL_SEXO_OPTIONS,
        st.sidebar.selectbox(
            "Marital Status / Sex",
            [label for label, _ in ESTADO_CIVIL_SEXO_OPTIONS],
        ),
    )
    deudores_garantes = _get_option_value(
        DEUDORES_GARANTES_OPTIONS,
        st.sidebar.selectbox(
            "Co-debtors / Guarantors",
            [label for label, _ in DEUDORES_GARANTES_OPTIONS],
        ),
    )
    propiedad = _get_option_value(
        PROPIEDAD_OPTIONS,
        st.sidebar.selectbox(
            "Property",
            [label for label, _ in PROPIEDAD_OPTIONS],
        ),
    )
    otros_planes_pago = _get_option_value(
        OTROS_PLANES_PAGO_OPTIONS,
        st.sidebar.selectbox(
            "Other Payment Plans",
            [label for label, _ in OTROS_PLANES_PAGO_OPTIONS],
        ),
    )
    vivienda = _get_option_value(
        VIVIENDA_OPTIONS,
        st.sidebar.selectbox(
            "Housing",
            [label for label, _ in VIVIENDA_OPTIONS],
        ),
    )
    telefono = _get_option_value(
        TELEFONO_OPTIONS,
        st.sidebar.selectbox(
            "Phone",
            [label for label, _ in TELEFONO_OPTIONS],
        ),
    )
    trabajador_extranjero = _get_option_value(
        TRABAJADOR_EXTRANJERO_OPTIONS,
        st.sidebar.selectbox(
            "Foreign Worker",
            [label for label, _ in TRABAJADOR_EXTRANJERO_OPTIONS],
        ),
    )
    trabajo = _get_option_value(
        TRABAJO_OPTIONS,
        st.sidebar.selectbox(
            "Job Qualification",
            [label for label, _ in TRABAJO_OPTIONS],
        ),
    )

    data = {name: 0 for name in FEATURE_NAMES}
    data.update({
        "duracion_meses": st.sidebar.slider("Duration (months)", 4, 72, 24),
        "monto_credito": st.sidebar.number_input("Credit Amount", 250, 20000, 3000),
        "tasa_pago_pct_ingreso": st.sidebar.slider("Payment % of Income", 1, 50, 20),
        "residencia_actual_anos": st.sidebar.slider("Residence (years)", 0, 40, 5),
        "edad": st.sidebar.slider("Age", 18, 75, 30),
        "creditos_existentes": st.sidebar.slider("Existing credits", 0, 6, 1),
        "personas_a_cargo": st.sidebar.slider("People liable", 0, 3, 0),
    })

    data["estado_cuenta"] = _encode_ordinal("estado_cuenta", estado_cuenta)
    data["ahorros"] = _encode_ordinal("ahorros", ahorros)
    data["empleo_actual"] = _encode_ordinal("empleo_actual", empleo_actual)
    data["trabajo"] = _encode_ordinal("trabajo", trabajo)

    _set_one_hot(data, "historial_credito", historial_credito)
    _set_one_hot(data, "proposito", proposito)
    _set_one_hot(data, "estado_civil_sexo", estado_civil_sexo)
    _set_one_hot(data, "deudores_garantes", deudores_garantes)
    _set_one_hot(data, "propiedad", propiedad)
    _set_one_hot(data, "otros_planes_pago", otros_planes_pago)
    _set_one_hot(data, "vivienda", vivienda)
    _set_one_hot(data, "telefono", telefono)
    _set_one_hot(data, "trabajador_extranjero", trabajador_extranjero)

    return pd.DataFrame([data], columns=FEATURE_NAMES)

input_df = user_input()

# =========================
# PREDICTION
# =========================
st.subheader("Prediction")

prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

col1, col2 = st.columns(2)

with col1:
    st.metric("Prediction", "Bad Credit ❌" if prediction == 1 else "Good Credit ✅")

with col2:
    st.metric("Default Probability", f"{prob:.2%}")

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Test Metrics (actual)**  
    - ROC-AUC: {test_metrics['roc_auc']:.4f}  
    - Accuracy: {test_metrics['accuracy']:.4f}  
    - F1-score: {test_metrics['f1']:.4f}  
    - Recall: {test_metrics['recall']:.4f}  
    - Precision: {test_metrics['precision']:.4f}  
    """)

with col2:
    st.info("""
    ⚠️ Results are computed on the held-out test set loaded from
    `data/processed/X_test.csv` and `y_test.csv`.  
    This is the true confusion matrix for the pipeline.
    """)

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=["Good", "Bad"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Test set confusion matrix")
st.pyplot(fig)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Feature Importance")

features = [
    "estado_cuenta", "monto_credito", "duracion_meses",
    "edad", "ahorros", "empleo_actual",
    "tasa_pago_pct_ingreso"
]

importance = [0.1526, 0.1181, 0.0875, 0.0814, 0.0587, 0.0517, 0.0354]

df_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(df_imp.set_index("Feature"))

# =========================
# RAG DEMO (OPTIONAL)
# =========================
st.subheader("Regulatory Q&A (RAG)")

query = st.text_input("Ask about credit risk regulation:")

if query and st.button("Ask"):
    results = retrieve(retriever, query, top_k=3)
    if not results:
        st.info("No relevant document snippets found for that query.")
    else:
        st.markdown("**Top RAG results from the local corpus:**")
        for result in results:
            with st.expander(f"{result['title']} — score {result['score']:.2f}"):
                st.write(result["text"][:800] + ("..." if len(result["text"]) > 800 else ""))