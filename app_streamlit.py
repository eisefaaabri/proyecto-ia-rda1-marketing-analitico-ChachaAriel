import os
import sys
import warnings
import joblib
import pandas as pd
import streamlit as st

# Supresión de avisos de advertencia de versión en la interfaz
warnings.filterwarnings("ignore")

# Parche de compatibilidad hacia atrás para _RemainderColsList en sklearn >= 1.7+
import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

# Configuración inicial de la página
st.set_page_config(
    page_title="GTA VI | Predictor de Intención de Compra",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para interfaz moderna temática GTA VI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff007f, #7928ca, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff007f, #7928ca);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #e60073, #6a22b3);
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Carga en memoria en caché del modelo entrenado
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), "modelo_final_gta6.pkl")
    if not os.path.exists(model_path):
        model_path = "modelo_final_gta6.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = get_model()

# Encabezado principal
st.markdown('<div class="main-header">GTA VI — Preorder Intent Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistema de Inteligencia Artificial para estimar la probabilidad de reserva anticipada (Preorder Intent)</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ No se encontró el archivo 'modelo_final_gta6.pkl'. Coloca el archivo del modelo en el mismo directorio que este script.")

# Formulario estructurado para captura de entradas del usuario
with st.form("form_prediccion"):
    st.subheader("📋 Parámetros del Usuario y Perfil Gamer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👤 Perfil Demográfico")
        age = st.number_input("Edad (age)", min_value=12, max_value=90, value=24, step=1)
        gender = st.selectbox("Género (gender)", options=["Male", "Female", "Non-binary", "Prefer not to say"])
        region = st.selectbox(
            "Región (region)",
            options=["North America", "Europe", "Latin America", "Asia-Pacific", "Middle East", "Africa"]
        )
        follows_rockstar_socials = st.selectbox("Sigue redes de Rockstar (follows_rockstar_socials)", options=["Yes", "No"])
        willingness_to_pay_usd = st.number_input("Disposición a pagar en USD (willingness_to_pay_usd)", min_value=0.0, max_value=300.0, value=70.0, step=5.0)

    with col2:
        st.markdown("##### 🎮 Plataforma e Historial")
        primary_platform = st.selectbox(
            "Plataforma principal (primary_platform)",
            options=["PlayStation 5", "Xbox Series X/S", "PC", "PlayStation 5 Pro", "Undecided"]
        )
        favorite_previous_gta = st.selectbox(
            "GTA previo favorito (favorite_previous_gta)",
            options=[
                "GTA V",
                "GTA San Andreas",
                "GTA Vice City",
                "GTA IV",
                "GTA III",
                "Red Dead Redemption 2",
                "Never played GTA"
            ]
        )
        years_following_gta6 = st.number_input("Años siguiendo GTA 6 (years_following_gta6)", min_value=0, max_value=15, value=4, step=1)
        primary_info_source = st.selectbox(
            "Fuente principal de info (primary_info_source)",
            options=[
                "YouTube",
                "X/Twitter",
                "Reddit",
                "TikTok",
                "Official Rockstar trailer",
                "Gaming news sites",
                "Friends",
                "Twitch"
            ]
        )
        trailer1_times_watched = st.number_input("Vistas del Trailer 1 (trailer1_times_watched)", min_value=0, max_value=200, value=3, step=1)

    with col3:
        st.markdown("##### 🔥 Expectativas y Sentimiento")
        most_anticipated_feature = st.selectbox(
            "Característica más esperada (most_anticipated_feature)",
            options=[
                "Story & narrative depth",
                "Dual protagonists (Jason & Lucia)",
                "Graphics & ray tracing",
                "Map size & exploration",
                "Open-world Vice City map",
                "Better driving physics",
                "Realistic economy & businesses",
                "Improved police/AI systems",
                "Online/multiplayer mode",
                "Deeper character customization"
            ]
        )
        hype_score_1to10 = st.slider("Nivel de Hype (1 a 10) (hype_score_1to10)", min_value=1, max_value=10, value=9)
        confidence_rockstar_delivers_1to5 = st.slider("Confianza en Rockstar (1 a 5) (confidence_rockstar_delivers_1to5)", min_value=1, max_value=5, value=4)
        expected_firstweek_hours = st.number_input("Horas esperadas 1ra semana (expected_firstweek_hours)", min_value=0, max_value=120, value=25, step=1)
        expects_release_delay = st.selectbox("¿Espera retraso de salida? (expects_release_delay)", options=["No", "Yes"])
        worried_about_price = st.selectbox("¿Preocupado por el precio? (worried_about_price)", options=["No", "Yes"])
        hype_change_vs_6mo_ago = st.selectbox("Cambio de hype vs hace 6 meses (hype_change_vs_6mo_ago)", options=["Increased", "Same", "Decreased"])
        overall_sentiment = st.selectbox("Sentimiento general (overall_sentiment)", options=["Positive", "Very Positive", "Neutral/Cautious"])

    st.markdown("---")
    submitted = st.form_submit_button("🔮 Predecir Intención de Compra")

# Inferencia del modelo
if submitted:
    if pipeline is None:
        st.error("No se puede realizar la predicción porque el modelo no está disponible.")
    else:
        fila = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "region": region,
            "primary_platform": primary_platform,
            "favorite_previous_gta": favorite_previous_gta,
            "years_following_gta6": years_following_gta6,
            "primary_info_source": primary_info_source,
            "trailer1_times_watched": trailer1_times_watched,
            "most_anticipated_feature": most_anticipated_feature,
            "hype_score_1to10": hype_score_1to10,
            "confidence_rockstar_delivers_1to5": confidence_rockstar_delivers_1to5,
            "willingness_to_pay_usd": float(willingness_to_pay_usd),
            "expected_firstweek_hours": expected_firstweek_hours,
            "expects_release_delay": expects_release_delay,
            "worried_about_price": worried_about_price,
            "follows_rockstar_socials": follows_rockstar_socials,
            "hype_change_vs_6mo_ago": hype_change_vs_6mo_ago,
            "overall_sentiment": overall_sentiment
        }])

        try:
            prediction = pipeline.predict(fila)[0]

            has_proba = hasattr(pipeline, "predict_proba")
            prob_dict = {}
            confianza = 0.0

            if has_proba:
                probas = pipeline.predict_proba(fila)[0]
                classes = list(pipeline.classes_)
                prob_dict = {str(c): float(p) for c, p in zip(classes, probas)}
                if prediction in prob_dict:
                    confianza = prob_dict[prediction] * 100
                elif str(prediction) in prob_dict:
                    confianza = prob_dict[str(prediction)] * 100
                else:
                    confianza = max(probas) * 100

            st.markdown("### 🎯 Resultado de la Predicción")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                pred_label = str(prediction).strip()
                if pred_label.lower() == "yes":
                    st.success(f"**Intención de Compra (Preorder Intent): SÍ ({pred_label})**")
                    st.write("El perfil evaluado presenta una alta inclinación a realizar la reserva anticipada de Grand Theft Auto VI.")
                elif pred_label.lower() == "maybe":
                    st.warning(f"**Intención de Compra (Preorder Intent): TAL VEZ ({pred_label})**")
                    st.write("El perfil se encuentra indeciso o condicionado por el precio, fecha final de lanzamiento o críticas iniciales.")
                else:
                    st.error(f"**Intención de Compra (Preorder Intent): NO ({pred_label})**")
                    st.write("El perfil no tiene intenciones de reservar anticipadamente el título.")

                if has_proba:
                    st.metric(label="Confianza / Probabilidad", value=f"{confianza:.1f}%")

            with res_col2:
                if has_proba and prob_dict:
                    st.write("**Distribución de Probabilidades por Clase:**")
                    prob_df = pd.DataFrame({
                        "Clase": list(prob_dict.keys()),
                        "Probabilidad (%)": [round(val * 100, 2) for val in prob_dict.values()]
                    })
                    st.dataframe(prob_df, hide_index=True, use_container_width=True)
                    st.progress(min(max(confianza / 100.0, 0.0), 1.0))

            with st.expander("Ver vector de características ingresado (DataFrame)"):
                st.dataframe(fila, use_container_width=True)

        except Exception as e:
            st.error(f"Error durante el proceso de inferencia del modelo: {str(e)}")
