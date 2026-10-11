# GTA VI — Preorder Intent Predictor

Aplicación web interactiva construida con **Streamlit** que utiliza un modelo de Machine Learning (Pipeline de `scikit-learn`) para predecir la **intención de pre-orden** (`preorder_intent`) del videojuego *Grand Theft Auto VI* a partir de datos de expectativas de los fans.

El modelo devuelve una clase (`Yes`, `No`, `Maybe`) junto con la probabilidad asociada a cada clase, permitiendo anticipar conversión, dirigir campañas segmentadas y prever la demanda.

---

## Estructura del proyecto

```
Pagina_Web/
├── app_streamlit.py         # Aplicación web (interfaz + inferencia)
├── modelo_final_gta6.pkl    # Pipeline entrenado y serializado (joblib)
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Este archivo
```

> El archivo `modelo_final_gta6.pkl` debe permanecer en el **mismo directorio** que `app_streamlit.py`.

---

## Requisitos

- **Python 3.9 o superior**
- pip actualizado

Dependencias (contenidas en `requirements.txt`):

| Librería | Uso |
|---|---|
| `streamlit` | Interfaz web interactiva |
| `pandas` | Construcción del `DataFrame` de entrada |
| `scikit-learn` | Pipeline de ML (preprocesamiento + clasificador) |
| `joblib` | Carga del modelo serializado |

---

## Instalación

1. **Crear un entorno virtual (recomendado)**

   ```bash
   python -m venv venv
   ```

2. **Activar el entorno virtual**

   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
   - Windows (CMD): `venv\Scripts\activate.bat`
   - macOS / Linux: `source venv/bin/activate`

3. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución

Desde la carpeta `Pagina_Web`, ejecutar:

```bash
streamlit run app_streamlit.py
```

Streamlit levantará un servidor local y abrirá el navegador en una URL similar a:

```
http://localhost:8501
```

Para detener la aplicación, presionar `Ctrl + C` en la terminal.

---

## Uso

1. Completar el formulario guiado, organizado en tres bloques:
   - **Perfil Demográfico** (edad, género, región, disposición a pagar, redes de Rockstar)
   - **Plataforma e Historial** (plataforma, GTA previo favorito, años siguiendo la saga, fuente de información, vistas del tráiler)
   - **Expectativas y Sentimiento** (característica más esperada, *hype*, confianza, horas esperadas, retraso, precio, sentimiento general)
2. Presionar **"Predecir Intención de Compra"**.
3. La aplicación muestra:
   - La clase predicha (`Sí`, `No`, `Tal vez`) con realimentación visual.
   - El porcentaje de confianza de la predicción.
   - La distribución de probabilidades por clase.
   - El vector de características ingresado (panel expandible).

---

## Funcionamiento interno

1. El modelo (`modelo_final_gta6.pkl`) se carga **una sola vez** en memoria con `joblib.load()`, usando `@st.cache_resource` para evitar recargas.
2. Al enviar el formulario, las variables se empaquetan en un **`pd.DataFrame` de una sola fila** con el mismo esquema del entrenamiento.
3. Se invocan `pipeline.predict()` y `pipeline.predict_proba()` directamente sobre el `Pipeline`, que aplica internamente la codificación **One-Hot** y el **escalado** de forma sincronizada con el entrenamiento.

---

## Limitaciones conocidas

- **Ejecución solo local:** no es accesible públicamente por internet sin un despliegue en la nube.
- **Rigidez del esquema de entrada:** requiere exactamente las mismas variables, nombres y categorías usadas en el entrenamiento.
- **Sesgo hacia la clase mayoritaria ("Sí"):** el dataset original está desbalanceado.
- **Datos auto-reportados:** las variables provienen de encuestas de expectativas, no de comportamiento real.
