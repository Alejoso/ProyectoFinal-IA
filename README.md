# Proyecto Final - IA - EAFIT 2026-1

Credit Scoring sobre el dataset **German Credit Data** (UCI) usando Random Forest + un sistema RAG sobre normativa de riesgo crediticio (Basilea II y SARC Colombia).

## Requisitos

- Python 3.10 o superior
- ~2 GB de espacio libre (modelos de embeddings + ChromaDB + corpus)
- Cuenta gratuita en [Groq](https://console.groq.com) para usar el LLM del RAG

## Estructura del proyecto

```
proyecto-final-ia/
├── data/
│   ├── raw/                   
│   └── processed/             
├── notebooks/
│   ├── 01_eda.ipynb            
│   ├── 02_preprocessing.ipynb  
│   ├── 03_ml_classifiers.ipynb
│   └── 04_rag.ipynb            
├── rag/
│   ├── corpus/                
│   └── chroma_db/              
├── .env                        
├── requirements.txt
└── README.md
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/[USUARIO]/proyecto-ia-eafit.git
cd proyecto-ia-eafit
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar la API key de Groq

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```
GROQ_API_KEY=gsk_tu_key_aqui
```

La key se obtiene gratuitamente en [console.groq.com](https://console.groq.com) → API Keys → Create API Key.

### 4. Registrar el kernel de Jupyter (opcional, recomendado)

```bash
python -m ipykernel install --user --name=proyecto-ia --display-name "Python (proyecto-ia)"
```

## Cómo ejecutar los notebooks

Los 4 notebooks deben correrse **en orden**, ya que cada uno depende de los archivos generados por el anterior.

```bash
jupyter notebook
```

Luego en la interfaz de Jupyter, abrir cada notebook y ejecutar todas las celdas (`Cell → Run All`):

| Notebook | Qué hace | Tiempo aprox. |
|----------|----------|---------------|
| `01_eda.ipynb` | Carga el dataset, analiza distribuciones, correlaciones y tests chi-cuadrado | ~1 min |
| `02_preprocessing.ipynb` | Aplica encoding (ordinal + one-hot), escalado, split estratificado 80/20 | ~30 seg |
| `03_ml_classifiers.ipynb` | Entrena 4 modelos, hace CV 5-fold, tunea Random Forest con GridSearch | ~2-3 min |
| `04_rag.ipynb` | Construye el vector store con ChromaDB y prueba el RAG con Groq | ~3-5 min (primera vez) |

**Nota:** la primera ejecución del notebook 04 descarga el modelo de embeddings (~120 MB) y construye el vector store. Las siguientes ejecuciones son mucho más rápidas porque el vector store queda persistido en `rag/chroma_db/`.

## Tecnologías utilizadas

- **scikit-learn 1.5.2** — Modelos ML (Random Forest, Logistic Regression, Gradient Boosting)
- **pandas, numpy** — Manipulación de datos
- **matplotlib, seaborn** — Visualización
- **sentence-transformers** — Embeddings multilingües (paraphrase-multilingual-MiniLM-L12-v2)
- **ChromaDB** — Vector store local
- **LangChain** — Orquestación del pipeline RAG
- **Groq** — Inferencia del LLM (llama-3.3-70b-versatile)

