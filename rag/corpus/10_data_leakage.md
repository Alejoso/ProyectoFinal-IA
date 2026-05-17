# Data leakage en pipelines de machine learning

El data leakage (fuga de información) ocurre cuando información del conjunto de prueba se filtra al entrenamiento, inflando artificialmente las métricas de evaluación. En modelos de credit scoring esto produce sobreestimación del rendimiento real, lo que puede llevar a desastres financieros cuando el modelo se despliega en producción.

**Tipos comunes de data leakage:**

**Fuga por preprocesamiento.** Si se calculan estadísticas (media, desviación, categorías observadas) sobre todo el dataset antes de hacer split en train/test, esa información se contamina. Por ejemplo, si se aplica `StandardScaler` antes del split, la media y desviación usadas para escalar incluyen información del test. La solución es: hacer split primero, luego ajustar (`fit`) los transformadores **solo con el train**, y aplicar (`transform`) al test con los parámetros aprendidos del train.

**Fuga por features futuras.** Usar variables que en el momento de la decisión real no estarían disponibles. Por ejemplo, incluir "número de pagos realizados" para predecir incumplimiento es trampa: esa información solo existe después de haber otorgado el crédito.

**Fuga por estratificación de variables relacionadas con el target.** Si se balancean los datos antes del split, ambos conjuntos comparten información estadística del target.

**Fuga por target encoding mal hecho.** Si se calcula el promedio del target por categoría sobre todo el dataset, esa estadística contamina ambos conjuntos. La forma correcta es calcular el target encoding usando validación cruzada o solo con el conjunto de entrenamiento.

**Buenas prácticas para evitar leakage:**

1. Separar train y test al inicio del pipeline, antes de cualquier exploración profunda o transformación.
2. Encapsular el preprocesamiento en un `Pipeline` o `ColumnTransformer` de sklearn que se ajuste con `fit` solo al train.
3. En validación cruzada, ajustar el preprocesamiento dentro de cada fold, no antes.
4. Documentar qué variables están disponibles en el momento real de la decisión.

Un modelo con métricas demasiado buenas (por ejemplo, AUC > 0.95 en credit scoring) suele ser síntoma de data leakage. Los AUC típicos para datasets como German Credit están entre 0.75 y 0.82.
