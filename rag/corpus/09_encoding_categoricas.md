# Encoding de variables categóricas en modelos de credit scoring

Los algoritmos de machine learning requieren entradas numéricas. Las variables categóricas como `proposito` del crédito o `estado_cuenta` deben convertirse a representaciones numéricas antes del entrenamiento. La elección de la técnica de encoding tiene impacto directo en el rendimiento del modelo.

**One-Hot Encoding** crea una columna binaria por cada categoría. Si una variable tiene k categorías, genera k columnas (o k-1 si se omite una para evitar colinealidad perfecta). Es la opción correcta para variables nominales, donde no existe un orden natural entre categorías. Por ejemplo, `proposito` (carro nuevo, educación, vacaciones, negocio) es nominal: no se puede decir que "educación" esté entre "carro nuevo" y "vacaciones".

**Ordinal Encoding** asigna un entero a cada categoría según un orden predefinido. Apropiado para variables con orden natural, como `ahorros` (menos de 100, de 100 a 500, de 500 a 1000, más de 1000) o `empleo_actual` (desempleado, menos de 1 año, 1 a 4 años, 4 a 7 años, más de 7 años). El ordinal encoding aprovecha esta estructura: el modelo entiende que "más ahorros" es una dirección.

**Trampa común: variables que parecen ordinales pero no lo son.** En el dataset German Credit, la variable `historial_credito` parece ordinal (sin créditos < pagados < retrasos < críticos), pero el análisis exploratorio muestra un patrón no monótono: "sin créditos previos" tiene tasa de incumplimiento mayor que "créditos críticos en otros bancos". Esto rompe el supuesto de orden y justifica usar one-hot encoding para esta variable.

**Otras técnicas:**

*Target Encoding* reemplaza cada categoría por la tasa de incumplimiento observada en el train. Captura más información pero introduce riesgo de data leakage si no se hace con validación cruzada.

*Embeddings de categorías* aprendidos en redes neuronales. Útil con muchas categorías de alta cardinalidad, no relevante para credit scoring tradicional.

**Recomendación:** clasificar manualmente cada variable categórica como ordinal (con orden defendible) o nominal, y aplicar el encoding correspondiente. Esta decisión debe estar documentada y justificada para superar auditorías regulatorias.
