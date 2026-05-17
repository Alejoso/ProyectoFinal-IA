# Desbalance de clases en credit scoring

En la mayoría de problemas de credit scoring, la clase de interés (mal pagador) es minoritaria. En el dataset German Credit la proporción es 70% buenos pagadores frente a 30% malos pagadores, lo que se considera un desbalance moderado. En carteras reales de bancos, la proporción de morosos suele ser aún menor, entre 2% y 10%, lo que se denomina desbalance severo.

El desbalance plantea dos problemas principales para los modelos de clasificación:

**Problema 1: la métrica accuracy engaña.** Un modelo trivial que siempre prediga la clase mayoritaria (todos son buenos pagadores) obtiene una accuracy igual a la proporción de la clase mayoritaria. En el German Credit esto significa un 70% de aciertos sin haber aprendido nada útil. Por eso accuracy nunca debe ser la métrica principal en problemas desbalanceados.

**Problema 2: los modelos tienden a ignorar la clase minoritaria.** Los algoritmos optimizan por defecto la pérdida agregada, lo que favorece predecir bien la clase mayoritaria a costa de la minoritaria.

Existen varias estrategias para tratar el desbalance:

**Ponderación de clases (class weighting):** se asigna mayor peso a los errores de la clase minoritaria en la función de pérdida. En sklearn esto se hace con el parámetro `class_weight="balanced"`, que ajusta los pesos inversamente proporcionales a la frecuencia de cada clase.

**Sobremuestreo (oversampling):** se generan ejemplos sintéticos de la clase minoritaria. La técnica más popular es SMOTE (Synthetic Minority Over-sampling Technique), que crea nuevos puntos interpolando entre vecinos cercanos de la clase minoritaria.

**Submuestreo (undersampling):** se reduce la cantidad de ejemplos de la clase mayoritaria. Funciona bien con datasets grandes pero pierde información.

**Ajuste del umbral de decisión:** en lugar de modificar los datos o pesos, se cambia el umbral de probabilidad a partir del cual se clasifica un caso como positivo (por ejemplo, bajar de 0.5 a 0.3 para detectar más positivos).

La elección entre estas estrategias depende del tamaño del dataset, la severidad del desbalance y los objetivos de negocio.
