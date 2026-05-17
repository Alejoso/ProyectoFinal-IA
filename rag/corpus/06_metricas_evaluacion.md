# Métricas de evaluación para credit scoring

Las métricas de evaluación de modelos de credit scoring deben elegirse según el contexto de negocio y la presencia de desbalance entre clases. Las más utilizadas son:

**Accuracy** mide el porcentaje total de predicciones correctas. Es engañosa cuando hay desbalance: un modelo que siempre predice la clase mayoritaria obtiene alta accuracy sin ser útil. No debe usarse como métrica principal en credit scoring.

**Precision** (precisión) es la proporción de positivos predichos que efectivamente son positivos. En credit scoring: de los clientes que el modelo clasificó como malos pagadores, qué porcentaje realmente lo era. Alta precision significa pocos falsos positivos, es decir, pocos buenos pagadores rechazados injustamente.

**Recall** (sensibilidad o tasa de verdaderos positivos) es la proporción de positivos reales que el modelo detectó correctamente. En credit scoring: de los malos pagadores que existen, qué porcentaje el modelo identificó. Alto recall significa pocos falsos negativos, es decir, pocos morosos que se cuelan en la cartera aprobada.

**F1-score** es la media armónica entre precision y recall. Sirve cuando se quiere equilibrar ambos sin priorizar uno.

**ROC-AUC** (Area Under the Receiver Operating Characteristic curve) mide la capacidad del modelo para distinguir entre clases independientemente del umbral elegido. Un AUC de 0.5 equivale a azar, 1.0 es discriminación perfecta. En credit scoring se considera un buen modelo aquel con AUC entre 0.70 y 0.85. Es la métrica estándar de la industria.

**Costo asimétrico de errores:** en credit scoring, los dos tipos de errores tienen costos muy diferentes. Aprobar a un mal pagador (falso negativo) implica perder potencialmente todo el monto del préstamo. Rechazar a un buen pagador (falso positivo) implica perder el ingreso por intereses de ese cliente. Estudios clásicos en banca estiman que el costo de un falso negativo es entre 3 y 5 veces el costo de un falso positivo. Esto justifica priorizar recall sobre precision en la mayoría de modelos de credit scoring.
