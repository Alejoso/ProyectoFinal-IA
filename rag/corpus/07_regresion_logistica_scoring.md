# Regresión logística aplicada a credit scoring

La regresión logística es el algoritmo más utilizado históricamente en credit scoring por varias razones que la hacen especialmente apropiada para este dominio:

**Interpretabilidad regulatoria.** Los reguladores financieros (Superintendencia Financiera en Colombia, BIS a nivel internacional) exigen que las decisiones automatizadas de crédito sean explicables. La regresión logística produce coeficientes interpretables: un coeficiente positivo para una variable significa que aumenta la probabilidad de incumplimiento, y la magnitud cuantifica el efecto. Esto permite explicar a un cliente por qué fue rechazado.

**Probabilidades calibradas.** A diferencia de algunos modelos no lineales, las probabilidades que produce la regresión logística suelen estar bien calibradas, es decir, una predicción de 0.30 efectivamente corresponde a aproximadamente un 30% de incumplimiento observado. Esto es crítico para cálculos de pérdida esperada (PD × LGD × EAD).

**Robustez con datasets pequeños.** Con datasets de pocos miles de registros, la regresión logística suele rendir competitivamente frente a modelos más complejos, sin sobreajuste excesivo.

**Limitaciones.** La regresión logística asume una relación lineal entre las variables y el logit de la probabilidad. No captura automáticamente interacciones ni relaciones no lineales, que sí capturan Random Forest o Gradient Boosting. Por esta razón, los modelos modernos de credit scoring suelen combinar enfoques: regresión logística como base interpretable, y modelos basados en árboles para validar y mejorar.

**Tratamiento del desbalance.** En sklearn, el parámetro `class_weight="balanced"` ajusta los pesos en la función de pérdida inversamente proporcional a las frecuencias de clase. En datasets como el German Credit, esto típicamente sube el recall significativamente a costa de la accuracy general, lo cual es deseable en credit scoring.
