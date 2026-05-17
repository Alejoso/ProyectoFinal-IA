# Random Forest y modelos de ensamble en credit scoring

Random Forest es un algoritmo de ensamble que combina múltiples árboles de decisión entrenados sobre muestras bootstrap del dataset, con selección aleatoria de variables en cada split. Es uno de los algoritmos más utilizados en credit scoring moderno por las siguientes características:

**Captura de relaciones no lineales.** A diferencia de la regresión logística, Random Forest puede modelar interacciones complejas entre variables sin necesidad de especificarlas manualmente. Por ejemplo, puede aprender que el efecto de la duración del crédito sobre el riesgo depende del monto solicitado.

**Robustez ante outliers y variables sin escalar.** Los árboles particionan el espacio por umbrales, no se ven afectados por valores extremos ni requieren normalización de las variables.

**Feature importance natural.** Random Forest reporta la importancia de cada variable, lo que permite identificar qué factores son más predictivos. En el dataset German Credit, las variables típicamente más importantes son `estado_cuenta`, `duracion_meses`, `monto_credito`, `historial_credito` y `edad`.

**Probabilidades menos calibradas.** Las probabilidades que produce Random Forest tienden a estar menos calibradas que las de la regresión logística (suelen estar comprimidas hacia el centro). Para usar estas probabilidades en cálculos de pérdida esperada, suele aplicarse una calibración posterior (Platt scaling o calibración isotónica).

**Hiperparámetros clave:**
- `n_estimators`: número de árboles. Más árboles = más estable, pero también más lento. Típicamente 100-500.
- `max_depth`: profundidad máxima. Limita el sobreajuste.
- `min_samples_split` y `min_samples_leaf`: controlan cuándo un árbol puede seguir dividiéndose. Valores mayores reducen sobreajuste.
- `class_weight`: en datasets desbalanceados, "balanced" ajusta automáticamente.

**Comparación con Gradient Boosting.** Modelos como XGBoost, LightGBM y CatBoost suelen rendir ligeramente mejor que Random Forest en credit scoring, pero a costa de mayor complejidad de tuning y menor robustez si los hiperparámetros no están bien ajustados. Para proyectos académicos y comparaciones limpias, Random Forest es un excelente punto medio entre rendimiento e interpretabilidad.
