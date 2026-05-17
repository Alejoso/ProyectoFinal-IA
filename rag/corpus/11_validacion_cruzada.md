# Validación cruzada en problemas de clasificación binaria desbalanceados

La validación cruzada (cross-validation) es una técnica para estimar el rendimiento real de un modelo de manera robusta, reduciendo la dependencia de un único split aleatorio en train y test.

**K-Fold Cross-Validation** divide el conjunto de entrenamiento en k partes (folds) iguales. El modelo se entrena k veces, cada vez usando k-1 folds para entrenar y el fold restante para evaluar. Las métricas finales son la media y desviación estándar de las k evaluaciones.

**Stratified K-Fold** es una variante que preserva la proporción de clases en cada fold. En problemas desbalanceados como credit scoring, esto es crítico: con K-Fold normal, por azar un fold podría tener muy pocos casos de la clase minoritaria, dando estimaciones inestables. Stratified K-Fold garantiza que cada fold tenga aproximadamente la misma proporción de buenos y malos pagadores que el dataset completo.

**Número de folds recomendado.** Para datasets pequeños como German Credit (1000 registros), 5 folds es un buen compromiso: cada fold tiene 200 registros, suficiente para evaluación estable. Con 10 folds los folds serían de 100 registros, lo que aumenta la varianza. Con menos de 5 folds se desperdicia información.

**Interpretación de resultados.** Un modelo con AUC 0.78 ± 0.01 es más confiable que uno con AUC 0.79 ± 0.05, aunque la media sea menor. La desviación estándar indica robustez ante distintos splits. Modelos con alta varianza pueden estar sobreajustando o ser sensibles al azar de los datos.

**Validación cruzada para tuning de hiperparámetros.** Herramientas como `GridSearchCV` y `RandomizedSearchCV` de sklearn ejecutan validación cruzada para cada combinación de hiperparámetros, eligiendo automáticamente la mejor configuración según una métrica de scoring definida (por ejemplo, `roc_auc` para credit scoring).

**Importante.** La validación cruzada se hace **solo sobre el train**, no sobre el test. El test permanece intocado hasta la evaluación final, para evitar sobreajuste indirecto a través de múltiples evaluaciones.
