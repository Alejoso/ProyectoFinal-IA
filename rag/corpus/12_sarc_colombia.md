# SARC y la regulación colombiana del riesgo crediticio

En Colombia, la gestión del riesgo crediticio en entidades financieras está regulada por la Superintendencia Financiera a través del SARC (Sistema de Administración de Riesgo Crediticio), establecido en el Capítulo II de la Circular Básica Contable y Financiera (Circular Externa 100 de 1995).

**Objetivos del SARC.** El SARC busca que las entidades financieras identifiquen, midan, controlen y monitoreen el riesgo de crédito asociado a su cartera. Aplica a bancos, corporaciones financieras, compañías de financiamiento, cooperativas financieras y entidades similares.

**Componentes del SARC:**

*Políticas de crédito*: deben estar documentadas y aprobadas por la Junta Directiva. Incluyen criterios de otorgamiento, límites de exposición, y políticas de cobranza.

*Procesos de administración del riesgo*: deben cubrir todas las etapas del ciclo de vida del crédito: otorgamiento, seguimiento, cobranza y recuperación.

*Modelos internos de calificación y provisiones*: las entidades deben implementar modelos para estimar las probabilidades de incumplimiento y las pérdidas esperadas. Estos modelos deben ser validados periódicamente.

*Documentación y auditoría*: todos los modelos y decisiones de crédito deben estar documentados y ser auditables, tanto interna como externamente por la Superintendencia.

**Relación con Basilea.** El SARC adopta los principios del Acuerdo de Basilea II, particularmente la estimación de PD (Probability of Default), LGD (Loss Given Default) y EAD (Exposure at Default) para el cálculo de la pérdida esperada. Las entidades grandes en Colombia pueden usar modelos internos (enfoque IRB), mientras que las pequeñas usan el enfoque estandarizado con factores predefinidos por el regulador.

**Implicaciones para credit scoring.** Cualquier modelo de credit scoring usado por una entidad colombiana regulada debe:

1. Estar documentado, con justificación de variables y técnicas usadas.
2. Ser interpretable o tener mecanismos de explicación de decisiones individuales.
3. Ser validado periódicamente con métricas estándar (KS, AUC, Gini).
4. Tener procesos de backtesting (comparación de PD estimada con incumplimiento observado).
5. No discriminar por variables prohibidas (raza, género en algunos contextos, religión).

El dataset German Credit, aunque proviene de Alemania de los años 90, contiene la variable `estado_civil_sexo` que mezcla sexo y estado civil. En un contexto regulatorio colombiano actual, usar directamente esta variable podría considerarse discriminatorio y requeriría análisis adicional para verificar que no introduce sesgo en las decisiones.
