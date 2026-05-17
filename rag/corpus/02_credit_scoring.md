# Credit scoring y probabilidad de incumplimiento

El credit scoring es una técnica estadística que asigna una puntuación numérica a un solicitante de crédito, representando la probabilidad de que cumpla o incumpla con la obligación. Esta puntuación permite a las entidades financieras tomar decisiones objetivas, consistentes y auditables sobre la aprobación o rechazo de solicitudes.

El concepto central detrás del credit scoring es la **probabilidad de incumplimiento** (Probability of Default, PD), definida como la probabilidad de que un cliente no cumpla con sus obligaciones financieras en un horizonte de tiempo determinado, típicamente 12 meses.

Los modelos de credit scoring producen una probabilidad continua entre 0 y 1, que luego se compara contra un umbral de decisión. Si la PD estimada supera el umbral, el crédito se rechaza. La elección del umbral no es técnica sino de negocio: depende del apetito de riesgo de la entidad y del costo asimétrico de los errores.

Los algoritmos más utilizados históricamente son la regresión logística (por su interpretabilidad y por requisitos regulatorios) y, más recientemente, modelos basados en árboles como Random Forest y Gradient Boosting (XGBoost, LightGBM), que ofrecen mayor poder predictivo a costa de menor interpretabilidad directa.

En el dataset German Credit, el objetivo es exactamente este: predecir si un solicitante será un "mal pagador" (clase 1) o un "buen pagador" (clase 0) a partir de sus características demográficas y financieras.
