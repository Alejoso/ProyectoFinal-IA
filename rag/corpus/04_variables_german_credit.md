# Variables del dataset German Credit Data

El dataset German Credit Data del UCI Machine Learning Repository contiene 1000 registros de solicitantes de crédito en Alemania, con 20 variables predictoras y una variable objetivo binaria. A continuación se describen las variables más relevantes para el análisis de riesgo:

**estado_cuenta** (A11-A14): estado de la cuenta corriente del solicitante. Es la variable más predictiva del dataset. A11 indica saldo negativo (mayor riesgo), A12 saldo entre 0 y 200 DM, A13 saldo mayor a 200 DM, y A14 indica que el solicitante no tiene cuenta corriente registrada. Paradójicamente, A14 tiene menor tasa de incumplimiento que A11.

**historial_credito** (A30-A34): historial de pagos pasados. A30 significa sin créditos previos o todos pagados, A31 todos los créditos en este banco pagados al día, A32 créditos existentes pagados al día hasta ahora, A33 retrasos en pagos anteriores, A34 cuenta crítica u otros créditos existentes en otros bancos. Contraintuitivamente, A34 puede mostrar menor riesgo que A30 porque "sin historial" representa incertidumbre.

**proposito** (A40-A410): destino del crédito. Va desde compra de carro nuevo (A40) hasta educación (A46), negocio (A49) y otros. El propósito influye en el riesgo: créditos para reentrenamiento o educación tienden a tener mayor tasa de incumplimiento.

**monto_credito**: monto solicitado en marcos alemanes. Variable numérica con cola larga a la derecha (sesgada).

**duracion_meses**: plazo del crédito en meses. Va de 4 a 72 meses. Plazos más largos están correlacionados con mayor riesgo.

**ahorros** (A61-A65): nivel de ahorros del solicitante, ordinal desde menos de 100 DM hasta más de 1000 DM. A65 mezcla "desconocido" con "sin ahorros".

**empleo_actual** (A71-A75): antigüedad en el empleo actual, desde desempleado hasta más de 7 años.

**edad**: edad del solicitante en años, de 19 a 75.

La variable objetivo `target` originalmente codifica 1=buen pagador, 2=mal pagador. En este proyecto se invierte a 0=buen pagador, 1=mal pagador siguiendo la convención de que la clase positiva es la que se quiere detectar.
