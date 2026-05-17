# PD, LGD y EAD: los tres componentes del riesgo crediticio según Basilea

El Acuerdo de Basilea II introdujo el enfoque IRB (Internal Ratings-Based) para que las entidades financieras estimen su riesgo de crédito a partir de tres componentes fundamentales: PD, LGD y EAD.

**PD (Probability of Default)** es la probabilidad de incumplimiento del deudor en un horizonte de un año. Se estima típicamente con modelos de clasificación binaria — exactamente lo que se construye en un proyecto de credit scoring como el que utiliza el dataset German Credit. Es la salida directa de algoritmos como regresión logística o Random Forest entrenados sobre datos históricos de incumplimiento.

**LGD (Loss Given Default)** es la pérdida esperada en caso de que ocurra incumplimiento, expresada como porcentaje del monto expuesto. Si un cliente debe 10.000 EUR y en caso de default el banco recupera 4.000 (por colateral o cobranza), la LGD es del 60%. Variables como `propiedad` y `deudores_garantes` se relacionan con la LGD: un cliente con bienes raíces como respaldo implica una LGD menor.

**EAD (Exposure at Default)** es el monto total al que está expuesta la entidad en el momento del incumplimiento. En créditos de cuota fija es simplemente el saldo pendiente. La variable `monto_credito` del dataset es una aproximación a la EAD inicial.

La pérdida esperada (Expected Loss, EL) se calcula como: **EL = PD × LGD × EAD**.

Esta fórmula explica por qué el credit scoring (que estima PD) es solo una parte del análisis de riesgo. Sin embargo, es la parte más estandarizable y donde más se aplican técnicas de machine learning.
