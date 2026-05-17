# Las 5C del crédito

Las 5C del crédito son el marco tradicional que utilizan las entidades financieras para evaluar el riesgo crediticio de un solicitante. Son: carácter, capacidad, capital, colateral y condiciones.

**Carácter** se refiere a la historia y disposición del solicitante para cumplir con sus obligaciones financieras. Se evalúa a través del historial crediticio, referencias y comportamiento de pago en créditos anteriores. En el dataset German Credit, la variable `historial_credito` (categorías A30 a A34) representa directamente este factor.

**Capacidad** mide la habilidad del solicitante para generar ingresos suficientes y pagar la deuda. Se evalúa con el nivel de empleo, antigüedad laboral e ingresos. Las variables `empleo_actual` y `trabajo` del dataset capturan este componente.

**Capital** es el patrimonio o ahorros del solicitante. Indica respaldo propio frente a imprevistos. La variable `ahorros` mide directamente este factor.

**Colateral** son los bienes que respaldan el crédito en caso de incumplimiento. Las variables `propiedad` y la presencia de garantes (`deudores_garantes`) representan esta dimensión.

**Condiciones** incluye el propósito del crédito, el monto solicitado y el plazo, así como condiciones económicas externas. Las variables `proposito`, `monto_credito` y `duracion_meses` corresponden a este factor.

Un buen modelo de credit scoring debe capturar información de las cinco dimensiones para producir una evaluación robusta del riesgo.
