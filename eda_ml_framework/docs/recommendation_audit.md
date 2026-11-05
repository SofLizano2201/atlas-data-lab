# Auditoría de coherencia de recomendaciones

Revisión del 2 de noviembre de 2026.

## Corregido

- Selección inicial de variables: excluye por defecto nombres explícitos de ID,
  fechas reconocidas por nombre/tipo y columnas constantes. La unicidad de una
  variable continua (por ejemplo precio) no basta para excluirla.
- Clustering manual advierte si el usuario conserva esas columnas. Las políticas
  guiadas piden revisar la configuración antes de entrenar con ellas.
- La imputación recomendada no completa identificadores ni fechas con mediana o moda.
- Un identificador como objetivo genera una advertencia para revisar qué predecir.
- El agente usa la selección actual de variables y escalamiento, no una copia antigua.
- Los cálculos de agrupamiento para más de 5000 filas no se ofrecen como ejecutables.
- La configuración del agente excluye variables desaconsejadas sin borrar columnas.
- Preparación conserva vista previa, confirmación y versiones. Comparación conserva
  las restricciones de cohorte y usa métricas calculadas por el modelo.

## Limitaciones explícitas

Los roles se identifican con reglas conservadoras de nombre/tipo; no se conoce el
significado de todas las columnas. El usuario debe revisar variables ambiguas.
Un identificador llamado de otra forma puede requerir exclusión manual.

La muestra del formulario manual de Clustering no se transfiere automáticamente al
agente. Para datasets grandes, el asistente lo explica y dirige al flujo manual;
no debe afirmar que sus resultados corresponden a una muestra que no utilizó.

La mediana/moda y el umbral de faltantes son alternativas iniciales, no una garantía
estadística universal. No se eliminan observaciones automáticamente. Las comparaciones
no garantizan estabilidad ni generalización fuera de la evaluación registrada.

Esta auditoría prueba los flujos y casos descritos; no certifica todas las combinaciones
posibles de datasets, dependencias opcionales, algoritmos e hiperparámetros.

## Validación

Batería final: 238 pruebas aprobadas en 60.86 s. Siete advertencias conocidas:
una media sobre una columna vacía y seis avisos de convergencia MLP en pruebas cortas.
Ruff: sin errores.
