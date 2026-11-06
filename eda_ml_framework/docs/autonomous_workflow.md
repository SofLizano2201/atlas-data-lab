> Nota de arquitectura: la organización actual de agentes está documentada en
> [agents_structure.md](agents_structure.md). Las referencias al antiguo archivo
> único describen la estructura anterior a la modularización.

# Recorrido autónomo local

Al confirmar un archivo en Datos, la opción **Analizar todo automáticamente al
confirmar** está activada por defecto. La confirmación inicia el trabajo, no
solo la carga del archivo. Se puede desactivar para trabajar manualmente.
Con datasets ya activos o ejemplos, **Analizar todo automáticamente** inicia
el mismo recorrido desde Inicio o Datos.

## Qué hace

1. Reutiliza el diagnóstico EDA y sus gráficos calculados.
2. El agente EDA propone un objetivo provisional. Prioriza nombres como `target`,
   `price` o `churn`; si no encuentra uno, usa la última columna compatible y
   advierte expresamente que esa elección no demuestra un objetivo real.
3. Excluye identificadores, fechas sin preparar, constantes, columnas casi vacías
   y categorías excesivas de los predictores. Usa hasta 12 características en
   orden del esquema; no se afirma que sea una selección óptima.
4. Explora clasificación o regresión, según las características del objetivo,
   con tres familias: modelo sencillo, árbol y bosque de 40 árboles.
5. Prepara predictores mediante el pipeline existente: imputación numérica con
   mediana, categorías con la estrategia existente, codificación y escalamiento.
   Los transformadores se ajustan dentro de entrenamiento y de cada fold.
6. Compara usando validación cruzada de entrenamiento de tres particiones:
   F1 macro para clasificación y RMSE negativo para regresión. Mayor puntuación
   es mejor en ambos casos. La prueba final del 25 % no decide el ganador.
7. Prueba K-Means con 2, 3 y 4 grupos, y agrupamiento aglomerativo con 3 grupos.
   La silueta orienta la recomendación entre los resultados compatibles.
8. Genera gráficos, registra los experimentos en el almacén compartido y crea
   un informe HTML autónomo con Plotly embebido, decisiones, variables,
   configuraciones, métricas, predicciones de muestra y advertencias.

Los cinco agentes intervienen mediante la interfaz polimórfica
`BaseAgent.autonomous_plan`. El especialista que no corresponda al tipo de
predicción explica por qué no debe entrenar para ese objetivo.

## Componentes

- `agents/agent.py`: todas las políticas y explicaciones agénticas.
- `controllers/autonomous_controller.py`: coordinación, presupuestos, registro
  de resultados y ensamblado del informe; delega cálculos a controladores existentes.
- `views/components/autonomous_panel.py`: resumen, gráficos, decisiones y
  modificación opcional del objetivo.
- `views/utils/state.py`: progreso de la ejecución explícitamente iniciada.
- `views/pages/workspace.py`: inicio automático tras confirmar archivos.
- `views/pages/reports.py`: descarga del informe automático.
- `services/dataset_service.py`: aislamiento y reinicio al activar otro dataset.

## Reproducibilidad y límites

No cambia el original ni el dataset de trabajo. No elimina filas duplicadas o
outliers sin una decisión del usuario y no imputa el objetivo: un objetivo
incompleto bloquea la predicción y deja disponibles los demás análisis.

Se usa semilla 42, hasta 3.000 filas para predicción y 1.000 para agrupamiento.
Cada comparación conserva las mismas variables, partición y filas. El muestreo
se indica en resultados e informe; las métricas describen la selección usada.

Existe un presupuesto de 90 segundos comprobado entre operaciones; no es una
interrupción forzada de una operación en curso. No se ejecutan todos los modelos,
todas las combinaciones de parámetros o todas las posibles variables: se realiza
una exploración inicial acotada. Se conservan resultados parciales ante fallos.

La recomendación es la mejor **entre las alternativas probadas según el criterio
indicado**, no un óptimo universal. La elección automática del objetivo no
conoce el propósito del usuario. Tampoco detecta por sí sola todas las fugas
semánticas, agrupaciones de sujetos o dependencias temporales; la validación
aleatoria no sustituye un diseño de evaluación propio del dominio.

Los resultados se reutilizan por identidad/versión del dataset y objetivo;
interactuar con controles o volver a abrirlos no entrena de nuevo. Cambiar el
objetivo crea otra ejecución conservando experimentos anteriores. Todo reside
en la sesión y no requiere proveedor externo ni credenciales.

El informe contiene las primeras 50 filas de tablas exploratorias y 100
predicciones/etiquetas por experimento. El resto de las predicciones y modelos
se exportan desde sus módulos. El formato automático es HTML, no PDF.

## Validación

`tests/test_autonomous_workflow.py` ejecuta clasificación Iris y regresión
Diabetes reales, comprueba selección por CV, ausencia del objetivo en los
predictores, particiones disjuntas, preservación de originales, repetición sin
reentrenar, objetivo incompleto, presupuesto agotado y reinicio de dataset.
También ejecuta el botón y cambia gráficos mediante Streamlit AppTest.

Suite completa tras la implementación: **273 pruebas aprobadas**, 7 advertencias
de NumPy/MLP ya presentes. Ruff aprobado. No se introdujeron dependencias.

```powershell
..\.venv\Scripts\python.exe -m pytest tests/test_autonomous_workflow.py -q
..\.venv\Scripts\python.exe tools/sidebar_browser_check.py --port 8513 --all-pages --autonomous
```
