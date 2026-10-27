# Diagnóstico inicial — 27 de octubre de 2026

Se conserva el proyecto existente y su jerarquía académica. Se inspeccionaron las
clases, auxiliares, páginas, componentes, estado, dependencias y pruebas actuales.

## Evidencia de partida

- Suite existente: **99 passed, 6 warnings in 19.83s** en Python 3.11, fuera del
  sandbox. Las advertencias son MLP con 20 iteraciones deliberadas en pruebas.
- El entorno virtual funciona con la cuenta propietaria; el sandbox no puede iniciar
  su intérprete. Git también requiere la cuenta propietaria por seguridad de ownership.
- Framework separado de Streamlit, herencia real y pipelines con preprocesamiento
  dentro de CV. Comparaciones validan cohorte y no reentrenan.
- No se encontraron botones decorativos ni métricas simuladas en las páginas inspeccionadas.
  Las acciones llaman al framework; varios resultados de botones desaparecen tras un
  rerun porque solo algunas figuras/resultados se guardan explícitamente en la sesión.

## Faltantes y defectos identificados

1. No existe capa de agentes, proveedor LLM, contratos de herramientas ni consentimiento.
2. Reportes son HTML/tablas y descargas; faltan PDF, selección de secciones e historial.
3. El dataset original de frontend es una copia mutable; faltan identificadores y
   contratos de versiones estables, aunque EDA ya tiene copias defensivas e historial.
4. Experimentos registran cohorte/configuración, pero faltan metadatos explícitos de
   dataset/modelo y versiones de dependencias exportables.
5. Sidebar no respeta la nueva paleta exacta y aún hay emojis en rutas internas y
   accesos rápidos; se actualizará la presentación preservando rutas antiguas mediante alias.
6. En clasificación con decision_function ovo no se deben interpretar sus columnas
   como puntuaciones OvR. También deben limitarse todos los candidatos de búsquedas,
   no solo el estimador inicial, y acotarse contingencias de alta cardinalidad.
7. No se adjuntaron mockups a este mensaje. No puede afirmarse fidelidad ni realizarse
   una comparación visual contra referencias que no están disponibles. Se conservará
   la composición existente y se ajustarán tokens y controles a la nueva especificación.

## Integración prevista

`framework` conserva cálculos; `agents` añade contratos, contexto, proveedor, registro
de herramientas y cinco especialidades. El frontend renderiza resultados tipados y
confirma planes antes de operaciones costosas o cambios. Sin configuración LLM,
acciones deterministas siguen disponibles y la conversación se identifica como no disponible.

Validación por capas: regresión de suite existente, contratos/seguridad, herramientas,
proveedores con transporte de prueba explícito, integración Streamlit, servidor y
capturas. Una prueba con transporte simulado no se presentará como llamada real al LLM.
