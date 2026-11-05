# Asistentes naturales y contextuales

Actualización del 2 de noviembre de 2026 sobre el proyecto existente.

## Diagnóstico y experiencia

El panel anterior exponía selección de herramientas, parámetros y estados internos.
Ahora los cinco especialistas comparten un diálogo contextual: al abrirlo muestran
entre dos y tres ideas útiles, sin ejecutar entrenamientos ni modificar datos.
La bienvenida utiliza el dataset, el objetivo vigente y los resultados disponibles.

Seleccionar una idea permite revisar información existente o explica una propuesta.
Las acciones costosas y los cambios requieren **Continuar / Cancelar**. Cada propuesta
explica por qué ayuda, qué se espera obtener y qué cambiará. Las consultas informativas
no añaden confirmaciones. Los resultados proceden de herramientas reales del framework.

Los identificadores, herramientas, parámetros e historial técnico solo se renderizan
al activar **Ver detalles técnicos**. La asistencia local no exige credenciales ni
pretende comprender preguntas libres. La conversación con un proveedor configurado
sigue siendo opcional.

## Especializaciones

- Datos y EDA: información incompleta, filas repetidas, distribuciones y relaciones.
- Agrupamiento: preparación de variables, búsqueda y revisión de grupos existentes.
- Clasificación: categorías del objetivo actual, primer modelo y errores existentes.
- Regresión: objetivo numérico, predicciones y diferencias respecto a valores reales.
- Comparación: alternativas compatibles y selección según la medida calculada por el
  comparador existente. Si faltan modelos, ofrece una ruta válida para crearlos.

No se ofrecen errores de un modelo cuyo objetivo difiere del seleccionado actualmente.
Sin dataset, el diálogo dirige a la importación. Las ideas validan los requisitos de
las capacidades autorizadas antes de presentarse.

## Arquitectura y continuidad

Las políticas, especializaciones y explicaciones permanecen en
`src/atlas/agents/agent.py`, con BaseAgent y cinco implementaciones polimórficas.
AssistantIdea y AssistantOverview son contratos tipados. AgentController coordina
selección, confirmaciones y estado; agent_panel.py renderiza un componente compartido.
Los modelos siguen realizando todos los cálculos científicos.

Los metadatos ligeros se reutilizan dentro de la sesión y versión del dataset.
Cerrar y reabrir conserva una revisión completada sin ejecutarla otra vez. Cerrar
una propuesta pendiente cancela su autorización. Un cambio externo de dataset,
versión o configuración invalida propuestas anteriores; nuevas alternativas en el
historial actualizan las ideas. No se comparten caches de datos entre usuarios.

## Verificación

**229 pruebas aprobadas**, en 54.22 segundos. Siete advertencias conocidas: una media
sobre una columna vacía y seis avisos de convergencia MLP en entrenamientos cortos.
Las pruebas cubren los cinco agentes, permisos, ejecución real, confirmaciones,
reapertura, navegación, cambios de objetivo, dataset vacío y funcionamiento manual.

La revisión Edge/CDP captura las diez páginas y los cinco asistentes. Se corrigió un
recorte real entre el contenedor nativo y el contenido del diálogo en escritorio.
El diálogo mide 700 px en escritorio y 358 px en un viewport móvil de 390 px, sin
ensanchar el documento. Las capturas locales están en `artifacts/atlas-2.0/` y no se
publican automáticamente en Git. La disponibilidad de un LLM generativo no forma
parte de esta verificación; el motor local sí está probado.

## Límites

Las recomendaciones son políticas explícitas basadas en evidencia, no garantía de
causalidad ni de un modelo óptimo. El diálogo móvil permite desplazamiento vertical.
Los controles avanzados conservan su terminología técnica por elección del usuario.

## Recomendaciones por etapa (actualización)

Preparación usa los diagnósticos existentes por columna: propone una vista previa
con mediana para numéricas parcialmente incompletas y moda para categóricas; columnas
con al menos 50% de faltantes se presentan para revisar una posible exclusión, no para
eliminarlas automáticamente. También permite revisar duplicados exactos y columnas
constantes. Estas reglas son puntos de partida y explican sus limitaciones. La vista
previa y la aplicación son acciones distintas, con confirmación de aplicación.

Clustering ofrece probar 2, 3 y 4 grupos mediante el entrenamiento existente de K-Means,
con los mismos predictores y escalamiento, después de confirmar. La búsqueda local está
limitada a 10–5000 registros. La comparación usa el comparador científico existente y
explica el número de grupos favorecido según la métrica, sin afirmar estabilidad ni
optimalidad universal. No compara cohortes incompatibles ni vuelve a entrenar al revisar.

Verificación de esta actualización: batería completa de 232 pruebas aprobadas, siete
advertencias conocidas; posteriormente 20 pruebas específicas aprobadas, incluida la
nueva prueba de aplicar una limpieza desde el diálogo solo tras confirmarla.
