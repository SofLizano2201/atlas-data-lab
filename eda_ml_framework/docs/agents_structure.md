# Organización modular de los agentes

La implementación canónica está en `src/atlas/agents/`. Cada clase tiene una sola
definición. Los modelos científicos, controladores y vistas conservan sus
responsabilidades MVC; esta reorganización no cambia algoritmos ni permisos.

```text
src/atlas/agents/
├── __init__.py            API pública de los cinco agentes y la fábrica
├── base.py                BaseAgent: ejecución, validación y confirmaciones
├── eda_agent.py           EDAAgent: exploración y preparación
├── clustering_agent.py    ClusteringAgent: agrupamientos
├── classification_agent.py ClassificationAgent: categorías
├── regression_agent.py    RegressionAgent: predicciones numéricas
├── comparison_agent.py    ModelComparisonAgent: comparación de experimentos
├── factory.py             Registro de especialistas, alias e inyección
├── contracts.py           Planes del proveedor y solicitudes pendientes
├── providers.py           Proveedor opcional y transporte; modo sin LLM
├── policies.py            Reglas declarativas, seguridad y textos compartidos
├── defaults.py            Parámetros iniciales para capacidades existentes
├── planning.py            Selección de acciones y comprobación de requisitos
├── guidance.py            Ideas y orientación contextual sin cálculos científicos
├── events.py              Reacción a eventos con evidencias autorizadas
└── agent.py               Adaptador para importaciones anteriores
```

Los tres auxiliares de comportamiento (`events`, `guidance`, `planning`) no crean
agentes independientes: `BaseAgent` hereda esas responsabilidades compartidas y
los cinco especialistas heredan de `BaseAgent`. No mantienen una segunda sesión
ni duplican entrenamientos. Cada especialista define sus instrucciones,
herramientas permitidas e interpretación mediante la misma interfaz polimórfica.

## Uso recomendado

```python
from atlas.agents import AgentContext, AgentFactory, EDAAgent
from atlas.agents.providers import NoProvider

# eda es una instancia existente del modelo de exploración.
context = AgentContext(eda)
assistant = AgentFactory.get_agent("eda", context, NoProvider())
assert isinstance(assistant, EDAAgent)
overview = assistant.suggestions()
```

Las importaciones anteriores desde `atlas.agents.agent` y el paquete raíz
`agents` siguen apuntando a las mismas clases. No hay una implementación antigua
en paralelo. El backend nuevo usa los módulos canónicos; los adaptadores se
conservan para consumidores y pruebas de compatibilidad.

## Dónde realizar un cambio

- Instrucciones, permisos o interpretación de una especialidad: su archivo
  `*_agent.py`.
- Reglas locales o textos compartidos: `policies.py`; coordinación de planes:
  `planning.py`; propuestas iniciales: `defaults.py`.
- Confirmaciones y ejecución autorizada: `base.py`. El gateway sigue en
  `controllers/capabilities.py` y continúa validando las operaciones.
- Nuevo proveedor: `providers.py`, respetando `LLMProvider`; no introducir
  proveedores en vistas o controladores.
- Nuevo agente: crear una subclase de `BaseAgent`, registrar la clase en
  `factory.py`, exportarla en `__init__.py` y probar sus permisos e interpretación.
- Nuevas fórmulas, métricas o algoritmos: `models/`, nunca la capa de agentes.

Los contratos científicos compartidos continúan en `core/schemas.py` y el
contexto aislado de sesión en `core/session.py`. Los contratos propios de la
planificación opcional están en `agents/contracts.py`.

Las pruebas de arquitectura exigen una definición por clase, un módulo por
especialista, compatibilidad de importaciones y ausencia de dependencias
científicas o de Streamlit dentro de los agentes.

## Verificación de la reorganización

- Suite completa: **289 pruebas aprobadas** en 79,78 segundos.
- Ruff: sin errores.
- Se verifican polimorfismo, contratos, permisos, confirmaciones, proveedores,
  interpretación de evidencias, flujo manual sin IA e integración Streamlit AppTest.
- Siete advertencias de cálculo: un caso de columna completamente vacía y seis
  avisos de convergencia de MLP limitados a 20 iteraciones en las pruebas.
- Los contratos y proveedores no se recargan como estilos de la interfaz: se
  mantiene la identidad de sus clases entre interacciones de Streamlit.
