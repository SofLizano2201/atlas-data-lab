# Verificación para publicar Atlas 1.1

27 de octubre de 2026. Revisión de todos los cambios locales antes de crear un solo
commit en el repositorio existente.

| Verificación ejecutada | Resultado |
|---|---|
| `python -m pytest -q` | **155 passed, 7 warnings in 35.06s** |
| `python -m ruff check .` | **All checks passed!** |
| `python -m ruff format --check .` | **68 archivos ya formateados** |
| `python -m pip check` | **No broken requirements found** |
| `pip wheel --no-deps --wheel-dir artifacts/dist .` | **Wheel 1.1.0 construido correctamente** |
| Instalación del wheel en carpeta aislada | **Imports de framework, frontend, agents y contracts correctos** |
| Recursos dentro del wheel | **CSS de composición, ilustración Atlas y SVG Lucide presentes** |
| Streamlit en instancia nueva | **Arranque correcto en 127.0.0.1:8504; salud `ok`** |
| Edge headless con Iris | **Nueve páginas sin errores; tarjetas alineadas; navegación funcional** |
| Sidebar | **260 px expandida, 72 px compacta; filas de 46 px** |
| Vista móvil | **390 px de viewport y documento, sin desbordamiento horizontal** |
| `git diff --check` | **Sin errores de espacios** |

Las siete advertencias son las ya documentadas: seis de convergencia de MLP con
20 iteraciones en pruebas y una de estadísticas sobre una columna completamente
vacía. No se suprimen. La construcción se verificó con el entorno aislado de PEP 517;
no depende de tener `wheel` instalado en el entorno de ejecución de la aplicación.

El wheel se instaló sin modificar el entorno de la aplicación, en una carpeta
temporal ignorada por Git. Se comprobó que las importaciones procedieran realmente
de esa instalación. Las dependencias de ejecución son las comprobadas por `pip check`;
no se afirma haber repetido su descarga en un equipo nuevo.

Los detalles de las etapas anteriores están en
[integración](verification_2026_10_27.md), [mockups](mockups_2026_10_27.md) y
[visualizaciones](visualization_standard.md). Esta revisión no convierte las
limitaciones allí descritas en funcionalidades verificadas: la llamada externa al
LLM requiere configuración y consentimiento, las integraciones opcionales mantienen
su detección controlada, y no hay autenticación ni envío programado de informes.

Se excluyen del repositorio entornos virtuales, artefactos de compilación, capturas
de sesiones, modelos exportados, ZIP generados y credenciales. `.env.example` contiene
solo una plantilla sin claves reales.
