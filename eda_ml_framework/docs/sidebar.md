# Sidebar Atlas

La composición y paleta actuales se describen en
[adaptación de los mockups](mockups_2026_10_27.md). Los colores e isotipo indicados
más abajo documentan el diseño inicial, posteriormente actualizado.

El menú usa botones nativos de Streamlit. Las nueve rutas existentes y la clave
`st.session_state.navigation` se conservan; las etiquetas visibles ya no contienen
emojis ni indicadores radio. Los accesos rápidos de Inicio siguen usando el mismo estado.
No se modificaron algoritmos ni lógica de datasets o experimentos.

## Diseño y comportamiento

- Ciruela oscuro #241b29, superficies #2d2333/#3a2c43, marfil #eee9f1,
  secundarios #b5a7be y acento lavanda #c4aedb.
- Identidad Atlas con isotipo Network, subtítulo y separación sutil.
- Filas de 44 px, iconos outline de 19 px, espaciado de 4 px y selección con fondo
  redondeado y línea interior de 2 px. Hover de 160 ms y foco de teclado visible.
- Escritorio: 260 px; botón de doble chevrón alterna a 72 px. Las etiquetas permanecen
  en el árbol de accesibilidad y los botones conservan tooltips nativos.
- Móvil ≤768 px: menú de hasta 260 px, textos visibles y cierre/apertura nativos de
  Streamlit. El control de rail compacto se oculta, conservando su preferencia para escritorio.
- Tarjeta del dataset obtiene el nombre del estado real, lo escapa en HTML y aplica
  ellipsis. La tarjeta se actualiza tras las acciones de página en el mismo rerun.
- Pie compacto con explicación completa mediante title y aria-label. Datos y modelos
  siguen en memoria por sesión; el toggle nunca reinicia su estado.
- Tipografía Inter si está instalada, con fuentes del sistema como fallback. No se
  descargan fuentes ni iconos durante el uso.

## Archivos

`frontend/components/sidebar.py`: catálogo, callbacks, estados, SVG y composición.
`frontend/styles/theme.css`: apariencia y responsive, limitado al sidebar por claves propias.
`app.py`: integra el componente y actualiza la tarjeta después de renderizar cada página.
`frontend/assets/lucide/`: SVG oficiales locales y licencia original.
`pyproject.toml`: incluye SVG, CSS y licencia en los paquetes.
`tests/test_integration.py`: navegación y estado adaptados a botones, prueba de modo compacto,
escape del nombre del dataset y permanencia del estado.
`tools/sidebar_browser_check.py`: QA opcional en Edge headless usando dependencias ya
instaladas; no es una dependencia de producción.

Los selectores usan claves `st-key-atlas_*` y testids semánticos. No dependen de clases
`st-emotion-cache` generadas. Los SVG se aplican como máscaras CSS a los botones reales;
no son enlaces decorativos desconectados del router.

## Origen y licencia de iconos

SVG de [Lucide Icons](https://github.com/lucide-icons/lucide/tree/main/icons).
La [licencia original](../frontend/assets/lucide/LICENSE) incluye los avisos ISC/MIT
correspondientes. No se añadió una biblioteca ni un CDN al runtime.

## Verificación

Pruebas de integración: las nueve páginas, accesos rápidos, dataset activo, cambios
confirmados, clasificación, regresión, clustering y navegación compacta.
La prueba visual local guarda capturas en `artifacts/sidebar/`: escritorio, dataset,
compacto y móvil, más medidas y estructura de contenedores para diagnóstico.
La comprobación utiliza Edge instalado y un perfil temporal separado de las sesiones
del usuario. Para repetirla, inicia Streamlit en 127.0.0.1:8501 y ejecuta
`python tools/sidebar_browser_check.py`.
