# EDA inicial del equipo — Phase 1

Fecha del análisis: **1 de noviembre de 2026**.

El [informe completo del EDA](INFORME_EDA.md) analiza los archivos originales de tracking, homografía, Smart Tagging y metadatos de vídeo. Incluye diez gráficos y los métodos necesarios para reproducir los resultados localmente.

## Hallazgos que conviene leer primero

| Hallazgo | Primera parte | Segunda parte |
|---|---:|---:|
| Frames con posición de balón | 59,35 % | 59,27 % |
| Frames con exactamente 11 tracks por equipo | 42,41 % | 41,49 % |
| Balón + 11 tracks por equipo + homografía | 32,78 % | 29,21 % |
| Mayor racha sin posición de balón | 138,64 s | 221,48 s |

Los denominadores son los bloques oficiales completos de las partes, incluidas las interrupciones. La ausencia de balón no implica balón muerto. Contar 11+11 tracks tampoco certifica 22 jugadores diferentes y bien localizados.

Otros puntos relevantes:

- No se entregan bounding boxes ni identidades reales estables; se entregan posiciones y equipo por slot de tracking.
- La homografía táctica está disponible en el 100 % de los frames, pero su disponibilidad no demuestra precisión.
- La velocidad de jugadores es mucho más coherente con m/s que con los km/h declarados por el XML; es un contraste interno, pendiente de confirmación del proveedor.
- El vídeo broadcast declara 164.487 frames frente a 164.488 filas en tracking y homografía. No se ha identificado el origen de la diferencia ni aplicado una corrección arbitraria.
- Casi todos los frames sin balón pertenecen a huecos de al menos cinco segundos. Interpolar únicamente huecos breves no resolverá la mayor parte de la ausencia.
- Los eventos automáticos pueden solaparse entre equipos; su atributo Team no equivale directamente al poseedor.

## Siguiente línea de trabajo

Bernardo empezará por su lado un **piloto de nueva detección y tracking**, con especial atención al balón, conservando los datos originales como referencia. Primero se comprobará si aporta observaciones útiles y coherentes; después se estudiará su integración en la inferencia de posesión.

El EDA justifica probar esta vía, pero **todavía no demuestra que YOLO u otro tracker mejoren la posesión**. Localizar el balón, inferir control y determinar si está en juego son problemas relacionados que requieren lógica temporal adicional.

## Reproducir el análisis

Desde la raíz del repositorio, con Python 3.12 y los datos oficiales descargados por cada participante:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
.\.venv\Scripts\python.exe starter\eda_initial_data.py --data-dir 'RUTA_AL_DATASET' --out evaluation\reports\eda-inicial
```

La ruta debe contener las carpetas `tactical/` y `broadcast/` con los nombres oficiales. El programa conserva los originales y genera informes, estadísticas y gráficos en una carpeta ignorada por Git. Para repetir la auditoría de cobertura:

```powershell
.\.venv\Scripts\python.exe starter\audit_tracking_coverage.py --data-dir 'RUTA_AL_DATASET' --out evaluation\reports\tracking-coverage
```

Se comprobó la integridad de los índices, los agregados por slot, la partición temporal y la coherencia con la auditoría anterior. Los vídeos se comprobaron mediante cabecera y decodificación de muestras, no con una revisión exhaustiva de todos los fotogramas.

## Alcance de esta publicación

Se comparten documentación, gráficos estadísticos agregados y código. No se incluyen vídeos, fotogramas, coordenadas por frame, etiquetas de posesión nuevas ni archivos originales del dataset. Los CSV y JSON generados localmente no forman parte de esta publicación.

No se utilizaron etiquetas públicas de ground truth para entrenar o ajustar parámetros. Las limitaciones de uso y distribución de los datos siguen siendo las de [RULES.md](../../RULES.md) y [DATA.md](../../DATA.md).
