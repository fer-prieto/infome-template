# AGENTS.md

## Qué es este repositorio

Informe de la Unidad 3 del curso **Arquitecturas de Alto Rendimiento (IF557ZIN)**
construido sobre la plantilla LaTeX `Template-Informe` (`templates/`, `src/`).

- `main.tex`: metadatos del documento (título, autor, curso, portada) y orden de ensamblado.
- `src/main/sections/`: contenido real del informe, un archivo por sección.
- `templates/`, `src/style/`, `src/page/`, `src/env/`, `src/cfg/`: infraestructura de la
  plantilla. **No es contenido del informe.**

## Regla principal para agentes LLM

No hace falta leer la plantilla ni los `.tex` para conocer el contenido del informe.
La fuente de contenido lista para consumir es el Markdown de la entrega:

```text
output_docs/Alto_Rendimiento-3da_entrega-David_Prieto.md
```

- Es texto plano, con encabezados, tablas y estilo ya normalizados: se puede leer
  completo, citar o buscar sin cargar los 300 KB de la plantilla.
- Refleja el PDF entregado, que es la versión "oficial" del documento.
- Úsalo como contenido de referencia para responder preguntas, resumir, verificar
  cifras o redactar texto nuevo.

Si la pregunta es sobre **cómo se compila o cómo está armada la plantilla**, ahí sí
hay que mirar `main.tex`, `src/main/template.tex` o `templates/`.

## Estructura de `output_docs/`

Cada entrega vive en su propia carpeta y contiene el PDF y su Markdown:

```text
output_docs/
	├── Alto_Rendimiento-3da_entrega-David_Prieto.pdf
	└── Alto_Rendimiento-3da_entrega-David_Prieto.md
```

El `.md` se genera con la skill `pdfmd` (nunca se edita el PDF):

```console
$ pdfmd Alto_Rendimiento-3da_entrega-David_Prieto.pdf \
    -o Alto_Rendimiento-3da_entrega-David_Prieto.md
$ mdlint Alto_Rendimiento-3da_entrega-David_Prieto.md
```

Si se actualiza el PDF, se regenera el `.md` y se verifican ambos en el mismo commit.

### Limitaciones del Markdown

- No incluye imágenes, fórmulas renderizadas ni OCR; el texto matemático queda como
  texto corrido.
- El índice y el índice de tablas salen como tablas partidas por los puntos guía del
  PDF. Es una limitación conocida: no se corrigen a mano ni se inventan celdas.
- Ante una duda de contenido, manda el PDF; el `.md` es el atajo de lectura.

## Compilación

```console
$ make            # compila main.tex
$ make otro.tex   # compila otro archivo
```

## Commits

Convención Conventional Commits en español e impersonal, según la skill
`commit-personal` (tipo `docs:` para documentación y entregas, sin `refactor`,
máximo 72 caracteres, sin firmas de IA).
