# Parte 0 — Tres fallas que no fallan

Todo corre sobre `ejemplos/` con `EMBEDDING_BACKEND=local QDRANT_URL=":memory:"`, es decir,
con el MiniLM multilingüe (`embed_notebook_s2`, tope de 128 tokens, verificado 2026-08-27),
sin clave, sin VPN y sin Docker. No se modificó nada del andamiaje.

Para reproducir:

```bash
EMBEDDING_BACKEND=local QDRANT_URL=":memory:" CORPUS_DIR=ejemplos \
  python rag_pipeline.py "¿cuántos días de vacaciones puedo transferir?"   # 0.a
EMBEDDING_BACKEND=local QDRANT_URL=":memory:" python parte0.py             # 0.a (contrafactual), 0.b, 0.c
```

Las salidas crudas están en `salidas/parte0*.txt`.

## 0.a — El PDF que se indexa vacío

Salida cruda de la ingesta (`salidas/parte0a_ingesta.txt`, sin el prompt que imprime el modo
inspección):

```
AVISO ingesta: instructivo_escaneado.pdf: 0 caracteres útiles en 2 página(s). ¿Escaneado o fotografiado? NO se indexa: sin texto no hay nada que recuperar (OCR aparte).
embeddings: local · sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 · tope 128 tokens · Qdrant :memory:
ingesta: nimbus_gastos.md: 1 página(s), 294 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_remoto.md: 1 página(s), 296 caracteres útiles → 1 fragmento(s)
ingesta: nimbus_vacaciones.md: 1 página(s), 363 caracteres útiles → 1 fragmento(s)
indexados 3 fragmentos de 3 documento(s)
```

Lo que habría pasado sin la comprobación (`salidas/parte0a_sin_comprobacion.txt`):

```
documento: instructivo_escaneado.pdf · 2 página(s) · useful_chars=0 · parece_escaneado=True
texto extraído por pypdf (repr): '\n[page=1]\n\n\n[page=2]\n'
fragmentos que produciría la fragmentación sin la comprobación: 1
  [0] '[page=1] [page=2]' · caracteres útiles=0 (el filtro por fragmento exige 20)
```

`load_corpus` cuenta los caracteres alfanuméricos de cada documento sin contar los marcadores
`[page=n]` que la propia ingesta añade, y si quedan menos de 200 (`MIN_CARACTERES_UTILES`)
avisa y salta el documento: el instructivo escaneado tiene **0**. Sin esa comprobación, pypdf
devuelve un texto vacío por página sin lanzar ninguna excepción, y la fragmentación convierte
el documento en un único fragmento `'[page=1] [page=2]'` que no tiene ni una palabra del
instructivo, como muestra el contrafactual de abajo.

Hay un detalle que se vio al correr el contrafactual: el andamiaje trae una segunda red,
`MIN_CARACTERES_FRAGMENTO = 20`, que descarta fragmentos con menos de 20 caracteres útiles.
Ese filtro también habría botado el `'[page=1] [page=2]'`, pero en silencio: el documento
desaparecería sin aviso. La diferencia con la comprobación por documento es justamente el
aviso. Con un PDF que tuviera, por ejemplo, una portada con texto y el resto escaneado, ninguno
de los dos filtros saltaría, y el documento quedaría indexado a medias.

**El corpus real no trae ningún documento así.** `salidas/corpus_verificacion.txt` (rama
`step-1-corpus`) muestra 16 documentos, todos indexables, y 0 páginas con menos de 50
caracteres útiles.

## 0.b — El fragmento que se corta a 128

Salida cruda (`salidas/parte0b_truncado.txt`):

```
aviso de la ingesta:
  AVISO ingesta: chunk_tokens=900 supera el tope del modelo (128 tokens): lo que pase de 126 tokens no llega al índice —el MiniLM lo trunca en silencio; la H200 rechaza la petición—.

texto: salidas/parte0b_texto.txt (900 palabras)
tokens del texto (tokenizador del modelo, sin especiales): 1463
tope del modelo (max_seq_length): 128
fracción que llega al vector: 8.6% (126 tokens de contenido + 2 especiales)
coseno(vector del texto entero, vector del texto recortado a 128): 1.000000
```

El texto son las primeras 900 palabras que pypdf extrae de Vaswani et al. (2017). Esas 900
palabras son **1463 tokens** para el tokenizador del modelo, y el tope es **128**. El coseno
entre el vector del texto entero y el del texto recortado es **1.000000**, o sea, son el mismo
vector: el 91,4 % del texto no influye en nada de lo que se indexa. Un matiz: el tokenizador
sí imprimió un aviso (`1463 > 128`) cuando se contaron los tokens, pero `encode` no dice nada.
Quien solo llama a `encode`, que es lo que hace el pipeline, no se entera.

**`chunk_tokens` para la Parte 1: 512, con `overlap_tokens` de 102 (un quinto).** Que bge-m3
admita 8192 tokens solo dice que un fragmento de 8192 no se trunca; no dice que sea útil. Un
fragmento así equivaldría a varias páginas de un paper mezclando secciones, y su vector sería
un promedio de temas distintos que no se parecería mucho a ninguna pregunta concreta. Con la
proporción que salió aquí, de 1463/900 ≈ 1,6 tokens por palabra (el tokenizador del MiniLM y el
de bge-m3 vienen ambos de XLM-RoBERTa, así que la proporción debería ser parecida, aunque no se
verificó), 512 tokens son unas 315 palabras: más o menos dos o tres párrafos de un paper, que es
la escala a la que suele estar la respuesta de una pregunta simple. Este valor es provisional;
en la Parte 1 se revisa contra la distribución real de largos de párrafo del corpus.

## 0.c — El índice que no se queja

Salida cruda (`salidas/parte0c_indice_no_se_queja.txt`):

```
pregunta: ¿cuál es la política de mascotas?
  1. 0.058  nimbus_gastos.md / nimbus_gastos-0000  # Política de reembolso de gastos de NimbusSoft Los gastos de viaje se…
  2. 0.032  nimbus_remoto.md / nimbus_remoto-0000  # Política de trabajo remoto de NimbusSoft El trabajo remoto está perm…
  3. -0.033  nimbus_vacaciones.md / nimbus_vacaciones-0000  # Política de vacaciones de NimbusSoft Los empleados a tiempo completo…

pregunta: ¿qué exige el instructivo de prácticas de campo sobre el seguro de accidentes?
  1. 0.229  nimbus_remoto.md / nimbus_remoto-0000  # Política de trabajo remoto de NimbusSoft El trabajo remoto está perm…
  2. 0.171  nimbus_gastos.md / nimbus_gastos-0000  # Política de reembolso de gastos de NimbusSoft Los gastos de viaje se…
  3. 0.099  nimbus_vacaciones.md / nimbus_vacaciones-0000  # Política de vacaciones de NimbusSoft Los empleados a tiempo completo…
```

Ninguna de las dos preguntas tiene respuesta en el índice, y el índice devuelve igual tres
vecinos con puntaje, sin error ni aviso. El Hit Rate no puede detectar esto porque solo mide si
el documento fuente aparece entre los k recuperados. En la primera pregunta no hay documento
fuente; en la segunda, el documento fuente nunca entró al índice (0.a). En ninguno de los dos
casos hay un «acierto» que pueda fallar. Por eso el golden set lleva preguntas negativas y se
mide aparte si el sistema se abstiene.

Hay que ser honesto con lo que se ve aquí: los puntajes no son de «aspecto sano» en el sentido
literal. Para la pregunta de mascotas el mejor es 0,058, y un umbral sobre el puntaje podría
filtrarla. La segunda pregunta ya es más delicada: 0,229 es el puntaje más alto de esta corrida,
y justamente es la pregunta cuya respuesta existe en un documento que no se indexó. Con un
corpus de 3 fragmentos cortos cualquier umbral sería arbitrario, y el pipeline no tiene
ninguno: pase lo que pase, entrega los k vecinos. En el corpus real conviene revisar si los
puntajes de las negativas se separan o no de los de las respondibles; esa columna está en
`resultados.csv` (`score_top1`).
