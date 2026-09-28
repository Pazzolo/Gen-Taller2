# Parte 1 — Baseline RAG

Corpus: los 16 papers seminales de las semanas 1 y 2 (Opción B), **en inglés**, 440 páginas,
todos con capa de texto (`salidas/corpus_verificacion.txt`).

Para reproducir las cifras de 1.1 y 1.2 (no necesita VPN, solo descarga el tokenizador de
`BAAI/bge-m3`):

```bash
python inspeccion_ingesta.py 512 102      # escribe salidas/parte1_ingesta.txt
```

## 1.1 Ingesta: qué se pierde

La ingesta es la de `ingestion.py` sin cambios: `pypdf` página por página, con el marcador
`[page=n]`. **No rechazó ningún documento:** los 16 superan el umbral de
`MIN_CARACTERES_UTILES` y ninguno tiene páginas vacías. Salida de `load_corpus`
(`salidas/parte1_ingesta.txt`):

```
ingesta: bai-2022-constitutional-ai.pdf: 34 página(s), 94294 caracteres útiles → 75 fragmento(s)
ingesta: blei-2003-lda.pdf: 30 página(s), 59953 caracteres útiles → 57 fragmento(s)
ingesta: brown-2020-gpt3.pdf: 75 página(s), 186342 caracteres útiles → 156 fragmento(s)
ingesta: doersch-2016-tutorial-vae.pdf: 23 página(s), 36740 caracteres útiles → 33 fragmento(s)
ingesta: edge-2024-graphrag.pdf: 26 página(s), 70385 caracteres útiles → 60 fragmento(s)
ingesta: es-2023-ragas.pdf: 8 página(s), 25583 caracteres útiles → 21 fragmento(s)
ingesta: gao-2023-rag-survey.pdf: 21 página(s), 86641 caracteres útiles → 78 fragmento(s)
ingesta: kingma-2013-vae.pdf: 14 página(s), 31786 caracteres útiles → 31 fragmento(s)
ingesta: lewis-2020-rag.pdf: 19 página(s), 55094 caracteres útiles → 47 fragmento(s)
ingesta: ng-jordan-2001-discriminative-vs-generative.pdf: 8 página(s), 18996 caracteres útiles → 19 fragmento(s)
ingesta: ouyang-2022-instructgpt.pdf: 68 página(s), 143167 caracteres útiles → 118 fragmento(s)
ingesta: rafailov-2023-dpo.pdf: 27 página(s), 71700 caracteres útiles → 66 fragmento(s)
ingesta: rag-anything-2025.pdf: 18 página(s), 57304 caracteres útiles → 41 fragmento(s)
ingesta: reimers-2019-sentence-bert.pdf: 11 página(s), 35133 caracteres útiles → 31 fragmento(s)
ingesta: vaswani-2017-attention-is-all-you-need.pdf: 15 página(s), 31566 caracteres útiles → 25 fragmento(s)
ingesta: wei-2022-chain-of-thought.pdf: 43 página(s), 105503 caracteres útiles → 90 fragmento(s)
```

Que todos tengan texto no quiere decir que el texto esté bien. Revisando muestras del texto
extraído se ven cinco degradaciones, de más a menos grave para este corpus:

- **Tablas.** Quedan como líneas de números sin columnas. En la Tabla 2 de Vaswani et al. la
  fila `ByteNet [18] 23.75` ya no dice a qué columna pertenece el 23.75, porque las celdas vacías
  desaparecen y lo de la derecha se corre. Los exponentes pierden el superíndice: `1.0· 1020` es
  1,0·10²⁰. En Reimers y Gurevych la leyenda de la Tabla 1 aparece pegada al texto de la columna
  vecina, y el contenido de la tabla queda en otra parte. Una pregunta cuya respuesta sea una
  cifra de tabla es candidata a fallar en la generación aunque la recuperación acierte.
- **Ecuaciones.** Se aplanan: `Attention(Q,K,V ) = softmax(QK T\n√dk\n)V`. El texto
  alrededor sobrevive, la fórmula no se puede leer.
- **Palabras cortadas por el guion de fin de línea.** `col-\nlection`, `neu-\ntral`. El conteo
  va de 0 (Ng y Jordan) a 221 (Reimers y Gurevych), más alto en los papers a dos columnas. Para
  bge-m3 no da lo mismo: `collection` es un token y `col- lection` son cuatro. Además, una
  `fragmento_esperado` del golden set que cruce una palabra cortada **nunca va a coincidir**,
  porque el evaluador normaliza tildes y puntuación, pero no une palabras. Las frases del golden
  set se eligen evitándolas.
- **Encabezados y pies repetidos.** Solo aparecen en tres documentos: Blei et al.
  (`LATENT DIRICHLET ALLOCATION`, `BLEI ,N G, AND JORDAN`), RAG-Anything y la plantilla de prompt
  de RAGAS. Son pocos tokens, pero se repiten en muchos fragmentos del mismo documento.
- **Ligaduras** (`ﬂexibility`, hasta 444 en Brown et al.). Se verificó que el tokenizador de
  bge-m3 las normaliza (`ﬂexibility` y `flexibility` dan los mismos tokens), así que para el
  índice denso no pesan. Para un BM25 que tokenice por espacios sí pesarían (Parte 3, Opción B).

Hay una observación que no es una degradación del parseo pero puede serlo de la recuperación:
**el 40,6 % de los tokens del corpus está después del último encabezado «References»**, es
decir, en la bibliografía y en los apéndices. En Wei et al. es el 75,1 %, casi todo apéndice con
ejemplos de prompts; en Ouyang et al. y Rafailov et al., cerca del 60 %. En el baseline no se
quitan: el baseline se mide como viene, y si en la Parte 2 aparecen fragmentos de bibliografía
en el top-k (listas de títulos que comparten palabras con la pregunta), eso va al análisis de
fallos como falla de ingesta.

## 1.2 Fragmentación

**`chunk_tokens = 512`, `overlap_tokens = 102`**, en tokens del tokenizador de `BAAI/bge-m3`,
de tamaño fijo, con la función `fixed_size_chunks_tokens` del andamiaje. Resultado:
**948 fragmentos**, todos de 512 tokens salvo el último de cada documento (el más corto tiene
61).

La justificación es contra las preguntas, no contra el tope. Sobre este corpus, bge-m3 da una
mediana de **1,74 tokens por palabra** y **854 tokens por página** (p25 635, p75 1071). Así que
un fragmento de 512 tokens son **unas 293 palabras, un 60 % de página**: dos o tres párrafos de
un paper. Las preguntas simples del golden set van a apuntar a una afirmación concreta (una
cifra de un experimento, la definición de un método), que en un paper suele caber en un párrafo.
Con 512 tokens, ese párrafo entra completo junto con algo de su contexto, y el vector del
fragmento sigue siendo sobre un solo tema.

Que bge-m3 admita 8192 tokens no cambia esto. Un fragmento de 8192 serían unas 10 páginas de
paper, la mitad de un documento entero de este corpus, y su vector promediaría la introducción,
el método y los resultados. Una pregunta específica se parecería poco a ese promedio. En el
otro extremo, un fragmento de 128 como el del MiniLM de la Parte 0 serían unas 70 palabras, y la
respuesta a una pregunta multi-fragmento quedaría repartida entre muchos pedazos que el top-5 no
alcanza a juntar.

El solapamiento es un quinto del fragmento, el valor por defecto del Lab-02: 102 tokens, unas 59
palabras. Su función es que una idea que cae justo en el corte aparezca entera en alguno de los
dos fragmentos vecinos. Con un paso de 410 tokens, cuesta un 25 % más de fragmentos que sin solapamiento, algo que no
importa con 948 fragmentos y embeddings a costo cero.

Este valor es el del baseline y no se ajusta mirando el golden set. Si la Parte 2 muestra fallas
de fragmentación (la respuesta cortada entre dos fragmentos, o fragmentos que mezclan temas), la
Opción A de la Parte 3 es donde se prueba otra estrategia.

## 1.3 Embedding e indexación

_Pendiente: necesita la VPN GlobalProtect (bge-m3 en la H200)._

## 1.4 Recuperación: top-5 para tres preguntas de prueba

_Pendiente._

## 1.5 Generación

_Pendiente: necesita `.env` con `OPENAI_API_KEY` y `GENERATION_MODEL`._
