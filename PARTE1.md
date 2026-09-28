# Parte 1 — Baseline RAG

Corpus: los 16 papers seminales de las semanas 1 y 2 (Opción B), **en inglés**, 440 páginas,
todos con capa de texto (`salidas/corpus_verificacion.txt`).

Embeddings: **bge-m3**, fila `embed_local_multilingue`, servido por un **Ollama local** y no por
el de la H200, porque no se tenía la VPN. Después se comprobó que la H200 da los mismos
resultados (el detalle está en 1.3). Generación: **`qwen3:1.7b`**
por Ollama, fila `open_weight_pequeno`. Todo corre a costo cero. La configuración completa y la
comparación con otras rutas están en `EXPERIMENTOS.md`.

Para reproducir la Parte 1:

```bash
export H200_EMBED_URL=http://localhost:11434 EMBEDDING_BACKEND=h200 OPENAI_API_KEY= \
       GENERATION_MODEL=qwen3:1.7b QDRANT_COLLECTION=taller2_bge_m3_local
python inspeccion_ingesta.py 512 102   # 1.1 y 1.2: solo tokeniza
python demo_top5.py                    # 1.3 a 1.5: indexa, top-5 y generación
```

Las salidas crudas están en `salidas/bge-m3@ollama-local/`.

## 1.1 Ingesta: qué se pierde

La ingesta es la de `ingestion.py` sin cambios: `pypdf` página por página, con el marcador
`[page=n]`. **No rechazó ningún documento:** los 16 superan el umbral de
`MIN_CARACTERES_UTILES` y ninguno tiene páginas vacías. Salida de `load_corpus`
(`salidas/bge-m3@ollama-local/parte1_ingesta.txt`):

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
  el tokenizador de los embeddings no da lo mismo: `collection` es un token y `col- lection` son
  cuatro. Además, una
  `fragmento_esperado` del golden set que cruce una palabra cortada **nunca va a coincidir**,
  porque el evaluador normaliza tildes y puntuación, pero no une palabras. Las frases del golden
  set se eligen evitándolas.
- **Encabezados y pies repetidos.** Solo aparecen en tres documentos: Blei et al.
  (`LATENT DIRICHLET ALLOCATION`, `BLEI ,N G, AND JORDAN`), RAG-Anything y la plantilla de prompt
  de RAGAS. Son pocos tokens, pero se repiten en muchos fragmentos del mismo documento.
- **Ligaduras** (`ﬂexibility`, hasta 444 en Brown et al., y en 9 de los 16 documentos). Con
  bge-m3 no pesan: se comprobó que su tokenizador las normaliza, y `ﬂexibility` y `flexibility`
  dan los mismos tokens. Aquí el modelo de embeddings sí importa, porque `cl100k_base`, el
  tokenizador de `text-embedding-3-small`, no normaliza Unicode: `flexibility` son 2 tokens y
  `ﬂexibility` son 5, tres de ellos bytes sueltos de la ligadura. Con la ruta de OpenAI esta
  degradación existiría. También pesaría en un BM25 que tokenice por espacios (Parte 3,
  Opción B). El evaluador no se ve afectado, porque `_normalizar_texto` aplica NFKD y convierte
  `ﬂ` en `fl`.

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
61). Al re-tokenizar el texto decodificado, alguno llega a 513: la decodificación y
re-codificación no es exacta en los bordes, y un token de diferencia contra un tope de 8192 no
importa.

La justificación es contra las preguntas, no contra el tope. Sobre este corpus, bge-m3 da una
mediana de **1,74 tokens por palabra** y **854 tokens por página** (p25 635, p75 1071). Así que
un fragmento de 512 tokens son **unas 293 palabras, un 60 % de página**: dos o tres párrafos de
un paper. Las preguntas simples del golden set van a apuntar a una afirmación concreta (una
cifra de un experimento, la definición de un método), que en un paper suele caber en un párrafo.
Con 512 tokens, ese párrafo entra completo junto con algo de su contexto, y el vector del
fragmento sigue siendo sobre un solo tema.

Que bge-m3 admita 8192 tokens no cambia esto. Un fragmento de 8192 serían unas 10 páginas de
paper, la mitad de un documento típico de este corpus, y su vector promediaría la introducción,
el método y los resultados. Una pregunta específica se parecería poco a ese promedio. En el
otro extremo, un fragmento de 128 como el del MiniLM de la Parte 0 serían unas 74 palabras, y la
respuesta a una pregunta multi-fragmento quedaría repartida entre muchos pedazos que el top-5 no
alcanza a juntar.

El solapamiento es un quinto del fragmento, el valor por defecto del Lab-02: 102 tokens, unas 59
palabras. Su función es que una idea que cae justo en el corte aparezca entera en alguno de los
dos fragmentos vecinos. En 1.4 se ve funcionando: la frase «Training took 3.5 days on 8 P100
GPUs» está completa en dos fragmentos seguidos de Vaswani et al. (`-0013` y `-0014`). Con un
paso de 410 tokens, el solapamiento cuesta un 25 % más de fragmentos que sin él, algo que no
importa a costo cero.

Este valor es el del baseline y no se ajusta mirando el golden set. Si la Parte 2 muestra fallas
de fragmentación (la respuesta cortada entre dos fragmentos, o fragmentos que mezclan temas), la
Opción A de la Parte 3 es donde se prueba otra estrategia.

## 1.3 Embedding e indexación

**Fila de embeddings: `embed_local_multilingue`**. Es BAAI/bge-m3: 1024 dimensiones, tope de
8192 tokens, multilingüe, USD 0 (costo cero real), **verificada el 2026-08-27** en la tabla
semestral.

El pipeline es el del andamiaje sin cambios de código (`EMBEDDING_BACKEND=h200`,
`CodificadorH200`). Lo único distinto es `H200_EMBED_URL=http://localhost:11434`: el cliente
habla con un Ollama, y en vez del de la H200 se usó uno local con `bge-m3:latest`. Según
`ollama show`, ese modelo tiene arquitectura BERT, 566,70 M de parámetros, embeddings de 1024,
contexto de 8192 y pesos en F16, lo que coincide con bge-m3. Se mantiene `truncate: false`, así
que un texto que no quepa da error en vez de recortarse. La ingesta fragmenta con el tokenizador
de Hugging Face `BAAI/bge-m3`, el mismo que el pipeline usa contra la H200.

Hay que ser claro en que **no es la H200**. El modelo es el mismo, pero cuando se corrió el
baseline no se había verificado que el servidor de la H200 sirviera la misma versión ni la misma
cuantización. Por eso la etiqueta de estas corridas es `bge-m3@ollama-local` y no `bge-m3@h200`.

**Después se verificó contra la H200** (2026-09-28, ya con VPN). Se repitieron el índice, la
evaluación 2.b y el híbrido de la Parte 3 con embeddings de la H200, en su propia colección
(`taller2_bge_m3_h200`) y con el mismo generador local, para que lo único que cambiara fuera el
servidor de embeddings. `comparar_configuraciones.py` deja el resultado en
`salidas/comparacion_configuraciones.md`. Los dos Ollama sirven `bge-m3:latest` con el mismo
digest (`7907646426070047…`, F16, 566,70 M de parámetros), o sea los mismos pesos byte a byte.
Con 40 fragmentos del índice y las 10 preguntas, el coseno entre el vector local y el de la H200
nunca baja de 0,999730. Hit Rate, MRR y las dos tasas de abstención salen idénticos, denso e
híbrido, con k = 3 y k = 5. Con k = 3 el top-k coincide en las 10 preguntas, en el mismo orden.
Con k = 5 hay una sola diferencia: en la pregunta 8 los cuatro primeros fragmentos son los
mismos, pero el quinto puesto lo ocupa `ouyang-2022-instructgpt-0102` en local (0,5357) y
`ouyang-2022-instructgpt-0084` en la H200 (0,5335). Es un empate casi exacto en el corte y no
mueve ninguna métrica. Entonces las cifras de este informe, medidas en local, valen también para
la H200, y no hizo falta rehacerlas. Antes se intentó la
ruta de OpenAI (`embed_api_economico`), pero la clave fue rechazada (401) y no se indexó nada con
ella. Un índice construido con un modelo solo se consulta con ese mismo modelo, así que cada
configuración tiene su propia colección de Qdrant.

Indexación en el **Qdrant del contenedor** (`http://localhost:6333`, Qdrant 1.19.1, colección
`taller2_bge_m3_local`), no en `:memory:`: **948 puntos, dimensión 1024, distancia coseno**,
479 701 tokens vectorizados en **136,6 s** en la Mac.

## 1.4 Recuperación: top-5 para tres preguntas de prueba

Tres preguntas en inglés, como el corpus, que no son del golden set: sirven para ver que el
pipeline funciona, no para medirlo. Salida cruda (`salidas/bge-m3@ollama-local/parte1_top5.txt`):

```
embeddings: h200 · BAAI/bge-m3 · tope 8192 · Qdrant http://localhost:6333 · colección taller2_bge_m3_local
indexando 948 fragmentos (512/102) · 479701 tokens · costo estimado USD 0.0000
indexados: 948 puntos · dimensión 1024 · distancia Cosine · 136.6 s

[1] How long did it take to train the big Transformer model, and on what hardware?
  1. 0.630  vaswani-2017-attention-is-all-you-need.pdf / vaswani-2017-attention-is-all-you-need-0014  previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new…
  2. 0.569  vaswani-2017-attention-is-all-you-need.pdf / vaswani-2017-attention-is-all-you-need-0013  s than previous state-of-the-art models on the English-to-German and English-to-French new…
  3. 0.567  vaswani-2017-attention-is-all-you-need.pdf / vaswani-2017-attention-is-all-you-need-0012  ching We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 m…
  4. 0.536  vaswani-2017-attention-is-all-you-need.pdf / vaswani-2017-attention-is-all-you-need-0016  too many heads. In Table 3 rows (B), we observe that reducing the attention key size dk hu…
  5. 0.526  vaswani-2017-attention-is-all-you-need.pdf / vaswani-2017-attention-is-all-you-need-0002  they generate a sequence of hidden statesht, as a function of the previous hidden stateht−…
  respuesta [ollama:qwen3:1.7b] abstuvo=False:
    La entrenación del modelo Transformer grande se llevó a cabo en 3.5 días utilizando 8 unidades de procesamiento gráfico P100.

[2] Why does DPO not need to train an explicit reward model?
  1. 0.686  rafailov-2023-dpo.pdf / rafailov-2023-dpo-0003  reward maximization with a KL-divergence constraint) but is simple to implement and straig…
  2. 0.640  rafailov-2023-dpo.pdf / rafailov-2023-dpo-0002  , and then use RL to find a policy that maximizes the learned reward. In contrast, DPO dir…
  3. 0.638  rafailov-2023-dpo.pdf / rafailov-2023-dpo-0012  the DPO method, provide theoretical backing, and relate advantages of DPO to issues with a…
  4. 0.633  rafailov-2023-dpo.pdf / rafailov-2023-dpo-0000  [page=1] Direct Preference Optimization: Your Language Model is Secretly a Reward Model Ra…
  5. 0.627  rafailov-2023-dpo.pdf / rafailov-2023-dpo-0025  about as often as humans agree with each other, suggesting that GPT-4 is a reason- able pr…
  respuesta [ollama:qwen3:1.7b] abstuvo=False:
    DPO no necesita entrenar un modelo de recompensa explícito porque utiliza un cambio de variables para definir la pérdida de preferencias directamente como una función del política, evitando la necesidad de un modelo de recompensa separado. Esto permite optimizar la política directamente mediante un objetivo de entropía binaria simple, sin recurrir a métodos de aprendizaje por reacción (RL) o a un modelo de recompensa explícito.

[3] Which quality aspects of a RAG answer does RAGAS evaluate without human references?
  1. 0.634  es-2023-ragas.pdf / es-2023-ragas-0012  we observed that ChatGPT often struggles with the task of selecting the sen- tences from t…
  2. 0.609  es-2023-ragas.pdf / es-2023-ragas-0000  [page=1] Ragas: Automated Evaluation of Retrieval Augmented Generation Shahul Es†, Jithin …
  3. 0.609  gao-2023-rag-survey.pdf / gao-2023-rag-survey-0035  quiry. 2) Required Abilities: RAG evaluation also encompasses four abilities indicative of…
  4. 0.606  es-2023-ragas.pdf / es-2023-ragas-0005  ailability of one or more reference answers. For instance, BERTScore (Zhang et al., 2020) …
  5. 0.602  gao-2023-rag-survey.pdf / gao-2023-rag-survey-0040  of noise or contradictory information during retrieval can detrimentally affect RAG’s outp…
  respuesta [ollama:qwen3:1.7b] abstuvo=False:
    The quality aspects of a RAG answer that RAGAS evaluates without human references are: **Context Relevance**, **Faithfulness**, and **Answer Relevance**.
```

Lo que se ve:

- **Pregunta 1:** los cinco vecinos son de Vaswani et al. La respuesta («Training took 3.5 days
  on 8 P100 GPUs») está en los fragmentos de las posiciones 1 y 2, y la sección «Hardware and
  Schedule» en la 3.
- **Pregunta 2:** los cinco son de Rafailov et al.
- **Pregunta 3:** mezcla dos documentos. RAGAS ocupa las posiciones 1, 2 y 4, y el survey de Gao
  et al. las 3 y 5. Tiene sentido, porque el survey resume RAGAS en su sección de evaluación.
- **Los puntajes están todos entre 0,52 y 0,69**, con poca separación entre el primero y el
  quinto (0,630 contra 0,526 en la pregunta 1). Es lo que la Parte 0.c ya mostró: el puntaje
  solo ordena, no dice si la respuesta está. Habrá que ver en la Parte 2 si las negativas quedan
  en ese mismo rango.

## 1.5 Generación

Generador: **`qwen3:1.7b` por Ollama local** (fila `open_weight_pequeno`), con temperatura 0,
porque la clave de OpenAI no funcionó. El prompt y la frase de abstención son los del andamiaje
sin tocar. `ABSTENCION` sigue siendo «El corpus no contiene información suficiente.», la misma
en el prompt, en el golden set y en el detector. `OPENAI_API_KEY=` se deja vacía a propósito,
porque `generate` prefiere OpenAI apenas hay una clave, aunque sea inválida.

Las tres respuestas son correctas contra el paper. Hay dos detalles que conviene anotar para la
Parte 2:

- **Idioma.** El modelo responde en español en las preguntas 1 y 2, porque el prompt está en
  español, y en inglés en la 3. No es un error, pero una respuesta esperada en inglés comparada
  por cadena no coincidiría. Además, si el modelo se abstuviera en inglés, el detector (que busca
  la frase en español) no lo vería. Hay que revisarlo en la Parte 2.b.
- **Pequeños errores de traducción** propios de un modelo de 1,7 B. En la pregunta 2 escribe
  «aprendizaje por reacción (RL)» en vez de «aprendizaje por refuerzo». El contenido está bien.
