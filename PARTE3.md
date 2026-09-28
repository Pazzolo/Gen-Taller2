# Parte 3 — Extensión: búsqueda híbrida (Opción B)

## Hipótesis, declarada antes de implementar

Se elige por lo que midió la Parte 2, no por preferencia. El baseline denso tiene Hit Rate@5 de
1,000 y MRR de 0,781, así que casi no le falta encontrar el documento. Los dos peores casos de la
2.c apuntan a otra cosa: **el recuperador denso ignora los nombres propios de la pregunta**.

- **Pregunta 5** («Lewis et al.'s RAG and the RAGAS framework…»): con k = 3, los tres vecinos son
  de RAGAS y ninguno de Lewis et al. La pregunta nombra los dos trabajos, pero el vector la acerca
  a uno solo.
- **Pregunta 6** («InstructGPT and DPO…»): con k = 5, cuatro vecinos son de DPO y solo uno de
  InstructGPT, en la posición 5, y ese no contiene ninguna palabra clave de la respuesta.

BM25 puntúa por coincidencia exacta de términos, y «RAGAS», «Lewis», «InstructGPT» o «DPO» son
términos raros en el corpus, con IDF alto. **Lo que se espera corregir** es que, al fusionar BM25
con el denso por Reciprocal Rank Fusion, el documento que falta en las preguntas que nombran dos
trabajos entre al top-k.

Cómo se va a medir, con el mismo golden set, el mismo índice (948 fragmentos, 512/102) y el mismo
generador:

1. **Hit Rate@k y MRR** para k = 3 y k = 5, contra el baseline, que es lo que pide el enunciado.
2. **Cobertura de fuentes en las preguntas multi-documento (5 y 6):** qué fracción de sus
   documentos fuente aparece en el top-k. Es la medición directa de la hipótesis. El Hit Rate
   no la ve, porque en esas preguntas el acierto es a nivel de documento y le basta uno de los
   dos (limitación declarada en la 2.a).
3. **Las dos tasas de abstención**, porque en la pregunta 6 el baseline se abstuvo teniendo la
   respuesta en el contexto. Si el contexto híbrido nombra InstructGPT, puede que ya no se
   abstenga.
4. **BM25 solo**, como diagnóstico, para saber qué aporta cada lado de la fusión.

Lo que **no** se espera corregir:

- la pregunta 4 con k = 5, que es una falla de generación (el modelo se distrae con «540B»);
- la pregunta 3, que es una falla de la anotación del golden set.

Si la fusión no mejora nada, o empeora las preguntas simples (BM25 puede meter fragmentos que
repiten palabras de la pregunta sin responderla), eso también es un resultado, y se explica.

Parámetros fijados antes de medir: 20 candidatos de cada recuperador, RRF con la constante
habitual k = 60, y tokenización de BM25 con la misma normalización del evaluador (minúsculas,
NFKD, sin puntuación, lo que además deshace las ligaduras) y sin palabras vacías en inglés.

## Implementación

`hibrido.py`: `PipelineHibrido` hereda de `RagPipeline` y solo reemplaza `retrieve`.

- **BM25** (`rank_bm25.BM25Okapi`) se construye sobre los **mismos 948 fragmentos** de la
  colección de Qdrant: se leen del índice, no se re-fragmenta.
- **La fusión:** el denso y BM25 dan 20 candidatos cada uno, y RRF suma `1/(60 + posición)` de
  cada lista.
- **Lo que no cambia:** el prompt, el generador (`qwen3:1.7b`), el detector de abstención y
  `evaluate_retrieval`, que es la misma función del evaluador. La comparación cambia una sola
  cosa.
- **Costo:** USD 0, todo local. Las dos corridas híbridas (k = 3 y k = 5, con generación)
  tardaron 2 min 40 s.

```bash
export H200_EMBED_URL=http://localhost:11434 EMBEDDING_BACKEND=h200 OPENAI_API_KEY= \
       GENERATION_MODEL=qwen3:1.7b QDRANT_COLLECTION=taller2_bge_m3_local
python hibrido.py --modo bm25 --sin-generar   # diagnóstico
python hibrido.py --modo hibrido
python comparar_extension.py                  # salidas/bge-m3@ollama-local/comparacion_extension.md
```

## Resultados

Desde los CSV crudos (`salidas/bge-m3@ollama-local/{,bm25/,hibrido/}resultados_k*.csv`):

| variante | k | Hit Rate@k | MRR | abstención correcta | abstención indebida | cobertura P5 | cobertura P6 |
|---|---|---|---|---|---|---|---|
| denso (baseline) | 3 | 0.875 | 0.750 | 1.000 | 0.250 | 1/2 | 1/2 |
| denso (baseline) | 5 | 1.000 | 0.781 | 1.000 | 0.125 | 2/2 | 2/2 |
| bm25 solo | 3 | 0.875 | 0.750 | sin medir | sin medir | 1/2 | 2/2 |
| bm25 solo | 5 | 0.875 | 0.750 | sin medir | sin medir | 2/2 | 2/2 |
| **híbrido (RRF)** | 3 | **1.000** | **0.854** | 1.000 | **0.125** | 1/2 | **2/2** |
| **híbrido (RRF)** | 5 | 1.000 | **0.854** | 1.000 | **0.000** | 2/2 | 2/2 |

Posición del primer acierto por pregunta (— sin acierto; A = se abstuvo):

| id | tipo | denso k=3 | denso k=5 | bm25 k=3 | bm25 k=5 | híbrido k=3 | híbrido k=5 |
|---|---|---|---|---|---|---|---|
| 1 | simple | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 | simple | 1 | 1 | 2 | 2 | 1 | 1 |
| 3 | simple | — | 4 | 2 | 2 | 2 | 2 |
| 4 | simple | 2 | 2 | — | — | 3 | 3 |
| 5 | multi-chunk | 1 A | 1 | 1 | 1 | 1 A | 1 |
| 6 | multi-chunk | 1 A | 1 A | 1 | 1 | 1 | 1 |
| 7 | multi-chunk | 1 | 1 | 1 | 1 | 1 | 1 |
| 8 | negativo | A | A | — | — | A | A |
| 9 | negativo | A | A | — | — | A | A |
| 10 | adversarial | 2 | 2 | 1 | 1 | 1 | 1 |

## Qué se confirmó y qué no

**La hipótesis se cumplió a medias: en la pregunta 6 sí, en la 5 no.**

**Pregunta 6, se cumplió.** Con k = 3, la cobertura pasa de 1/2 a 2/2: InstructGPT entra en la
posición 2 (`rafailov, ouyang, rafailov`). Y el efecto llega a la generación: el baseline se
abstenía con los dos k, y el híbrido ya no se abstiene con ninguno. Responde bien: DPO optimiza
la política directamente con una pérdida de entropía cruzada binaria, sin aprendizaje por
refuerzo, e InstructGPT entrena un modelo de recompensa y luego usa PPO. Esto es casi toda la
baja de la abstención indebida (0,250 → 0,125 con k = 3 y 0,125 → 0,000 con k = 5). También
refuerza la lectura de la 2.c: el modelo de 1,7 B se abstenía porque ningún fragmento nombraba
«InstructGPT», no porque faltara la información.

**Pregunta 5, no se cumplió.** Con k = 3 la cobertura sigue en 1/2, los tres vecinos siguen
siendo de RAGAS y el sistema se sigue absteniendo. Esta vez BM25 solo tampoco trae a Lewis et
al. La razón está en el corpus, y es lo más interesante de esta parte. Se contó en cuántos
fragmentos aparece cada término de la pregunta:

| término | fragmentos | de qué documentos |
|---|---|---|
| `dpo` | 37 | los 37 de Rafailov et al. |
| `instructgpt` | 54 | 43 de Ouyang et al., 10 de Bai et al., 1 de Wei et al. |
| `ragas` | 12 | 5 de RAGAS, 4 del survey de Gao et al., 3 de Edge et al. |
| `lewis` | 29 | **solo 6 de Lewis et al.**; 9 de Gao, 4 de Edge, 4 de RAGAS, 3 de Brown… |

«DPO» e «InstructGPT» son nombres de **métodos**, y el paper que los propone los repite en el
cuerpo del texto. «Lewis» es el nombre de un **autor**. En su propio paper aparece en el fragmento
0 (la línea de autores) y en cinco fragmentos de la segunda mitad (del 0026 al 0036, de 47), que
por su posición parecen ser la bibliografía; no se revisó uno por uno. Donde más aparece es en las
citas de *otros* papers («Lewis et al., 2020»). Para BM25, «lewis» apunta a los trabajos que citan el RAG, no al RAG. Así que la
hipótesis era correcta para preguntas que nombran métodos y falsa para las que nombran autores.
Eso no se sabía antes de medir.

**De dónde sale la subida del MRR (0,750 → 0,854 con k = 3).** No de las preguntas de la
hipótesis, que ya tenían su primer acierto en la posición 1, sino de dos simples:

- **Pregunta 3:** de «—» a 2. Hay que leerla con cuidado. La pregunta comparte «community» y
  «detection» con la frase anotada (el solapamiento de 2 de 5 medido en la 2.a), y BM25 premia
  justo eso. Parte de la mejora es coincidencia de palabras con la anotación, y la respuesta ya
  era correcta con el baseline (2.c).
- **Pregunta 10** (adversarial): de 2 a 1, porque «dropout rate» y «base» coinciden literalmente
  con el fragmento de Vaswani et al.

**Lo que empeoró.** La pregunta 4 baja de la posición 2 a la 3. BM25 solo no la encuentra nunca:
«chain-of-thought» está en casi todos los fragmentos de Wei et al., así que no discrimina, y la
frase con la respuesta dice «∼100B parameters», no «model size». En la fusión, ese fragmento pierde
el aporte del lado léxico. Como se había previsto, la falla de generación de la pregunta 4 con
k = 5 sigue igual: el híbrido también responde «540B».

**Las negativas no cambian:** abstención correcta de 1,000 en las dos variantes. BM25 no las
empeora, aunque «GPT-4» y «Llama» sí aparecen en el corpus.

## Lectura final

Con 8 respondibles, cada pregunta mueve el Hit Rate en 0,125, así que estas diferencias son de
una o dos preguntas y no alcanzan para decir que el híbrido es mejor en general. Lo que sí se
puede decir, con evidencia, es que el híbrido **arregla la pregunta que nombra métodos (6)**,
**no arregla la que nombra autores (5)**, y que **la subida del MRR viene sobre todo de la
coincidencia léxica en dos preguntas simples**, una de ellas inflada por la propia anotación.
Para la pregunta 5 haría falta otra cosa, por ejemplo reescribir la consulta para separar las
dos partes («RAG model of Lewis et al.» y «RAGAS framework») y recuperar para cada una. Eso ya
no es la Opción B.
