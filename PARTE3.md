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
