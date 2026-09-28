# Parte 4 — Reflexión

Las cifras salen de los CSV crudos de la Parte 2.b (`bge-m3@ollama-local`) con
`parte4_cifras.py`, que también corre la pregunta global de la 4.2 por el baseline. La salida
cruda está en `salidas/bge-m3@ollama-local/parte4_cifras.txt`.

## 4.1 El Hit Rate de 0,70

Con la plantilla anterior, el Hit Rate dividía los aciertos entre las 10 preguntas, incluidas las negativas, que no pueden acertar porque no tienen documento fuente. Con mis resultados da **0,700 con k = 3** (7 de 10) y **0,800 con k = 5** (8 de 10), y 0,800 ya es el techo aunque el sistema acertó las 8 respondibles. Ese número mezclaba cuánto recupera el sistema con cuántas negativas trae el golden set. Sin ellas en el denominador queda solo la recuperación: **0,875 y 1,000**, y el MRR con k = 5 pasa de 0,625 a **0,781**. Mi 0,700 coincide con la cifra del enunciado por otra razón: aquí sí hay un fallo real. Para las negativas lo que importa es si el sistema se abstuvo, **2 de 2**, y eso se lee junto con la abstención indebida (0,125), porque abstenerse siempre también da 2 de 2.

| k | Hit Rate antes (÷ 10) | Hit Rate ahora (÷ 8) | MRR antes | MRR ahora | abstención correcta |
|---|---|---|---|---|---|
| 3 | 0.700 | 0.875 | 0.600 | 0.750 | 2 de 2 |
| 5 | 0.800 | 1.000 | 0.625 | 0.781 | 2 de 2 |

## 4.2 Una pregunta global

Pregunta global: *«What are the main themes that run across all the papers in this corpus, and how do they relate to each other?»*. Responderla bien exige leer los 16 papers. El RAG vectorial solo trae los 5 fragmentos más parecidos a la pregunta: 2560 tokens de 387 086, el **0,66 %** del corpus. Corrida en el baseline, 4 de los 5 vecinos fueron del paper de LDA, que habla literalmente de «topics» dentro de un «corpus», y el sistema se abstuvo. Recuperó lo que se parece a la pregunta, no lo que la responde. GraphRAG (Edge et al., 2024) no depende de esa búsqueda. Al indexar, extrae un grafo de entidades, lo divide en comunidades con Leiden y resume cada comunidad. En la consulta trabaja en map-reduce: cada resumen da una respuesta parcial y luego se combinan. Así la respuesta cubre el corpus entero, no cinco fragmentos.

Lo que devolvió el baseline:

```
1. 0.482  blei-2003-lda-0001
2. 0.471  blei-2003-lda-0031
3. 0.469  blei-2003-lda-0037
4. 0.447  brown-2020-gpt3-0153
5. 0.446  blei-2003-lda-0005
documentos distintos en el top-5: 2 de 16
respuesta: El corpus no contiene información suficiente.
```
