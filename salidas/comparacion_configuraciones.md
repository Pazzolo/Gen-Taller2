# Comparación de configuraciones: bge-m3 local contra H200

## 1. Identidad del modelo

| servidor | nombre | digest | tamaño | cuantización | parámetros |
|---|---|---|---|---|---|
| ollama-local | `bge-m3:latest` | `7907646426070047…` | 1.16 GB | F16 | 566.70M |
| h200 | `bge-m3:latest` | `7907646426070047…` | 1.16 GB | F16 | 566.70M |

**Digest idéntico** en los dos servidores.

## 2. Vectores: mismo texto, dos servidores

40 fragmentos del índice (uno de cada 23) y las 10 preguntas del golden set, con la misma petición que usa el pipeline.

| textos | n | coseno mínimo | coseno medio | coseno máximo |
|---|---|---|---|---|
| fragmentos | 40 | 0.999866 | 0.999983 | 0.999994 |
| preguntas | 10 | 0.999730 | 0.999918 | 1.000000 |

## 3. Recuperación y generación

| configuración | variante | k | Hit Rate@k | MRR | abstención correcta | abstención indebida |
|---|---|---|---|---|---|---|
| bge-m3@h200 | denso | 3 | 0.875 | 0.750 | 1.000 | 0.250 |
| bge-m3@h200 | denso | 5 | 1.000 | 0.781 | 1.000 | 0.125 |
| bge-m3@ollama-local | denso | 3 | 0.875 | 0.750 | 1.000 | 0.250 |
| bge-m3@ollama-local | denso | 5 | 1.000 | 0.781 | 1.000 | 0.125 |
| bge-m3@h200 | hibrido | 3 | 1.000 | 0.854 | 1.000 | 0.125 |
| bge-m3@h200 | hibrido | 5 | 1.000 | 0.854 | 1.000 | 0.000 |
| bge-m3@ollama-local | hibrido | 3 | 1.000 | 0.854 | 1.000 | 0.125 |
| bge-m3@ollama-local | hibrido | 5 | 1.000 | 0.854 | 1.000 | 0.000 |

**k=3, local contra H200:** mismo top-k en el mismo orden en 10 de 10 preguntas, mismo conjunto en 10 de 10; diferencia máxima de puntaje en los top-k iguales: 0.0028.

- k=5, pregunta 8: top-k distinto

**k=5, local contra H200:** mismo top-k en el mismo orden en 9 de 10 preguntas, mismo conjunto en 9 de 10; diferencia máxima de puntaje en los top-k iguales: 0.0010.

