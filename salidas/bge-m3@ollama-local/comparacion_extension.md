| variante | k | Hit Rate@k | MRR | abstención correcta | abstención indebida | cobertura P5 | cobertura P6 |
|---|---|---|---|---|---|---|---|
| denso (baseline) | 3 | 0.875 | 0.750 | 1.000 | 0.250 | 1/2 | 1/2 |
| denso (baseline) | 5 | 1.000 | 0.781 | 1.000 | 0.125 | 2/2 | 2/2 |
| bm25 solo | 3 | 0.875 | 0.750 | sin medir | sin medir | 1/2 | 2/2 |
| bm25 solo | 5 | 0.875 | 0.750 | sin medir | sin medir | 2/2 | 2/2 |
| híbrido (RRF) | 3 | 1.000 | 0.854 | 1.000 | 0.125 | 1/2 | 2/2 |
| híbrido (RRF) | 5 | 1.000 | 0.854 | 1.000 | 0.000 | 2/2 | 2/2 |

Posición del primer acierto (— sin acierto, n/a negativa) · A = se abstuvo

| id | tipo | denso (baseline) k=3 | denso (baseline) k=5 | bm25 solo k=3 | bm25 solo k=5 | híbrido (RRF) k=3 | híbrido (RRF) k=5 |
|---|---|---|---|---|---|---|---|
| 1 | simple | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 | simple | 1 | 1 | 2 | 2 | 1 | 1 |
| 3 | simple | — | 4 | 2 | 2 | 2 | 2 |
| 4 | simple | 2 | 2 | — | — | 3 | 3 |
| 5 | multi-chunk | 1 A | 1 | 1 | 1 | 1 A | 1 |
| 6 | multi-chunk | 1 A | 1 A | 1 | 1 | 1 | 1 |
| 7 | multi-chunk | 1 | 1 | 1 | 1 | 1 | 1 |
| 8 | negativo | n/a A | n/a A | n/a | n/a | n/a A | n/a A |
| 9 | negativo | n/a A | n/a A | n/a | n/a | n/a A | n/a A |
| 10 | adversarial | 2 | 2 | 1 | 1 | 1 | 1 |
