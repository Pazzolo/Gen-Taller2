# Parte 2 — Evaluación con golden set

Configuración: la del baseline de la Parte 1, `bge-m3@ollama-local` (fila
`embed_local_multilingue`, verificada 2026-08-27), 512/102, 948 fragmentos, generación con
`qwen3:1.7b` (`open_weight_pequeno`) por Ollama. Para reproducir:

```bash
export H200_EMBED_URL=http://localhost:11434 EMBEDDING_BACKEND=h200 OPENAI_API_KEY= \
       GENERATION_MODEL=qwen3:1.7b QDRANT_COLLECTION=taller2_bge_m3_local
python verificar_golden.py                    # 2.a: el golden set contra los fragmentos reales
python evaluation.py --k 3 --sin-indexar      # 2.b (reusa el índice de la Parte 1)
python evaluation.py --k 5 --sin-indexar
python tabla_metricas.py --entregable         # tabla desde los CSV + ./resultados.csv
python analisis_fallos.py                     # 2.c: evidencia de los peores casos
```

Cambio declarado sobre el andamiaje: `evaluation.py` ahora escribe un CSV por configuración y
por k (antes la corrida con k = 5 pisaba la de k = 3) y guarda la pregunta, la respuesta entera
y los puntajes de todos los recuperados. Las métricas y sus definiciones no se tocaron.

## 2.a El golden set

`golden_set.json`, 10 preguntas en inglés, como el corpus:

| id | tipo | documento(s) fuente | qué prueba |
|---|---|---|---|
| 1 | simple | Reimers y Gurevych | una cifra concreta (65 horas) |
| 2 | simple | Kingma y Welling | un concepto (la reparametrización) |
| 3 | simple | Edge et al. | un nombre propio (Leiden) |
| 4 | simple | Wei et al. | una cifra con un distractor cerca (100B contra 540B) |
| 5 | multi-fragmento | Lewis et al. + RAGAS | combinar **dos documentos** |
| 6 | multi-fragmento | Ouyang et al. + Rafailov et al. | combinar **dos documentos** |
| 7 | multi-fragmento | Edge et al. | combinar **dos secciones** del mismo documento |
| 8 | negativa | — | tasa de aprendizaje de Llama 2 |
| 9 | negativa | — | número de parámetros de GPT-4 |
| 10 | adversarial | Vaswani et al. | una instrucción de ignorar el contexto |

Algunas decisiones de diseño:

- **Anotación por documento y frase literal**, no por `chunk_id`, para que la Opción A de la
  Parte 3 pueda re-fragmentar sin invalidar nada. `verificar_golden.py` comprueba contra los
  948 fragmentos reales que cada frase cabe entera en al menos un fragmento de su documento
  (entre 1 y 4, según la frase), y que ninguna cruza una palabra cortada por guion. Esto último
  hizo falta de verdad: la frase natural de la pregunta 1 («most similar pair») aparece en el
  PDF como `sim- ilar` y no habría coincidido nunca.
- **Preguntas parafraseadas.** El mismo script mide cuántas palabras de contenido comparte cada
  pregunta con su frase esperada: 0 de 2, 1 de 5, 2 de 5, 1 de 7 y 2 de 4. La recuperación no
  puede ganar por copiar palabras.
- **Negativas cercanas, no absurdas.** Una pregunta sobre recetas se abstendría sola. Las dos
  negativas nombran modelos que **sí aparecen** en el corpus: Llama 2 en la bibliografía de
  Edge et al., y GPT-4 en 49 fragmentos (como juez en Rafailov et al. y en bibliografías). Pero
  ningún fragmento contiene la respuesta. Se comprobó buscando co-apariciones: «gpt 4» con
  «trillion» solo aparece en una bibliografía, y «llama» con «learning rate» en ninguna parte.
  La 9, además, tienta al generador a responder con lo que «sabe».
- **La adversarial es respondible:** tiene documento fuente y cuenta en el Hit Rate y el MRR. Se
  decidió así porque lo que interesa es que el sistema **no obedezca** la instrucción metida en la
  pregunta y responda desde el contexto. Si se hubiera dejado sin fuente, se estaría premiando
  que se abstenga ante una pregunta que sí tiene respuesta.
- **Limitación que se asume:** en las multi-fragmento no se pone frase, así que el acierto es a
  nivel de documento, y basta un fragmento de *uno* de los dos documentos. Eso es generoso, y en
  la 2.c se ve el costo: la pregunta 5 cuenta como acierto con k = 3 aunque no se recuperó nada
  de Lewis et al.

## 2.b Métricas

Tabla reconstruida desde los CSV crudos por `tabla_metricas.py`, que coincide con lo que reportó
`evaluation.py` (`salidas/bge-m3@ollama-local/tabla_metricas.md`):

| k | preguntas | respondibles | negativas | Hit Rate@k | MRR | abstención correcta | abstención indebida |
|---|---|---|---|---|---|---|---|
| 3 | 10 | 8 | 2 | **0.875** | **0.750** | **1.000** | **0.250** |
| 5 | 10 | 8 | 2 | **1.000** | **0.781** | **1.000** | **0.125** |

Por tipo:

| k | tipo | n | Hit Rate | MRR | se abstuvo |
|---|---|---|---|---|---|
| 3 | simple | 4 | 0.750 | 0.625 | 0.000 |
| 3 | multi-chunk | 3 | 1.000 | 1.000 | 0.667 |
| 3 | adversarial | 1 | 1.000 | 0.500 | 0.000 |
| 3 | negativo | 2 | — | — | 1.000 |
| 5 | simple | 4 | 1.000 | 0.688 | 0.000 |
| 5 | multi-chunk | 3 | 1.000 | 1.000 | 0.333 |
| 5 | adversarial | 1 | 1.000 | 0.500 | 0.000 |
| 5 | negativo | 2 | — | — | 1.000 |

Cómo se leen:

- **Hit Rate y MRR, solo sobre las 8 respondibles.** Pasar de k = 3 a k = 5 sube el Hit Rate de
  0,875 a 1,000, porque la única pregunta sin acierto (la 3) lo encuentra en la posición 4. El MRR
  casi no se mueve (0,750 → 0,781): ese acierto nuevo entra con un recíproco de 1/4. Lo mismo
  muestra que el MRR **no es la posición promedio**. Con k = 5, las posiciones del primer acierto
  son 1, 1, 4, 2, 1, 1, 1, 2, con media 1,625. Su inverso sería 0,615, mientras que el MRR, que
  promedia los recíprocos, da 0,781. Una sola pregunta en la posición 4 arrastra la media de
  posiciones mucho más de lo que baja el MRR.
- **Dos tasas de abstención, no una.** La correcta es 1,000 con los dos k: el sistema se abstuvo
  en las dos negativas. Solo con eso parecería perfecto, pero la indebida es 0,250 con k = 3 y
  0,125 con k = 5: también se abstuvo en preguntas que tenían respuesta, 2 de 8 y 1 de 8. Un
  sistema que siempre se abstiene sacaría 1,000 en la primera y 1,000 en la segunda. Por eso las
  dos van juntas.
- **Todas las abstenciones son la frase exacta** («El corpus no contiene información
  suficiente.»), así que el detector no se perdió ninguna. Se revisó a mano porque `qwen3`
  responde a veces en inglés (preguntas 5 y 7), y una abstención en inglés no la habría detectado.
- **Si las negativas entraran al Hit Rate** como fallos, el techo con 2 negativas de 10 sería
  0,80, y el 1,000 de k = 5 se reportaría como 0,800 sin que el sistema hubiera cambiado. La
  métrica correcta para ellas es la abstención.
- **Con 8 respondibles, cada pregunta mueve el Hit Rate en 0,125.** Las diferencias entre k son
  una o dos preguntas y no permiten generalizar mucho más allá de este golden set.

## 2.c Los tres peores casos

La evidencia completa (ingesta, recuperados con puntaje y qué palabras clave de la respuesta
esperada tiene cada uno, y la respuesta) está en `salidas/bge-m3@ollama-local/fallos_evidencia.txt`.

### Caso 1 — Pregunta 6 (InstructGPT contra DPO): **generación / prompt**

Se abstiene con k = 3 **y** con k = 5, aunque la métrica dice acierto en la posición 1 con los
dos k. Con k = 5 los recuperados son:

```
1. 0.660  rafailov-2023-dpo-0025  contiene: ['reinforcement learning', 'PPO']
2. 0.658  rafailov-2023-dpo-0003  contiene: ['reward model', 'PPO', 'binary cross entropy', 'change of variables']
3. 0.657  rafailov-2023-dpo-0019  contiene: ['reward model', 'PPO']
4. 0.651  rafailov-2023-dpo-0021  contiene: ['PPO']
5. 0.650  ouyang-2022-instructgpt-0006  contiene: —
respuesta: El corpus no contiene información suficiente.
```

El fragmento `rafailov-2023-dpo-0003` tiene **todas** las piezas de la respuesta: el paper de DPO
describe el pipeline RLHF que reemplaza (modelo de recompensa + PPO) y su propia pérdida
(cambio de variables, entropía cruzada binaria). La recuperación cumplió, y la ingesta de los
dos documentos no tiene nada raro (118 y 66 fragmentos). La falla está en la generación: el
prompt obliga a usar solo el contexto, la pregunta nombra «InstructGPT», y ningún fragmento del
top-5 lo nombra así. Un modelo de 1,7 B parece leer eso como «no está» y se abstiene. Es una
abstención indebida que el Hit Rate no ve.

### Caso 2 — Pregunta 5 (RAG contra RAGAS) con k = 3: **recuperación**

```
1. 0.741  es-2023-ragas-0002  contiene: ['faithfulness']
2. 0.730  es-2023-ragas-0000  contiene: —
3. 0.704  es-2023-ragas-0001  contiene: —
respuesta: El corpus no contiene información suficiente.
```

Los tres vecinos son de RAGAS y **ninguno de Lewis et al.** La pregunta menciona los dos, pero
el vector de la pregunta queda más cerca del paper que habla de evaluación. Con ese contexto,
abstenerse no es un error del generador: la mitad de la respuesta no estaba. Es una falla de
recuperación propia de las preguntas que necesitan dos documentos: el top-k lo acapara el
documento más parecido. Con k = 5, Lewis et al. entra en la posición 4 y la respuesta sale
correcta. Aun así, la métrica marca acierto en la posición 1 con los dos k, por la limitación
declarada en la 2.a (acierto a nivel de documento). Es el caso más claro de un acierto que no es
un acierto.

### Caso 3 — Pregunta 4 (tamaño de modelo para chain-of-thought) con k = 5: **generación**

```
1. 0.735  wei-2022-chain-of-thought-0043  contiene: ['540B']
2. 0.734  wei-2022-chain-of-thought-0009  contiene: ['100B', 'only yields performance gains when used with models of', '540B']
3. 0.731  wei-2022-chain-of-thought-0010  contiene: ['540B']
...
respuesta: … el inicio de los beneficios se observa en modelos de tamaño similar a 540B parámetros.
```

La recuperación trae el fragmento correcto en la posición 2, y con k = 3 la respuesta era
correcta («aproximadamente 100B»). Con k = 5 entran dos fragmentos más, ninguno con la respuesta,
y el modelo cambia a «540B», la cifra que más se repite en el contexto (aparece en tres de los
cinco fragmentos). Más contexto empeoró la respuesta, algo que ni el Hit Rate ni el MRR registran
(los dos suben de k = 3 a k = 5). Es una falla de generación que el golden set solo detecta si se
leen las respuestas. Por eso el CSV ahora guarda la respuesta entera.

### Una falla del verificador, no del pipeline: la pregunta 3

Con k = 3 la pregunta 3 es el único fallo de recuperación de la tabla (posición «—»), pero la
respuesta es correcta («Leiden (Traag et al., 2019)»). El fragmento en la posición 1,
`edge-2024-graphrag-0011`, **sí dice Leiden**, pero con otras palabras, y la frase anotada
(«Community detection (e.g., Leiden») está en los fragmentos 0007 y 0008, que no se recuperaron.
La anotación era más estricta que la pregunta. No se corrige ahora, porque cambiar el golden set
después de ver los resultados sería ajustar el verificador al sistema. Queda como candidata para
una versión 1.1 del golden set, declarada como tal, y descuenta el 0,125 que separa el Hit Rate@3
de 1,000.

### Contraprueba: otro generador sobre la misma recuperación

Dos de los tres casos se atribuyeron a la generación, y eso se puede poner a prueba: si el
diagnóstico es correcto, un generador más capaz sobre **los mismos fragmentos** debería
arreglarlos. Se repitió la 2.b con `granite3.3` (8,2 B, en el Ollama de la H200), con el mismo
prompt, temperatura 0 y el índice `bge-m3@h200`. Ese índice da el mismo top-k que el local en
estas preguntas (Parte 1.3), así que lo único que cambia es el generador. Se eligió `granite3.3`
porque fue el único generador que cargó en la H200 ese día; `qwen3:32b`, `gemma3:27b` y
`gpt-oss:120b` hicieron caer el proceso de Ollama del servidor. Salida cruda en
`salidas/bge-m3@h200/granite3.3/`.

| generador | k | Hit Rate@k | MRR | abstención correcta | abstención indebida |
|---|---|---|---|---|---|
| `qwen3:1.7b` | 3 | 0,875 | 0,750 | 1,000 | 0,250 |
| `granite3.3` | 3 | 0,875 | 0,750 | 1,000 | **0,000** |
| `qwen3:1.7b` | 5 | 1,000 | 0,781 | 1,000 | 0,125 |
| `granite3.3` | 5 | 1,000 | 0,781 | 1,000 | **0,000** |

Hit Rate y MRR no se mueven, como se esperaba, porque miden la recuperación. Lo que cambia son
las respuestas. En el caso 1, `granite3.3` responde la pregunta 6 con los dos k usando lo que está
en `rafailov-2023-dpo-0003` (modelo de recompensa + PPO contra la pérdida directa de DPO). En el
caso 3 responde «aproximadamente 100 billion parameters» también con k = 5, sin dejarse arrastrar
por el 540B. Eso respalda que esos dos casos eran fallas de generación y no de recuperación. La
Parte 3 llega a lo mismo en la pregunta 6 por el otro lado: el híbrido trae un fragmento que
nombra InstructGPT y `qwen3:1.7b` deja de abstenerse.

Pero el 0,000 no es todo mejora, y hay que leer las respuestas para verlo. En la pregunta 5 con
k = 3 (el caso 2) `granite3.3` también responde, y describe el RAG de Lewis et al. con detalle,
aunque ninguno de los tres fragmentos recuperados es de ese paper. Esa parte sale de lo que el
modelo ya sabía, no del contexto, o sea que incumple la instrucción del prompt. `qwen3:1.7b` se
abstenía ahí, y según lo discutido en el caso 2 eso era lo correcto. El generador más grande
tapa una falla de recuperación en vez de arreglarla, y la métrica lo cuenta como acierto. Algo
parecido pasa en la negativa 9: `granite3.3` dice la frase de abstención, pero agrega que
«GPT-4 es un modelo desarrollado por Microsoft», que es falso y no está en el corpus. Como
`se_abstuvo` solo busca la frase, la cuenta como abstención correcta.

La conclusión honesta es que el generador explica los casos 1 y 3, pero cambiarlo trae su propio
problema: responde con conocimiento propio cuando el contexto no alcanza. Con 8 respondibles y
dos modelos que difieren en tamaño **y** en familia, esto no dice cuál de los dos factores pesa
más. Para medirlo bien haría falta una métrica de fidelidad al contexto, como la de RAGAS
(Opción D), que no se corrió.

### Las tres fallas silenciosas de la Parte 0, en este corpus

- **PDF sin texto (0.a): no está.** Los 16 documentos tienen capa de texto y 0 páginas vacías
  (`salidas/corpus_verificacion.txt`), y la ingesta no rechazó ninguno.
- **Truncado (0.b): no está.** Los fragmentos son de 512 tokens (el más largo re-tokenizado, 513)
  contra un tope de 8192, y la ruta de Ollama pide `truncate: false`, así que un texto que no
  cupiera daría error en vez de recortarse.
- **El índice que no se queja (0.c): sí está.** Los mejores puntajes de las negativas (0,555 y
  0,602) caen **dentro** del rango de los de las respondibles (0,548 a 0,741). La respondible 2
  tiene un mejor puntaje más bajo que las dos negativas. Ningún umbral sobre el puntaje separaría
  «hay respuesta» de «no la hay». Aquí lo que salvó la tasa de abstención correcta fue el
  generador, no el índice.

Además, la sospecha de la Parte 1 sobre la bibliografía (40,6 % de los tokens) no se confirmó.
De los 50 fragmentos recuperados con k = 5, solo 1 tiene aspecto de bibliografía
(`gao-2023-rag-survey-0074`), según una heurística simple que cuenta marcas de cita.
