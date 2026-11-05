# Propuesta de diseño: concept-probe por capas

Estado: **propuesta, sin código; decisiones tomadas el 2026-11-05** (sección 7). Base: rama
`review-consistency-fixes` (commit `558d06b` más los dos commits de documentación). Responde a
[`LIBRARY_ISSUES.md`](LIBRARY_ISSUES.md): las entradas 1–31, los criterios de diseño y la tabla de
funciones que faltan. Los nombres de módulos y funciones son provisionales, salvo los que fija la
sección 7.

## 1. Cómo está hoy

- `probe.py` tiene 3000 líneas y hace casi todo: codifica las entradas, pasa los textos por el
  modelo, resume los tokens, calcula la dirección y la estadística, elige la capa, genera, empuja,
  guarda y grafica.
- `ConceptProbe.train` (`probe.py:1384-2208`) son ~825 líneas seguidas: genera o lee los textos,
  pasa cada uno por el modelo, calcula las direcciones, evalúa, elige la capa, mide los textos de
  evaluación en modo generado ("transfer check"), calcula las referencias de z, grafica y escribe
  `metrics.json` y `tensors.npz`. Ninguna de esas partes se puede usar sola: por eso el workshop
  rehízo casi todo en `shared/` (entrada 20).
- `score_prompts` (`probe.py:2480`) y `multi_probe_score_prompts` (`multi_probe.py:103`) repiten
  el mismo bucle: un pedido por vez, cada resultado a disco.
- Hay una sola dirección por capa, `concept_vectors`, para leer, para estandarizar y para empujar.
- Los controles al azar son tres mecanismos sueltos: `random_scoring_vector`,
  `random_steering_vector` (una sola dirección al azar, compartida por todas las capas) y la
  permutación del d máximo.
- `eval_system.py` y `coherence.py` leen los resultados del disco (npz y `log.jsonl`): hoy no se
  puede dejar de escribir sin romperlos.

Lo que ya está bien y se conserva: `ResidualStreamEditor` (un mecanismo general de hooks), la
codificación de textos por rol y `preview_input` (qué ve exactamente el modelo), las referencias de
z atadas al contexto en que se midieron, el split entre la mitad que elige la capa y la que mide el
d, y la validación de opciones.

## 2. Tres ideas que atraviesan las capas

Estas tres ideas (los nombres son míos) son las que evitan escribir una función por entrada: varias
entradas resultan ser el mismo cambio visto desde lugares distintos.

**Contexto de lectura.** Cómo se le mostró un texto al modelo: solo, como turno del usuario o como
respuesta del modelo; con qué system prompt; después de qué pregunta; y qué tokens se resumen y
cómo. Hoy existe a medias (`_text_context`, `GENERATED_CONTEXT`). Pasa a ser un objeto que usan la
codificación (capa 1), la clave de la caché (25), las calibraciones de la probe (14, 19, 26) y los
sets de evaluación (17).

**Grupos.** Un id por ítem que dice qué ítems no son independientes: los dos polos de un par mínimo
(13), las respuestas de los dos polos a la misma pregunta (hoy `eval_paired`), los tokens de un
mismo texto cuando cada token es una muestra. El split de selección, el bootstrap, la permutación y
los tests apareados respetan los grupos. Resuelve 13 y 29 y habilita "entrenar con cada token".

**Controles.** Una sola pieza que fabrica controles: n direcciones al azar del mismo largo (una
distinta por capa, o una misma para todas las capas cuando se empuja en varias) y etiquetas
mezcladas. La usan el nulo del d (18), el control del empuje (24) y las tareas de control.
Reemplaza los tres mecanismos de hoy.

Y dos reglas para todas las capas, que salen de las decisiones D1, D4 y D5:

**Devolver y guardar.** Cada función que produce un resultado (leer, entrenar, generar, puntuar,
evaluar, empujar) lo devuelve completo en memoria y además lo guarda en disco por defecto. Guardar se
apaga en cada llamada o una vez para toda la sesión; ante la duda, se guarda de más. Las cuentas
sobre arrays (un d, una AUROC) no tienen nada que guardar. Esto cambia un criterio de
`LIBRARY_ISSUES.md` ("el núcleo trabaja con arrays y no tiene efectos secundarios"): en el código,
calcular y guardar siguen siendo piezas separadas (guardar es siempre el último paso y nada lo lee
para calcular), pero las funciones públicas llaman a guardar por defecto.

**Toda elección automática queda registrada y se puede cambiar.** Cuando la librería elige algo que
el usuario no pidió (el dtype con `"auto"`, la dirección con la que se empuja, la capa, el tamaño de
lote, las semillas de los controles, en qué contexto se calibraron los z), lo anota con su motivo
en el resultado, en la consola si está prendida, en el log de la corrida y en lo que se guarda
(config y métricas). Y cada una de esas elecciones tiene una opción para elegir otra cosa.

## 3. Las capas

Para cada capa: qué hace, qué recibe y devuelve, qué entradas resuelve y qué código de hoy absorbe.
El orden sigue el de los criterios; la capa 0 la agrego yo, porque los criterios empiezan en "Leer"
y siete entradas son de carga del modelo.

### Capa 0 — Cargar el modelo

- `load_model(model_id, dtype=..., quantization=..., device_map=...)` devuelve modelo, tokenizer e
  **identidad**: lo que cambia las activaciones (modelo, dtype ya resuelto, cuantización, chat
  template y su fecha). La identidad viaja con las representaciones, la caché y la probe, y es lo
  que se compara al usar una probe con un modelo (hoy `_MODEL_IDENTITY_KEYS`).
- `quantization`: nada, un nivel con nombre (`"nf4"`, `"int8"`; HQQ si lo sumamos) o cualquier
  `quantization_config` de transformers (3). `use_4bit` desaparece (D7).
- `dtype="auto"` por defecto: bf16 si la GPU lo soporta; si no, fp32 para Gemma 3 y fp16 para el
  resto. Se guarda el dtype resuelto, nunca "auto", y se registra qué se pidió, qué se eligió y por
  qué (la GPU y la familia del modelo) (2, D5). Aviso al cargar Gemma 3 en fp16 (1).
- Sin red, resuelve la ruta local en la caché de Hugging Face antes de cargar (6). Pasa `dtype` y
  cae a `torch_dtype` en versiones viejas de transformers (8). El parche de HQQ, con chequeo de
  versión, sólo si ofrecemos HQQ por nombre (9).
- Todo lo que hoy recibe un `ModelBundle` acepta también un modelo ya cargado, incluido
  `ProbeWorkspace` (4).
- Absorbe: `ModelBundle.load`, `apply_tokenizer_overrides`, `_MODEL_IDENTITY_KEYS`.

### Capa 1 — Leer

Recibe textos con su contexto de lectura, o entradas ya tokenizadas (las generaciones: así no se
vuelve a tokenizar lo que el modelo generó, como hoy). Devuelve arrays `(n, capas, dim)`, o una fila
por token con el texto como grupo, junto con la identidad y el contexto. Tres piezas separadas:

1. **Codificar:** texto + contexto → ids y el tramo de tokens de cada mensaje y del texto (28: hoy
   `_user_text_span` lo calcula en privado). `preview_input` pasa a mostrar todos los tramos.
2. **Pasar por el modelo:** en lotes con padding (5), capturando sólo las capas pedidas y resumiendo
   en la GPU, sin bajar a la CPU todas las capas de todos los tokens (hoy
   `forward_hidden_states_all_layers` lo hace, en float32). Chequea que todo sea finito y, si no,
   falla nombrando el dtype y la capa (1). Las capas siguen siendo las salidas de los bloques; la de
   embeddings se pide aparte (D6).
3. **Resumir:** un tramo × un agregador. Tramos: el texto, toda la secuencia, los últimos k tokens,
   una posición, una subcadena. Agregadores: media, último, máximo, ninguno (cada token es una
   muestra). Los seis modos de hoy (`assistant_all_mean`, etc.) pasan a ser nombres de
   combinaciones, así que la config no cambia.

- **Caché** en disco con clave identidad + contexto + resumen + texto (25). Es la forma en que
  leer guarda lo que lee, así que está prendida por defecto (D4) y también evita volver a pasar un
  texto por el modelo. Ahí se guarda en float16 cuando todo entra, con float32 en las capas donde no
  (5: las últimas capas de Gemma 3 4B pasan el máximo de float16). Ojo con el tamaño: 1000 textos de
  Llama 3.2 3B ocupan unos 170 MB.
- Leer en lotes es el default; el tamaño de lote se registra en cada resultado, y lotes de 1
  reproducen la lectura de hoy exacta (D8).
- Entrada 12: recomiendo no insertar un prompt que nadie escribió para el rol "respuesta del
  modelo" (va contra la idea de que el usuario vea exactamente qué entra). Mejor mensaje de error y
  la documentación al lado del rol.
- Absorbe: `_encode_text_for_read`, `_user_text_span`, `forward_hidden_states_all_layers`,
  `reps_from_hs_layers`, `extract_text_reps`, `extract_completion_reps`, y `score_text_tokens` (que
  pasa a ser leer cada token y proyectar).

### Capa 2 — Direcciones

- Un **estimador** recibe las representaciones de dos grupos en una capa y devuelve una dirección
  de largo 1 en el espacio crudo de activaciones, orientada para que el primer grupo proyecte más
  alto. `fit_directions(estimador, pos, neg)` lo aplica capa por capa.
- Estimadores: diferencia de medias con `none`/`var`/`sd` (lo de hoy), sin las dimensiones de
  varianza extrema que el concepto no mueve (`meandiff_drop_outliers` del workshop), logística y LDA
  con shrinkage (estas dos con scikit-learn, que ya es un extra opcional). Uno nuevo se registra sin
  tocar nada más.
- Un polo contra una línea de base (texto neutro, otros conceptos) es el mismo estimador con la
  línea de base como segundo grupo; lo que cambia es dónde queda el cero, y eso es de la probe.
- Interfaz pensada para crecer: aceptar `fit(X, y)` deja entrar la regresión sobre una variable
  continua sin cambios. Multiclase y subespacios necesitan k direcciones por capa: irían en una
  clase aparte, al final, sin complicar la probe de dos polos.
- Más adelante, transformaciones antes del estimador: proyectar afuera direcciones de confound,
  LEACE.
- Absorbe: `concept_direction`.

### Capa 3 — La probe

Un objeto en memoria, sin modelo y sin carpeta (20). Tiene:

- **direcciones para leer y direcciones para empujar**, por separado (15, D1). Por defecto se lee
  con la dirección del estimador elegido y se empuja con la diferencia de medias cruda; una opción
  elige otra dirección para empujar, y la probe registra con cuál lee y con cuál empuja;
- la **capa** elegida y qué capas usar al puntuar (la `layer_selection` de hoy);
- **calibraciones por contexto de lectura**: centro y escala de los z (lo de hoy), y opcionalmente
  dónde caen textos sin el concepto (19) y un umbral calibrado o una probabilidad (26). Cada
  calibración dice en qué contexto se midió. Leer en un contexto sin calibración devuelve el puntaje
  crudo con el motivo, en vez de un `None` callado (14). `probe.calibrate(...)` agrega una
  calibración sin reentrenar;
- las **unidades del empuje**: la norma media del residuo por capa, y la escala "sigma" medida sobre
  la dirección de empujar (hoy se mide sobre la única que hay);
- la identidad del modelo, los nombres de los polos y de dónde salió (estimador, resumen, contexto).

Se construye entrenando desde arrays (`Probe.fit(pos, neg, estimador, selección)`), desde una
dirección hecha a mano (`Probe.from_directions`), o desde una corrida guardada (incluidas las de
hoy). Los métodos que sólo usan arrays (puntaje crudo, z, clasificar) no necesitan el modelo.

### Capa 4 — Intervenir

- Una **intervención** describe qué hacer, en qué capas y en qué posiciones, con cualquier vector:
  **sumar**, **borrar** una dirección (proyectarla afuera o fijarla en un valor), **trasplantar**
  (el `ActivationPatch` de hoy), **rotar** en un plano, y **combinar** varias. Todas compilan a la
  misma función de edición y usan un solo mecanismo de hooks, el `ResidualStreamEditor` que ya
  existe.
- El dispositivo sale de la salida del bloque, no de sus parámetros (11).
- Una intervención no sabe de probes: `probe.steering(alpha, unidad, capas)` arma un "sumar" con las
  direcciones de empujar y las unidades de la probe. Con un vector cualquiera es lo mismo sin probe.
- Con padding a la izquierda, las intervenciones en posiciones fijas tienen que correrse por el
  padding de cada fila: es el detalle técnico más delicado de esta capa.
- `LayerSteerer` y `MultiLayerSteererLayerwise` quedan como envoltorios (D7). *En la etapa 4 se
  sacaron (D7 revisado).*

### Capa 5 — Generar y puntuar

- `generate(...)` en lotes con padding a la izquierda (22) y una intervención opcional; devuelve las
  generaciones en memoria: ids, dónde empieza y termina la respuesta, texto, ajustes (21).
- **Puntuar** = leer las generaciones (capa 1, sin la intervención, como hoy) y proyectar con una o
  varias probes: una tabla por generación con pedido, α, tokens, puntaje por token, por capa y por
  probe, y z donde hay calibración (21). Varias probes dejan de ser un bucle aparte: es puntuar con
  una lista.
- Callback por token mientras genera, con el puntaje de cada probe (30). Si además se empuja, esa
  lectura incluye el vector sumado; el puntaje que se reporta sigue siendo el de la relectura limpia.
- Probabilidad de respuestas dadas (27): pedidos + respuestas posibles + intervención opcional →
  log-probabilidad de cada respuesta, en lotes, con los tramos de la capa 1.
- Absorbe: `generate_once`, `generate_batch` y los bucles de `score_prompts` y
  `multi_probe_score_prompts`.

### Capa 6 — Evaluar

Todo recibe arrays (puntajes, o representaciones y direcciones), nunca textos.

**6a. Estadística**, funciones puras: d de Cohen, AUROC, acierto con un umbral, intervalos por
bootstrap, permutación, split de selección, y la diferencia entre dos clasificadores en los mismos
ítems con su intervalo apareado (29). Todas aceptan grupos (13).

**6b. Evaluar la lectura**, `evaluate_reading(direcciones, sets, principal, selección, controles)`:

- **sets de evaluación con nombre**, cada uno con su contexto y sus grupos; el principal elige la
  capa y da el d que se reporta, los demás se miden en esa capa. Los textos de evaluación de hoy en
  modo generado pasan a ser uno de esos sets (17, "varios sets de transferencia", D2);
- la curva del d por capa en la mitad de selección y en la de reporte, el d con su intervalo, AUROC
  y acierto con el umbral calibrado (26);
- el **nulo**: percentil 95 de |d| con n direcciones al azar por capa, en los mismos ítems (18);
- etiquetas mezcladas como control;
- la **tabla cruzada** de varias probes: cuánto lee cada una en los sets de las otras, con su nulo,
  y el coseno entre direcciones (31).

**6c. Evaluar el empuje**: un barrido de dosis que recibe la intervención, los pedidos, las dosis,
n controles al azar del mismo largo (24) y una lista de **puntuadores** enchufables: perplejidad
bajo el modelo sin empujar y tasa de repetición (locales), una probe distinta de la que empuja,
cualquier función externa (un clasificador, un juez). Devuelve la tabla y la mayor dosis coherente
según una regla que se puede cambiar (23). El juez de Groq de hoy pasa a ser un puntuador más, que
recibe los textos en memoria.

**6d. Datasets** (al final): auditoría de un dataset de dos polos, ataques a la probe y métricas de
diversidad (10), desde `shared/audit.py` y `shared/redteam.py` del workshop cuando se estabilicen.

### Capa 7 — Guardar y graficar

- Un guardador por tipo de resultado (lecturas, probe, evaluación, generaciones, barridos de
  empuje), que cada función pública llama al final por defecto (D4), y su lector. `metrics.json`
  conserva sus claves y suma las nuevas (nulo, AUROC, acierto, sets con nombre, elecciones
  automáticas).
- Los gráficos reciben el resultado de la evaluación: la curva del d en la mitad de reporte con
  `best_d` marcado y la banda gris del nulo (16, 18); el histograma con un título que dice qué
  ítems y qué capa muestra, y colores que no pongan el polo positivo en rojo (16).
- Absorbe: `visuals.py`, el guardado dentro de `train` y `score_*`, `reporting.py`.

### La API de alto nivel, como composición

`ProbeWorkspace.train_concept` hace lo mismo que hoy, como una secuencia de pasos de las capas:

1. armar los datos del concepto: los textos, o las respuestas generadas bajo el system prompt de
   cada polo (capas 1 o 5);
2. leer (capa 1);
3. calcular las direcciones (capa 2);
4. evaluar y elegir la capa (capa 6);
5. armar la probe con sus calibraciones (capa 3);
6. guardar y graficar (capa 7).

`ConceptProbe`, `score_texts`, `score_prompts`, `multi_probe_score_prompts` y `run_scored_eval`
mantienen sus firmas y sus defaults. `eval_system.py` y `coherence.py` pasan a leer los resultados
en memoria: los archivos se siguen escribiendo por defecto (D4), pero ya nada los necesita para
analizar.

### Organización del código

Desde la etapa 1, cada capa es un módulo en la raíz del paquete, en vez de una subcarpeta `core/`:
así se movió menos código, y `concept_probe.probe` reexporta lo que se mudó. Lo que ya existe:

```
concept_probe/
  modeling.py           # 0  cargar: dtype automático, cuantización, identidad, modelo ya cargado
  inputs.py             # 1  codificar: roles, tramos de cada mensaje y del texto, caracteres de cada token
  reading.py            # 1  pasar por el modelo en lotes, readouts, caché (SQLite por identidad)
  directions.py         # 2  estimadores (diferencia de medias y variantes, logística, LDA) y su registro
  linear_probe.py       # 3  Calibration (el único cálculo de z) y el umbral; ReadContext vive en inputs.py
  stats.py              # 6a la estadística básica, mudada de probe.py (la etapa 3 la completa)
  settings.py           #    ajustes de sesión: guardar, carpeta, consola
  record.py             #    registro de elecciones automáticas (consola, log, lo guardado)
```

Desde las etapas 3 y 4 existen también:

```
concept_probe/
  evaluation.py         # 6b evaluar la lectura: sets con nombre, nulos, etiquetas mezcladas
  interventions.py      # 4  sumar, borrar, trasplantar, rotar, una función propia; intervene()
  generation.py         # 5  generar en lotes, leer mientras genera, probabilidad de respuestas
  scoring.py            # 5 y 7  puntuar con varias probes, Scores, sus archivos
```

Lo que falta, con los nombres provisionales de la propuesta:

```
concept_probe/
  steering_eval.py      # 6c (etapa 5)
  probe.py, multi_probe.py, eval_system.py, ...   # API de hoy, como composición
```

`concept_probe.probe` sigue exportando lo que hoy importan otros proyectos (el workshop importa
`concept_direction`, `cohen_d_np`, `forward_hidden_states_all_layers`, `extract_text_reps`).

## 4. Entradas que se pisan

| Mismo cambio | Entradas |
|---|---|
| Contexto de lectura | 14, 17, 19, 25, 26 y "varios sets de transferencia" |
| Grupos | 13, 29, "entrenar con cada token" y el `eval_paired` de hoy |
| Controles | 18, 24, "etiquetas mezcladas" y los `random_*_vector` y la permutación de hoy |
| Un lector | 1, 5, 25, 28, los readouts nuevos (máximo, posición, cada token) y la capa de embeddings |
| Generar en memoria | 21, 22, 27, 30 y la duplicación de `multi_probe_score_prompts` |
| Probe en memoria | 20, 15, 26 y "un polo contra una línea de base" |
| Cargar | 2, 3, 4, 6, 8, 9 y "cuantización configurable" |
| Evaluar el empuje | 23, 24 y el juez de `coherence.py` |
| Datasets | 10, "auditoría y red teaming" y "diversidad" |

Además hay entradas repetidas tal cual entre la lista y la tabla de funciones: 15 = "una dirección
para leer y otra para empujar"; 3 = "cuantización configurable"; 10 = "auditoría y red teaming"; 17
es un caso de "varios sets de transferencia". Y 16 y 18 cambian el mismo gráfico.

## 5. Entradas que chocan entre sí o con los criterios

1. **15 contra "mismos defaults y resultados".** Separar las direcciones cambia el empuje de quien
   hoy usa `direction_normalization="var"` (con `"none"`, el default, no cambia nada). Y la unidad
   "sigma" se mide hoy sobre la dirección de leer; con dos direcciones tiene que medirse sobre la de
   empujar. → D1.
2. **17 contra "mismos resultados".** Cambiar qué d es el principal en los modos donde el modelo
   responde bajo un system prompt por polo cambia, en esos modos, la capa elegida, el d, las
   referencias de z y la unidad "sigma". Un dato a favor del cambio: las referencias de z que hoy
   usa `score_prompts` se midieron en respuestas leídas con el system prompt de su polo, y
   `score_prompts` lee respuestas generadas sin él. → D2.
3. **13 ("inferirlo si los dos polos tienen el mismo largo") contra "nada acoplado en silencio".**
   Recomiendo que el apareamiento se declare (pasando los grupos), nunca se infiera del largo.
4. **La capa de embeddings contra los índices de capa.** Si la capa 0 pasa a ser la de embeddings,
   se corren todos los índices y las probes guardadas apuntan a otra capa. → D6.
5. **2 (dtype automático) contra la identidad del modelo.** El mismo código da otro dtype en otra
   GPU: una probe entrenada en bf16 no se puede usar en una T4 sin `allow_model_mismatch` (es lo
   correcto, porque las activaciones cambian, pero es nuevo). Y cambia el default en GPUs sin bf16.
   → D5.
6. **5 y 22 (lotes) contra "mismos resultados".** En media precisión, el padding cambia la forma de
   las cuentas y los resultados difieren en el redondeo (ya pasa hoy al generar con
   `gen_batch_size`). → D8.
7. **21 (en memoria) contra lo que ya existe.** `eval_system.py` y `coherence.py` leen npz y
   `log.jsonl`; si las funciones dejan de escribir antes de portarlos, se rompen. El orden de las
   etapas lo evita. → D4.
8. **18 contra el control de empuje de hoy.** El nulo del d usa direcciones al azar independientes
   por capa; el control del empuje usa a propósito una sola para todas las capas (si no, se cancelan
   al empujar varias). No se contradicen, pero la pieza de controles tiene que ofrecer las dos formas
   de manera explícita.
9. **Las semillas de los controles al azar** mezclan el nombre de la carpeta de la corrida, y una
   probe en memoria no tiene carpeta. Para no cambiar los controles de las probes ya guardadas, las
   que tienen carpeta conservan la semilla de hoy.
10. **19 y 26 contra el cero de z.** Habría tres "ceros": el punto medio entre los polos, dónde caen
    los textos neutros y el umbral calibrado. Propongo mantenerlos separados: z sigue midiéndose
    desde el punto medio, el neutro se reporta, y el umbral se usa sólo para clasificar.
11. **30 con empuje.** Leer mientras se genera con el empuje activo lee también el vector sumado
    (por eso hoy se relee sin el hook). El callback lo tiene que decir.

## 6. Orden de trabajo

Cada etapa usa sólo lo de las anteriores, trae tests propios para cada pieza nueva, actualiza el
README y las notas de versión, y termina con los tests de referencia pasando. Al final de cada etapa
paro y te muestro qué cambió antes de seguir.

**Etapa 0 — Red de seguridad y arreglos sueltos.** No cambia la API ni ningún número. *Hecha el
2026-11-05: `tests/test_reference_outputs.py` fija los resultados de hoy sobre un modelo chico
determinista (`tests/tiny_model.py`), y siguen iguales después de los arreglos. Probada además con
modelos reales en la RTX 2060, sin red: Qwen 2.5 0.5B y Gemma 3 270M cargan desde la caché sin el
aviso de `torch_dtype`; Gemma 3 270M en fp16 avisa al cargar y frena al leer (11 de 18 capas no
finitas), y en fp32 lee bien; con las 24 capas de Qwen 0.5B descargadas a la CPU, el código
anterior del empuje fallaba ("Tensor on device meta") y el nuevo empuja (la entrada 11 era un error
real); entrenar con textos y con respuestas generadas, leer y empujar anda de punta a punta.*
- Tests de referencia: guardar lo que hoy devuelven `train`, `score_texts` y `score_prompts` con el
  modelo falso de los tests, para comprobar en cada etapa que la API de hoy sigue dando lo mismo.
- Entradas 7, 8, 11, 6, 1 (chequeo de valores no finitos y mensaje de error), 12 (mensaje y doc), y
  la documentación de 15 y 19.

**Etapa 1 — Cargar y leer (capas 0 y 1).** Entradas 2, 3, 4, 9, 5, 25 y 28; readouts nuevos
(máximo, posición o palabra, cada token) y la capa de embeddings; el contexto de lectura. También
las dos piezas de las reglas de la sección 2, que las etapas siguientes reusan: el registro de
elecciones automáticas y el guardado por defecto. `extract_text_reps` y el entrenamiento pasan a
usar el lector nuevo.
- Al terminar: el entrenamiento da lo mismo (con la tolerancia de D8) y leer cientos de textos es
  varias veces más rápido.
- *Hecha el 2026-11-05; los tests de referencia siguen iguales. Probada con modelos reales: dtype
  automático (fp16 para Qwen, fp32 para Gemma 3 en la RTX 2060), NF4, int8 y HQQ4; leer 64
  textos en lotes de 16 tardó 0.5 s contra 3.8 s de a uno, y un d pasó de 3.4043 a 3.4041;
  leer una palabra y los tramos de cada mensaje andan con los tokenizers de Qwen, Llama y Gemma.
  Cambios respecto de la propuesta: la caché es un archivo SQLite por identidad del modelo y no
  un archivo por texto (en Windows, abrir cada archivo tardaba 25 ms, más que leer el texto con un
  modelo chico); su clave son los ids de los tokens, que ya contienen el contexto de lectura, y no
  el texto más el contexto; el contexto de lectura como objeto pasa a la etapa 2, donde lo
  necesitan las calibraciones; `configure(save=False)` apaga también la caché de un workspace.*

**Etapa 2 — Direcciones y probe en memoria (capas 2 y 3).** Entradas 20, 15, 14, 19 (set neutro) y
26 (umbral); estimadores logística, LDA y sin dimensiones extremas; un polo contra una línea de base.
`ConceptProbe.train` se reescribe como composición.
- Al terminar: lo de `shared/estimators.py` y la probe de `shared/redteam.py` del workshop se hace
  con la librería.
- *Hecha el 2026-11-05; los tests de referencia siguen iguales (ninguno empuja con una probe `var`,
  que es lo único que D1 cambia). Probada con modelos reales: los cinco estimadores entrenan desde
  arrays en Gemma 3 270M; con `mean_diff_var` en Qwen 0.5B, entrenar anuncia que empuja con la
  diferencia de medias cruda (coseno 0.59 con la de leer) y mide "sigma" sobre ella; una probe de
  textos explica por qué no da z al leer generaciones, y la da después de `calibrate()`; los textos
  neutros cayeron en z = +5. Cambios respecto de la propuesta: **D3 cambió** (2026-11-05): no hay
  una clase `Probe` aparte. La necesidad de la entrada 20 (entrenar desde activaciones ya leídas,
  sin cargar el modelo de nuevo) la cubre la misma `ConceptProbe`, que siempre sabe de qué modelo es
  (`identity`) pero necesita el modelo cargado solo para leer textos, generar y empujar (`attach`).
  Se construye entrenando, con `fit` (desde activaciones), con `from_directions` o con `load`, y
  guarda por defecto. Sus campos se simplificaron sin mantener los viejos (la librería es para
  usuarios nuevos): `directions`, `layer`, `steer_directions`, `identity`, y `calibrations` en lugar
  de los seis campos de referencias; todo z pasa por la única fórmula de `Calibration`. La probe de
  dos polos y la de un polo contra una línea de base son la misma clase (`baseline=True`);
  `ConceptSpec` suma `neutral_texts`.*

**Etapa 3 — Evaluar la lectura (capas 6a y 6b, y sus gráficos).** Grupos (13), 29, 18, 16, 17, 26
(acierto en las métricas), 31, sets de evaluación con nombre, etiquetas mezcladas.
- Al terminar: `metrics.json` y el gráfico por capa traen el nulo, y `shared/evaluate.py` sobra.
- *Hecha el 2026-11-05 (`evaluation.py`, y AUROC, acierto, intervalos, comparación apareada y nulo
  en `stats.py`). Los números de los modos con textos no cambiaron; en los modos con un system
  prompt por polo el set principal pasó a leer las respuestas sin ese prompt (D2): en el modelo de
  los tests, d de 20.4 a 13.8, y con `evaluation.primary_set = "eval_with_system_prompt"` vuelve el
  20.4 de antes. Probada con Qwen 0.5B en modo compartido: en ese montaje chico la probe dio d 0.62
  contra 0.84 de las direcciones al azar y 0.66 con las etiquetas mezcladas, y el gráfico lo muestra.
  Cambios respecto de la propuesta: el apareamiento es un `paired` por set, no grupos generales
  (alcanza para los pares de preguntas y los pares mínimos; los tokens como muestra se puntúan por
  texto); el acierto se reporta solo para el set principal, porque el umbral calibrado no vale en
  otro contexto de lectura; `transfer_eval` desaparece de `metrics.json` (es el set `texts`).*

**Etapa 4 — Intervenir, generar y puntuar (capas 4 y 5).** Intervenciones (sumar, borrar,
trasplantar, rotar, combinar), entradas 22, 21, 30 y 27. `score_prompts` y
`multi_probe_score_prompts` como composición; `eval_system.py`, `coherence.py` y `reporting.py`
leen de memoria.
- Al terminar: generar y puntuar va en lotes, y nada obliga a escribir a disco.
- *Hecha el 2026-11-05 (`interventions.py`, `generation.py`, `scoring.py`). Los números de puntuar
  no cambiaron en los tests de referencia, salvo las respuestas con muestreo (en lotes el azar se
  sortea en otro orden, D8); cambiaron, como se esperaba, los archivos: cada llamada guarda una
  carpeta con `scores.npz`, `scores.jsonl`, `heatmaps.html` y `settings.json` (decidido el
  2026-11-05: "pocos por llamada"). Probada con Qwen 2.5 0.5B en la RTX 2060: ocho pedidos en un
  lote tardaron 1.3 s contra 10.5 s de a uno, con las mismas respuestas (7 de 8 iguales al empujar
  en todas las posiciones, por redondeo en fp16); empujar solo los tokens generados, con la caché de
  claves, da lo mismo en lote y de a uno; el z medio de las respuestas subió con la dosis (-0.47,
  +1.58, +3.22); la lectura por token mientras genera coincide con la relectura (diferencia máxima
  0.002); la probabilidad de una respuesta coincide con la calculada a mano, y borrar la dirección de
  valencia en todas las capas bajó las chances de "Positive" frente a "Negative" en una reseña
  positiva de 8.1 a 1.6. Cambios respecto de la propuesta: `LayerSteerer`,
  `MultiLayerSteererLayerwise`, `ResidualStreamEditor`, `ActivationPatch` y sus funciones no quedan
  como envoltorios sino que se sacan (D7 revisado), igual que `generate_once`, `generate_batch` y
  `forward_hidden_states_all_layers`; `ResidualStreamEditor` no se conserva como mecanismo, porque
  las posiciones por fila necesitan saber dónde está cada columna de cada llamada al modelo, y eso lo
  hace `intervene`. Leer bajo una intervención informa, en la capa editada, el flujo ya editado (lo
  que recibe la capa siguiente): Hugging Face registra la salida de un bloque antes de los hooks, y
  M5 midió así el puntaje en la capa empujada sin el vector. Se sacan opciones que el formato nuevo
  dejó sin sentido (`steering.do_steering`, `output.save_token_scores_npz` y las tres de los
  logits, que pasan a ser `output.generation_logits`). Los heatmaps usan los colores de los otros
  gráficos (positivo azul, negativo naranja).*

**Etapa 5 — Evaluar el empuje (capa 6c).** Entradas 23 y 24.
- Al terminar: `shared/steer.py` y las medidas de coherencia de `shared/measure.py` sobran.

**Etapa 6 — Datasets y extras.** Entrada 10 y diversidad; clasificadores de superficie junto a la
probe; probes de confounds y quitarlos (proyección, LEACE); curvas de aprendizaje; multiclase,
regresión, subespacios y ajuste circular; receta para correr varios modelos.

**Etapa 7 — Volver a correr lo representativo del workshop.** Los tests de referencia de cada etapa
prueban la API de hoy con un modelo falso; esta etapa prueba todo junto con modelos de verdad. En
probes-workshop, un script por experimento rehace su parte central con la librería nueva, sin usar
`shared/`, y compara con los números guardados en su `results/`. Todo en el modelo de trabajo
(Llama 3.2 3B sin cuantizar) salvo donde se dice.

| Qué se vuelve a correr | Qué prueba |
|---|---|
| Fase 0 y B0, reducidos: Gemma 3 4B con dtype automático, un modelo en NF4, uno en HQQ si se ofrece por nombre, la carga sin red, y Gemma 3 en fp16 fallando con un mensaje claro | capa 0 (entradas 1-9) |
| La batería de la Etapa 1 en dos de los doce modelos (Llama 3.2 3B y un Gemma 3), incluido el empuje con `var` | todas las capas, D1 |
| M0 (E0.1 y E0.2): la primera probe con la API de alto nivel, y los estimadores | API de alto nivel, capas 1-3, 6b, gráficos |
| M1: dónde se lee (cada token, máximo, una posición) | capa 1 |
| M3: textos escritos contra respuestas bajo un system prompt por polo | capa 6b, D2 |
| M5: borrar, trasplantar y la probabilidad de respuestas dadas | capas 4 y 5 |
| M8: un polo contra una línea de base, y textos neutros | capas 2 y 3 (19) |
| M9: varias probes, generación en lotes, lectura por token, controles del empuje, logística, intervalos, juez externo; y cuánto tarda contra la hora de antes | capas 5 y 6 |
| M2, M4 y M7, si en la etapa 6 entran sus funciones (curvas de aprendizaje; clasificadores de superficie y confounds; multiclase y rotación) | capa 6 |

Criterio: los mismos números salvo diferencias de redondeo (D8), con excepción de los cambios
esperados, que se listan antes de correr:
- el empuje de quien entrenó con `var` (D1);
- el d principal de M3 en los modos con un system prompt por polo (D2): tendría que parecerse a lo
  que M3 midió a mano leyendo esas respuestas sin su system prompt, no al 10.8 de hoy;
- la estadística de los pares mínimos de E0.1, ahora apareada (13): cambian el p y el intervalo, no
  el d;
- el dtype de Gemma 3 en la RTX 2060 (D5).

Cualquier otra diferencia es un bug. Salida: una tabla antes/después por experimento, y qué partes de
`shared/` quedaron reemplazadas por la librería.

Por qué este orden: las etapas 1-3 son la lectura, que es lo que más usó el workshop y de lo que
dependen todas las demás; la 4 y la 5 son la generación y el empuje; la 6 necesita todo lo anterior
estable; la 7 prueba el conjunto.

## 7. Decisiones (tomadas el 2026-11-05)

- **D1. Leer y empujar (15).** La probe guarda las dos direcciones: la normalización elige la de
  leer, y se empuja con la diferencia de medias cruda. Queda registrado con cuál se lee y con cuál
  se empuja, y una opción cambia la de empujar. Cambia el empuje de quien hoy usa `var`.
- **D2. El d principal con un system prompt por polo (17).** La capa se elige y el d se mide en las
  respuestas leídas sin el system prompt (con la pregunta, como respuesta del modelo); el número de
  hoy queda como otro set con nombre. Cambian la capa, el d, los z y "sigma" en esos modos. Queda
  abierta, sin proponerla ahora, una pregunta vecina: si el entrenamiento en esos modos también
  debería leer las respuestas sin el system prompt.
- **D3. El objeto nuevo.** Primero, una clase nueva `Probe` envuelta por `ConceptProbe`. Revisado
  el 2026-11-05: una sola clase, `ConceptProbe`, que se arma también desde activaciones y necesita
  el modelo cargado solo para leer textos, generar y empujar; sus campos se simplificaron.
- **D4. Disco.** Todas las funciones, las de hoy y las nuevas, devuelven el resultado completo y
  además lo guardan por defecto; se apaga por llamada o para toda la sesión. Ante la duda, guardar
  de más.
- **D5. dtype.** `"auto"` por defecto, con todo registrado: qué se pidió, qué se eligió y por qué.
- **D6. Embeddings.** Los índices de capa no se mueven (la capa l sigue siendo la salida del bloque
  l); los embeddings se piden aparte, con un nombre propio.
- **D7. Lo reemplazado.** Primero: sinónimos o envoltorios con aviso de obsoleto durante una versión.
  Revisado el 2026-11-05 (la librería es para usuarios nuevos y el único usuario hasta ahora es el
  autor): sin sinónimos. `use_4bit` y `direction_normalization` ya no existen como opciones; las
  corridas guardadas con ellos se siguen cargando, traducidas al leerlas. Las clases viejas de
  empuje y de trasplante también se sacan (etapa 4).
- **D8. "Mismos resultados"** quiere decir iguales salvo redondeo. Leer y generar van en lotes por
  defecto; los tests de referencia comparan con tolerancia; lotes de 1 reproducen lo de hoy exacto.

## 8. Lo que no se hace (aprobado el 2026-11-05)

- Un ejecutor de baterías para varios modelos: con un modelo ya cargado (capa 0) y una función chica
  de profundidad relativa alcanza; va como receta en la documentación.
- Agregar pandas como dependencia: las tablas son dataclasses con columnas de numpy, convertibles a
  registros.
- Inferir el apareamiento por el largo (13) e insertar un prompt oculto para el rol de respuesta
  (12).
- Multiclase y subespacios dentro de la probe de dos polos: van en una clase aparte, en la etapa 6.

## Anexo: cada entrada, su capa y su etapa

| # | Capa | Etapa | Cómo la resuelve el diseño |
|---|---|---|---|
| 1 | 1 (y 0) | 0 | chequeo de finitos al capturar, error con dtype y capa; aviso de fp16 con Gemma 3 |
| 2 | 0 | 1 | `dtype="auto"` por defecto, resuelto, registrado con su motivo (D5) |
| 3 | 0 | 1 | `quantization` (nivel o config) en la identidad; `use_4bit` desaparece |
| 4 | 0 | 1 | todo acepta un modelo ya cargado, incluido `ProbeWorkspace` |
| 5 | 1 | 1 | lectura en lotes con padding; float16 sólo en la caché |
| 6 | 0 | 0 | resolver la ruta local sin red |
| 7 | empaquetado | 0 | sacar la condición de plataforma |
| 8 | 0 | 0 | `dtype` con caída a `torch_dtype` |
| 9 | 0 | 1 | parche con chequeo de versión, sólo si HQQ va por nombre |
| 10 | 6d | 6 | auditoría y ataques como módulo de datasets |
| 11 | 4 | 0 | dispositivo de la salida del bloque |
| 12 | 1 | 0 | mensaje y documentación, sin prompt oculto |
| 13 | 6a (grupos) | 3 | grupos declarados; split, bootstrap y tests apareados |
| 14 | 3 y 5 | 2 | calibraciones por contexto; crudo + contexto + motivo |
| 15 | 2 y 3 | 0 doc, 2 | dos direcciones; se empuja con la cruda, registrado y cambiable (D1) |
| 16 | 7 | 3 | curva de la mitad de reporte con `best_d`; histograma titulado |
| 17 | 6b | 3 | sets con nombre; el principal se lee sin el system prompt de cada polo (D2) |
| 18 | 6b (controles) y 7 | 3 | nulo por capa en el gráfico y en las métricas |
| 19 | 3 | 0 doc, 2 | set neutro opcional, reportado aparte del cero |
| 20 | 3 | 2 | `Probe` en memoria desde arrays |
| 21 | 5 y 7 | 4 | resultados en memoria y también en disco por defecto (D4) |
| 22 | 5 | 4 | generar en lotes también al puntuar y al empujar |
| 23 | 6c | 5 | puntuadores enchufables, barrido de dosis, mayor dosis coherente |
| 24 | 6c (controles) | 5 | n direcciones al azar del mismo largo, con su p95 |
| 25 | 1 | 1 | caché por identidad, contexto, resumen y texto |
| 26 | 3 y 6b | 2 y 3 | umbral calibrado en la probe; acierto en las métricas |
| 27 | 5 y 4 | 4 | probabilidad de respuestas dadas, con o sin intervención |
| 28 | 1 | 1 | tramos de cada mensaje expuestos en la codificación |
| 29 | 6a | 3 | intervalos y diferencias apareadas entre clasificadores |
| 30 | 5 | 4 | callback por token durante la generación |
| 31 | 6b | 3 | tabla cruzada de probes con nulo y cosenos |

| Función que falta | Capa | Etapa |
|---|---|---|
| Readout de máximo u otros agregadores, también para entrenar | 1 | 1 |
| Readout en una posición o palabra | 1 | 1 |
| Entrenar con cada token como muestra | 1 y grupos | 1 y 2 |
| Capa de embeddings | 1 | 1 (D6) |
| Logística, LDA, sin dimensiones extremas | 2 | 2 |
| Una dirección para leer y otra para empujar | 3 | 2 (= 15) |
| Un polo contra una línea de base | 2 y 3 | 2 |
| Multiclase, regresión, subespacios, circular | 2 y 3 | 6 |
| Empuje por rotación | 4 | 4 |
| Varios sets de transferencia con nombre | 6b | 3 (≈ 17) |
| Curvas de aprendizaje | 6b | 6 |
| Etiquetas mezcladas y tareas de control | 6b (controles) | 3 |
| Clasificadores de superficie junto a la probe | 6b | 6 |
| Probes de confounds y su coseno | 6b | 6 |
| Quitar confounds al entrenar (proyección, LEACE) | 2 | 6 |
| Atajo para borrar una dirección | 4 | 4 |
| Diversidad del dataset | 6d | 6 |
| Auditoría y red teaming | 6d | 6 (= 10) |
| Varios modelos y profundidad relativa | receta | 6 |
| Cuantización configurable | 0 | 1 (= 3) |
