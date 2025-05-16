---
layout: post
title: El papel de la Tokenización en ChatGPT
---

| ![_config.yml]({{ site.baseurl }}/images/chatgpt.jpg) | 
|:--:| 
| *Figura 1. Imagen de <a href="https://unsplash.com/@maria_shalabaieva?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mariia Shalabaieva</a> en <a href="https://unsplash.com/es/fotos/nYSdjVD2ayo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>.* |
  
Hoy en día, la Inteligencia Artificial (IA) es un tema recurrente en diversas plataformas, y no necesariamente por personas dentro del ámbito científico. El torrente de aplicaciones que ha traído esta tecnología es realmente impactante; permeando en campos como el diseño gráfico, programación, medicina, y la vida cotidiana misma.

Entre todas estas aplicaciones, [ChatGPT](https://openai.com/chatgpt) se ha destacado como una de las más disruptivas. La aplicación insigniad de OpenAI ha llevado la interacción humano-máquina a un nuevo nivel, permitiendo a la máquina responder preguntas de manera coherente, y resolver tareas de cualquier contexto. Sin embargo, ¿Alguna vez se han preguntado cómo realmente funciona ChatGPT? ¿Cómo procesa una intrucción y generar una respuesta? En este blog, exploraremos un paso esencial dentro del funcionamiento de ChatGPT: la *Tokenización*. Analizaremos detalladamente cómo este proceso descompone el texto en unidades más manejables, permitiendo a ChatGPT comprender una instrucción y generar una respuesta coherente.

## ChatGPT es un modelo Generativo

Teniendo como base el modelo [GPT](https://arxiv.org/abs/2005.14165)(Generative Pre-trained Transformer), ChatGPT fue desarrollado con el mismo objetivo que [InstructGPT](https://arxiv.org/abs/2203.02155)&mdash;para seguir instrucciones. Este proceso involucró a un grupo de personas que proporcionaron preguntas y diálogos para guiar el entrenamiento. El resultado fue un modelo capaz de generar respuestas coherentes y contextualmente relevantes.

Pero retrocedamos un poco, ¿Qué tecnología esta detrás de ChatGPT? ChatGPT es un *Large Languaje Model* (LLM), cuyo objetivo es **predecir** la siguiente palabra a partir de las anteriores&mdash;pronto veremos que no se limita solo a palabras. Por ejemplo, si le decimos a ChatGPT:

<div class="example"><pre>
Tuvo un accidente con el
</pre></div>

Para ChatGPT, las palabras más probables que deberían seguir a la oración anterior son:

<div class="example"><pre>
Rpta #1: auto       (50%)
Rpta #2: vehículo   (35%)
Rpta #3: camión     (15%)
</pre></div>

Según estas opciones, la palabra más probable es `auto`, por lo tanto, ChatGPT escogerá esta palabra como respuesta a la oración anterior (`Tuvo un accidente con el auto`). Pero, ¿Cómo sabe ChatGPT que `auto` es la palabra más probable? La respuesta es simple, ChatGPT aprende de grandes bases de datos, las cuales contienen millones de oraciones. Durante el entrenamiento, el modelo es alimentado con una oración y su salida es un vector con la probabilidad de cada palabra en el vocabulario. Luego, una función de pérdida se encarga de incrementar la probabilidad de la palabra correcta y disminuir la probabilidad de las palabras incorrectas. Este proceso se repite millones de veces, lo que permite al modelo aprender a predecir la siguiente palabra en una oración cualquiera.


## ChatGPT lee Tokens, no palabras

Al recibir como entrada una instrucción, oración incompleta, pregunta o cualquier otra secuencia de palabras, letras, números o símbolos (caracteres), el modelo GPT no puede procesar esta información directamente. Es en este punto donde entra en juego la Tokenización.
|
La Tokenización descompone el texto en unidades más pequeñas, llamadas *Tokens*. Cada Token representa una unidad semántica con significado propio. Esta unidad tiene asignado un indetificador único o ID. Para explicar este proceso utilizaremos un ejemplo muy simple. Supongamos que en todo el universo de palabras que el modelo ha visto (Corpus) se encontró la siguiente lista de palabras:

<div class="example"><pre>
{menos: 8}, {mes: 9}, {tres: 7}, {tren: 8}, {norma, 4}
</pre></div>

Cada palabra tiene asociada sus repeticiones dentro del Corpus. Entonces, como primer paso, separamos cada palabra en caracteres.

<div class="example"><pre>
{m,e,n,o,s: 8}, {m,e,s: 9}, {t,r,e,s: 7}, {t,r,e,n: 8}, {n,o,r,m,a: 4}
</pre></div>

Ahora el Corpus en lugar de estar conformado por palabras, estará conformado por caracteres. Lo cual nos permite obtener el *Vocabulario*, el cual estará representado por todos los caracteres utilizados en el Corpus:

<div class="example"><pre>
Vocabulario: {e,m,n,o,r,s,t,a}
</pre></div>

Ahora, supongamos que nuestro objetivo es generar un vocabulario de 10 Tokens, entonces el siguiente paso es buscar pares de caracteres en el Corpus. Por ejemplo:

* `me` se repite en total 17 veces: 8 veces en `menos` y 9 en `mes`
* `es` se repite en total 16 veces: 9 veces en `mes` y 7 en `tres`
* `tr` se repite en total 15 veces: 7 veces en `tres` y 8 en `tren`

Finalmente, escogemos el par con la mayor cantidad de repeticiones y lo añadimos a nuestro vocabulario

<div class="example"><pre>
Vocabulario: {e,m,n,o,r,s,t,a,me}
</pre></div>

Esto también modificará nuestro Corpus:

<div class="example"><pre>
{me,n,o,s: 8}, {me,s: 9}, {t,r,e,s: 7}, {t,r,e,n: 8}, {n,o,r,m,a: 4}
</pre></div>

Luego, repetimos el mismo proceso. Ahora el par con mayor repeticiones es `tr`, por lo tanto el vocabulario será el siguiente:

<div class="example"><pre>
Vocabulario: {e,m,n,o,r,s,t,a,me,tr}
</pre></div>

Y el Corpus:

<div class="example"><pre>
{me,n,o,s: 8},{me,s: 9},{tr,e,s: 7},{tr,e,n: 8},{n,o,r,m,a: 4}
</pre></div>

El vocabulario tiene ahora 10 elementos (sub-palabras o Tokens). 

Finalmente, cada vez que el modelo encuentre una palabra que no esté en el vocabulario, será descompuesta en Tokens. Por ejemplo, para la siguiente instrucción: 

<div class="example"><pre>
Otra tormenta
</pre></div>

Obtendremos los siguientes Tokens:

<div class="example"><pre>
{o,tr,a,t,o,r,me,n,t,a}
</pre></div>

Así es como un LLM *visualiza* el texto de entrada. En ChatGPT. Es decir, ChatGPT no procesa palabra por palabra, sino Token por Token. 

Para nuestro ejemplo, el vocabulario base o inicial estaba compuesto por ocho caracteres. En ChatGPT, el vocabulario base son todos los caracteres posibles representados por Bytes&mdash;a esta técnica se la llama [Byte-Level BPE](https://research.facebook.com/publications/neural-machine-translation-with-byte-level-subwords/).

## Un Token a la vez

El esqueleto de ChatGPT es un [Transformer](https://arxiv.org/abs/1706.03762), una red neuronal encargada de generar el siguiente Token&mdash;que podría ser una palabra&mdash;a partir de una secuencia de Tokens. Antes de ingresar los Token al Transformer, estos son transformados en vectores de números reales llamados *Embeddings*. Un Embedding es la representación vectorial de un Token, la cual es aprendida durante el entrenamiento de la red neuronal.

Esta secuencia de vectores se insertan en el Transformer al mismo tiempo. Una vez que el Transformer procesa la secuencia de vectores, genera un nuevo vector, el cual es transformado en un nuevo Token&mdash;la siguiente palabra. Finalmente, este nuevo Token es añadido a la secuencia de entrada para luego ser nuevamente procesada por el Transformer en la siguiente iteración. Este proceso se repite, generando nuevas palabras de manera secuencial, hasta que el Transformer genera un Token especial, el cual indica que la generación de Tokens ha terminado. Este Token especial es conocido como *End-of-Sequence* (EOS).

El Transformer puede generar Tokens de todo tipo: palabras, números, símbolos, espacios, saltos de línea, puntos, comas, etc. Dado que cada Token generado es añadido luego a la secuencia de entrada, ChatGPT se retroalimenta de su propia respuesta.

Este proceso de retroalimentación es crucial para la generación de respuesta en ChatGPT. Sin embargo, también puede ser un arma de doble filo. Por ejemplo, si ChatGPT genera una palabra incorrecta, esta palabra incorrecta será utilizada como entrada para la siguiente iteración. Esto puede llevar a que el modelo genere respuestas incoherentes o incorrectas. Veamos esto con un ejemplo, supongamos que le decimos a ChatGPT:

<div class="example"><pre>
Instrucción: Intercambia la primera y última letra de cada palabra de la siguiente oración: "Hola Mundo"
</pre></div>
***
<div class="ok"><pre>
ChatGPT    : aloh odnuM
</pre></div>


No es la respuesta correcta. Pero démosle otra oportunidad más a ChatGPT:

<div class="example"><pre>
Instrucción: Vuelve a intentarlo, la respuesta es incorrecta.
</pre></div>
***
<div class="ok"><pre>
ChatGPT    : Aoll Mundh
</pre></div>

Es un hecho que ChatGPT no es capaz de resolver esta instrucción. Pero, ¿Por qué? La respuesta está en la retroalimentación. Al generar la primera palabra, ChatGPT tuvo que resolver el problema en base a alguna solución&mdash;incorrecta en este caso. Cuando ChatGPT genera la segunda palabra, esta se basa en la solución de la primera. Lo cual conlleva a que la segunda palabra también sea incorrecta.

Para solucionar este problema, podemos guiar a ChatGPT con la respuesta de la primera palabra. Por ejemplo, si le decimos a ChatGPT:

<div class="example"><pre>
Instrucción: Cuál es la primera y última letra de la palabra "Hola"
</pre></div>
***
<div class="ok"><pre>
ChatGPT    : En la palabra "Hola", la primera letra es "H" y la última letra es "a".
</pre></div>


Ahora ChatGPT ha reconocido la primera y última letra de la palabra `Hola`. Sigamos:

<div class="example"><pre>
Instrucción: En la palabra "Hola", reemplaza la primera letra por la "a".
</pre></div>
***
<div class="ok"><pre>
ChatGPT    : Si reemplazamos la "H" por "a" en la palabra "Hola", obtendríamos "aola".
</pre></div>
***
<div class="example"><pre>
Instrucción: Ahora, reemplaza la última letra por la "H".
</pre></div>
***
<div class="ok"><pre>
ChatGPT    : Si reemplazamos la última letra "a" por "H" en la palabra "aola", obtendríamos "aolH".
</pre></div>

Ahora que ChatGPT ha dado con la respuesta a la primera palabra, podemos continuar con la segunda palabra instruyendo a que repita el mismo proceso:

<div class="example"><pre>
Instrucción: Esa es la solución para la primera palabra. Ahora, ¿Cuál sería la solución para la palabra "Mundo"?. Desarrolla tu solución paso a paso.
</pre></div>
***
<div class="ok"><pre>
ChatGPT    : Vamos a aplicar el cambio paso a paso en la palabra "Mundo":
             1. La primera letra es "M".
             2. La última letra es "o".

             Ahora, vamos a intercambiar la primera y última letra:

             Reemplazamos la primera letra "M" por "o".
             Reemplazamos la última letra "o" por "M".
             Por lo tanto, al intercambiar la primera y última letra de la palabra "Mundo", obtenemos "oundM".
</pre></div>

Entonces, el punto clave aquí es la retroalimentación. Si ChatGPT no es capaz de resolver una instrucción, podemos guiarlo a través de la respuesta de la primera palabra. Esto puede ser útil para resolver instrucciones complejas. Una clave para guiar a ChatGPT es descomponer la instrucción en pasos más simples. Por ejemplo, dandole la siguiente instrucción:

<div class="example"><pre>
Instrucción: Intercambia la primera y última letra de cada palabra de la oración: "Hola Mundo". Resuelve paso a paso.
</pre></div>
***
<div class="ok"><pre>
ChatGPT    : Vamos a intercambiar la primera y última letra de cada palabra de la oración "Hola Mundo" paso a paso:

             La primera palabra es "Hola":

             Primera letra: "H"
             Última letra: "a"
             Intercambio: Reemplazo la primera "H" por "a" y la última "a" por "H". Nueva palabra: "aolH"
             La segunda palabra es "Mundo":

             Primera letra: "M"
             Última letra: "o"
             Intercambio: Reemplazo la primera "M" por "o" y la última "o" por "M". Nueva palabra: "oundM"

             Finalmente, al intercambiar la primera y última letra de cada palabra de la oración "Hola Mundo", obtenemos la nueva oración: "aolH oundM".
</pre></div>

El proceso por el cual se descompone una instrucción compleja en varias instrucciones más simples es conocido como [*Chain-of-Though*](https://arxiv.org/abs/2201.11903) (Cadena de Pensamiento). Este proceso permite el razonamiento paso a paso de un LLM, guiandolo hacia una respuesta más precisa. La idea detrás de este proceso es que al descomponer una tarea compleja en pasos más simples, el modelo puede abordar cada paso de manera más efectiva y, luego al conectar los pasos, llegar a una solución más precisa. Esto es especialmente útil en tareas que requieren razonamiento lógico o matemático, donde cada paso intermedio puede ser crucial para llegar a la respuesta final.