---
layout: post
title: Del Texto al Token:&nbsp;El papel de la Tokenización en ChatGPT
---

| ![_config.yml]({{ site.baseurl }}/images/chatgpt.jpg) | 
|:--:| 
| *Figura 1. Imagen de <a href="https://unsplash.com/@maria_shalabaieva?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mariia Shalabaieva</a> en <a href="https://unsplash.com/es/fotos/nYSdjVD2ayo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>.* |
  
Hoy en día, se escucha hablar de Inteligencia Artificial (IA) en múltiples plataformas, y no necesariamente por personas dentro del ámbito científico. El torrente de aplicaciones que a traído esta tecnología es realmente impactante; irrumpiendo en campos como el diseño gráfico, programación, medicina, y la vida cotidiana misma. Dentro de todas estas aplicaciones, [ChatGPT](https://openai.com/chatgpt) se ha destacado como una de las más disruptivas. Y es que ChatGPT ha llevado la interacción humano-máquina a un nuevo nivel, permitiendo que una máquina pueda responder de manera coherente a una pregunta o instrucción. Resolviendo tareas en cualquier contexto, desde la programación hasta la cocina. En este blog, nos sumergiremos en un paso esencial dentro del funcionamiento de ChatGPT: la *Tokenización*. Exploraremos en detalle cómo este proceso descompone el texto en unidades más manejables, permitiendo que ChatGPT comprenda una instrucción y genere una respuesta coherente.

## ChatGPT es un modelo Generativo

ChatGPT fue desarrollo con el mismo objetivo que [InstructGPT](https://arxiv.org/abs/2203.02155)&mdash;para seguir instrucciones. Teniendo como base el modelo [GPT](https://arxiv.org/abs/2005.14165)(Generative Pre-trained Transformer), ChatGPT se adaptó para recibir instrucciones. Este proceso involucró a un grupo de personas que proporcionaron preguntas y diálogos para guiar al modelo en su aprendizaje. El resultado fue un modelo capaz de generar respuestas coherentes y contextualmente relevantes.

Pero retrocedamos un poco, ¿Qué es realmente ChatGPT y qué tecnología esta detrás de ChatGPT? ChatGPT es un Large Languaje Model (LLM), cuyo objetivo es generar la siguiente palabra a partir de las anteriores&mdash;pronto veremos que no se limita solo a palabras. Por lo tanto, podemos categorizar a ChatGPT como un modelo *Generativo*. 

Entonces, ¿Qué es un modelo Generativo? Permítanme explicarles utilizando un [paper](https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html) publicado en el 2001 por Andrew Ng y Michael Jordan. En este trabajo se habla de dos tipos de *Clasificadores*, el Discriminativo y el Generativo. Para esto, el objetivo de un clasificador es predecir la etiqueta correcta para una determinada muestra; por ejemplo, predecir que objeto se encuentra dentro de una imagen. 

Entonces, un clasificador Discriminativo se encarga de predecir la etiqueta de una muestra de manera directa, a través de una red neuronal, por ejemplo. Mientras que un clasificador Generativo, primero aprende como las muestras y las etiquetas están distribuidas de manera conjunta, para luego clasificarlas según su etiqueta más probable. Al aprender la distribución conjunta, el clasificador generativo es capaz de **generar** muestras nuevas a través de una etiqueta&mdash;es esta la característica de un modelo Generativo. Por ejemplo, si sabemos como están distribuidos los rostros humanos (donde va la boca, los ojos, la nariz el cabello) podemos dibujar un nuevo rostro en base a alguna etiqueta&mdash;si, por ejemplo, queremos dibujar un hombre con lentes y barba.

Esta misma propiedad es lo que hace que ChatGPT pueda aprender como están distribuidas las palabras, saber, por ejemplo, que va antes y después de la palabra `caminando`. Al aprender de grandes base de datos, ChatGPT tiene la capacidad de generar contenido coherente y contextualmente relevante.

## ChatGPT lee Tokens, no palabras

ChatGPT es un ejemplo de Modelo de Lenguaje de Gran Escala (LLM), que como parte de su funcionamiento utiliza la técnica de Tokenización, un método para generar texto coherente y perspicaz en respuesta a instrucciones o preguntas. Al recibir como entrada una instrucción, oración incompleta, pregunta o cualquier otra secuencia de palabras, letras, números o símbolos (caracteres), el modelo de lenguaje de ChatGPT no puede procesarla directamente en su forma original. Es en este punto donde entra en juego la Tokenización, proceso que transformar un texto en una secuencia de identificadores numéricos (IDs).

### Tokenización

La Tokenización descompone el texto en unidades más pequeñas, llamadas *Tokens*, permitiendo al modelo comprender y procesar un texto. Cada Token representa una entidad semántica con significado propio&mdash;cada Token tiene asignado un ID único. Para explicar este proceso utilizaremos un ejemplo muy simple. Supongamos que en todo el universo de palabras que el modelo ha visto (Corpus) existen las siguientes:

<div class="example">
<pre>
{menos: 8}, {mes: 9}, {tres: 7}, {tren: 8}, {norma, 4}
</pre>
</div>

Cada palabra tiene asociada sus repeticiones dentro del Corpus. Entonces, como primer paso, separamos cada palabra en caracteres.

<div class="example"><pre>
{m,e,n,o,s: 8}, {m,e,s: 9}, {t,r,e,s: 7}, {t,r,e,n: 8}, {n,o,r,m,a: 4}
</pre></div>

Ahora el Corpus en lugar de estar conformado por palabras, estará conformado por caracteres. Lo cual nos permite obtener el *Vocabulario*, el cual estará representado por todos los caracteres&mdash;Tokens de un caracter&mdash;utilizados en el Corpus:

<div class="example"><pre>
Vocabulario: {e,m,n,o,r,s,t,a}
</pre></div>

Ahora, supongamos que nuestro objetivo es generar un vocabulario de 10 Tokens, entonces el siguiente paso es buscar pares de caracteres en el Corpus. Por ejemplo:

* `me` se repite en total 17 veces (8 + 9)
* `es` se repite en total 16 veces (9 + 7)
* `tr` se repite en total 15 veces (7 + 8)


Finalmente, escogemos el par con la mayor cantidad de repeticiones y lo añadimos a nuestro vocabulario

<div class="example"><pre>
Vocabulario: {e,m,n,o,r,s,t,a,me}
</pre></div>

Esto también modificará nuestro Corpus:

<div class="example"><pre>
{me,n,o,s: 8}, {me,s: 9}, {t,r,e,s: 7}, {t,r,e,n: 8}, {n,o,r,m,a: 4}
</pre></div>


Repetimos el mismo proceso, ahora el par con mayor repeticiones es `tr`, por lo tanto el vocabulario ahora será el siguiente:

<div class="example"><pre>
Vocabulario: {e,m,n,o,r,s,t,a,me,tr}
</pre></div>

Y el Corpus:

<div class="example"><pre>
{me,n,o,s: 8},{me,s: 9},{tr,e,s: 7},{tr,e,n: 8},{n,o,r,m,a: 4}
</pre></div>

El vocabulario tiene ahora 10 elementos&mdash;estos son los Tokens o sub-palabras. Ahora la Tokenizacion procesa el texto de entrada, descomponiéndolo en Tokens. Por ejemplo, para la siguiente instrucción: 

<div class="example"><pre>
Otra tormenta
</pre></div>

Obtendremos los siguiente Tokens:

<div class="example"><pre>
{o,tr,a,t,o,r,me,n,t,a}
</pre></div>

Así es como la red neuronal *visualiza* el texto de entrada. En ChatGPT, este proceso ocurre de la misma forma cada vez que ingresamos una instrucción. Es decir, ChatGPT no procesa palabra por palabra, sino Token por Token. 

Para nuestro ejemplo, el vocabulario base o inicial estaba compuesto por ocho caracteres. En ChatGPT, el vocabulario base son todos los caracteres posibles representados por Bytes&mdash;a esta técnica se le llama [Byte-Level BPE](https://research.facebook.com/publications/neural-machine-translation-with-byte-level-subwords/).

### ChatGPT genera un Token a la vez

El esqueleto de ChatGPT es un [Transformer](https://arxiv.org/abs/1706.03762), una red neuronal encargada de la generar el siguiente Token&mdash;que podría ser una palabra completa&mdash;a partir de un texto previo. Antes de ingresar los Token al Transformer, son transformados en vectores de números reales, estos vectores son llamados *Embeddings*. Un Embedding es la representación vectorial de un Token, la cual es aprendida durante el entrenamiento de la red neuronal.

Esta secuencia de vectores se instertan en el Transformer al mismo tiempo. Una vez que el Transformer procesa la secuencia de vectores, genera un nuevo vector, el cual es transformado en un nuevo Token&mdash;la siguiente palabra. El Transformer genera un Token a la vez, el cual es añadido a la secuencia de entrada para la siguiente iteración. El Token generado es el mas probable de aparecer a continuación del texto de entrada. Este Token puede ser un espacio, un salto de linea, un punto, una coma, un número, una palabra, etc. Este proceso se repite hasta que el Transformer genera un Token especial, el cual indica que la generación de Tokens ha terminado.  

## ChatGPT se retroalimenta de su respuesta

Dado que cada Token generado luego es añadido a la secuencia de entrada, ChatGPT se retroalimenta con información de su propia respuesta. Sin embargo, que sucede si el Token generado por ChatGPT es el más probable de aparecer a continuación, pero no es el correcto. Por ejemplo, ingrese la siguiente instrucción a ChatGPT:

<div class="example"><pre>
Instrucción: Intercambia la primera y última letra de cada palabra de la oración: "Hola Mundo"
ChatGPT    : aloh odnuM
</pre></div>

No es la respuesta correcta. Pero demosle otra oportunidad más a ChatGPT:

<div class="example"><pre>
Instrucción: Vuelve a intentarlo, la respuesta es incorrecta.
ChatGPT    : Aoll Mundh
</pre></div>

Es un hecho que ChatGPT no es capaz de resolver esta instrucción. Pero, ¿Por qué? La respuesta está en la retroalimentación. Al retroalimentarse de su solución para la primera palabra&mdash;incorrecta en este caso&mdash;ChatGPT genera la siguiente en base a esa solución. Generando una secuencia de Tokens que no resuelven la instrucción.

Pero, ¿Qué sucede si guiamos a ChatGPT con la solución a la primera palabra? Por ejemplo:

<div class="example"><pre>
Instrucción: Cual es la primera y última letra de "Hola"
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

ChaGPT ha dado con la respuesta a la primera palabra. Ahora, repitamos el mismo proceso para la segunda palabra:

<div class="example"><pre>
Instrucción: Esa es la solución para la primera palabra. Ahora cual sería la solución para la palabra "Mundo", desarrolla tu solución paso a paso.
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

Entonces, ahora ChatGPT es capaz de resolver la instrucción completa:

<div class="example"><pre>
Instrucción: Intercambia la primera y última letra de cada palabra de la oración: "Hola Mundo". Hazlo paso a paso.
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
