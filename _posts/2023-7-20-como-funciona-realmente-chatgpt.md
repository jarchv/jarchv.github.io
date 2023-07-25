---
layout: post
title: Del Texto al Token:&nbsp;El papel de la Tokenización en ChatGPT
---

| ![_config.yml]({{ site.baseurl }}/images/chatgpt.jpg) | 
|:--:| 
| *Figura 1. Imagen de <a href="https://unsplash.com/@maria_shalabaieva?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mariia Shalabaieva</a> en <a href="https://unsplash.com/es/fotos/nYSdjVD2ayo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>.* |
  
Hoy en día, se escucha hablar de Inteligencia Artificial (IA) en múltiples plataformas, y no necesariamente por personas dentro del ámbito científico. El torrente de aplicaciones que a traído esta tecnología es realmente impactante; irrumpiendo en campos como el diseño gráfico, programación, medicina, y la vida cotidiana misma. Dentro de todas estas aplicaciones, [ChatGPT](https://openai.com/chatgpt) se ha destacado como una de las más disruptivas. Y es que ChatGPT ha llevado la interacción humano-máquina a un nuevo nivel, permitiendo que una máquina pueda responder de manera coherente a una pregunta o instrucción. Resolviendo tareas en cualquier contexto, desde la programación hasta la cocina. En este blog, nos sumergiremos en un paso esencial dentro del funcionamiento de ChatGPT: la *Tokenización*. Exploraremos en detalle cómo este proceso descompone el texto en unidades más manejables, permitiendo que ChatGPT comprenda una instrucción y genere una respuesta coherente.

## ChatGPT es un modelo Generativo

ChatGPT fue desarrollo con el mismo objetivo que [InstructGPT](https://arxiv.org/abs/2203.02155)&mdash;para seguir instrucciones. El entrenamiento de ChatGPT involucró un grupo de personas que proporcionaron preguntas y diálogos para guiar al modelo en su aprendizaje. Estos datos de entrenamiento fueron utilizados para mejorar la capacidad de ChatGPT para responder de manera coherente, adecuada y versátil a diversas consultas.   

Pero retrocedamos un poco, ¿Qué es realmente ChatGPT y qué tecnología esta detrás de ChatGPT? El esqueleto de ChatGPT es un Large Languaje Model (LLM), cuyo objetivo es generar la siguiente palabra a partir de las anteriores&mdash;pronto veremos que no se limita solo a palabras. Por lo tanto, podemos categorizar a ChatGPT como un modelo *Generativo*. 

Entonces, ¿Qué es un modelo Generativo? Permítanme explicarles utilizando un [paper](https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html)&mdash;publicado en el 2001 por Andrew Ng y Michael Jordan. En este trabajo se habla de dos tipos de *Clasificadores*, el Discriminativo y el Generativo. Para esto, el objetivo de un clasificador es predecir la etiqueta correcta para una determinada muestra; por ejemplo, predecir que objeto se encuentra dentro de una imagen. 

Entonces, un clasificador Discriminativo se encarga de predecir la etiqueta de una muestra de manera directa, a través de una red neuronal, por ejemplo. Mientras que un clasificador Generativo, primero aprende como las muestras y las etiquetas están distribuidas de manera conjunta, para luego clasificarlas según su etiqueta más probable. Al aprender la distribución conjunta, el clasificador generativo es capaz de **generar** muestras nuevas a través de una etiqueta&mdash;es esta la característica de un modelo Generativo. Por ejemplo, si sabemos como están distribuidos los rostros humanos (donde va la boca, los ojos, la nariz el cabello) podemos dibujar un nuevo rostro en base a alguna etiqueta&mdash;si, por ejemplo, queremos dibujar un hombre con lentes y barba.

Esta misma propiedad es lo que hace que ChatGPT pueda aprender como están distribuidas las palabras, saber, por ejemplo, que va antes y después de la palabra `caminando`. Al aprender de grandes base de datos, ChatGPT tiene la capacidad de generar contenido coherente y contextualmente relevante.

## ChatGPT lee Tokens, no palabras

ChatGPT es un ejemplo de Modelo de Lenguaje de Gran Escala (LLM), que como parte de su funcionamiento utiliza la técnica de Tokenización, un método para generar texto coherente y perspicaz en respuesta a instrucciones o preguntas. Al recibir como entrada una instrucción, oración incompleta, pregunta o cualquier otra secuencia de palabras, letras, números o símbolos (caracteres), el modelo de lenguaje de ChatGPT no puede procesarla directamente en su forma original. Es en este punto donde entra en juego la Tokenización, proceso que transformar un texto en una secuencia de identificadores numéricos (IDs).

### Tokenización

La Tokenización descompone el texto en unidades más pequeñas, llamadas *Tokens*, permitiendo al modelo comprender y procesar un texto. Cada Token representa una entidad semántica con significado propio&mdash;cada Token tiene asignado un ID único. Para explicar este proceso utilizaremos un ejemplo muy simple. Supongamos que en todo el universo de palabras que el modelo ha visto (Corpus) existen las siguientes:

<pre>
  `{menos: 8}, {mes: 9}, {tres: 7}, {tren: 8}, {norma, 4}`
</pre>

Cada palabra tiene asociada sus repeticiones dentro del Corpus. Entonces, como primer paso, separamos cada palabra en caracteres.

* `{m,e,n,o,s: 8}`
* `{m,e,s: 9}`
* `{t,r,e,s: 7}`
* `{t,r,e,n: 8}`
* `{n,o,r,m,a: 4}`

Ahora el Corpus en lugar de estar conformado por palabras, estará conformado por caracteres. Lo cual nos permite obtener el *Vocabulario*, el cual estará representado por todos los caracteres&mdash;Tokens de un caracter&mdash;utilizados en el Corpus:

<pre>
  <code>Vocabulario: {e,m,n,o,r,s,t,a}</code>
</pre>

Ahora, supongamos que nuestro objetivo es generar un vocabulario de 10 Tokens, entonces el siguiente paso es buscar pares de caracteres en el Corpus. Por ejemplo:

* `me` se repite en total 17 veces (8 + 9)
* `es` se repite en total 16 veces (9 + 7)
* `tr` se repite en total 15 veces (7 + 8)


Finalmente, escogemos el par con la mayor cantidad de repeticiones y lo añadimos a nuestro vocabulario. Esto también modificará nuestro Corpus:

<pre>
  Vocabulario: {e,m,n,o,r,s,t,a,me}
  Corpus: {me,n,o,s: 8},{me,s: 9},{t,r,e,s: 7},{t,r,e,n: 8},{n,o,r,m,a: 4}
</pre>

Repetimos el mismo proceso, ahora el par con mayor repeticiones es `tr`, por lo tanto el vocabulario ahora será el siguiente:

<pre>
  Vocabulario: {e,m,n,o,r,s,t,a,me,tr}
  Corpus: {me,n,o,s: 8},{me,s: 9},{tr,e,s: 7},{tr,e,n: 8},{n,o,r,m,a: 4}
</pre>

El vocabulario tiene ahora 10 elementos&mdash;estos son los Tokens o sub-palabras. Ahora la Tokenizacion procesa el texto de entrada, descomponiéndolo en Tokens. Por ejemplo, supongamos que la instrucción de entrada es la siguiente: 

<pre>
  Otra tormenta
</pre>

La Tokenización descompone el texto en Tokens:

<pre>
  {o,tr,a,t,o,r,me,n,t,a}
</pre>

Así es como la red neuronal *visualiza* el texto de entrada. En ChatGPT, este proceso ocurre de la misma forma cada vez que ingresamos una instrucción. Es decir, ChatGPT no procesa palabra por palabra, sino Token por Token. 

Para nuestro ejemplo, el vocabulario base o inicial estaba compuesto por ocho caracteres. En ChatGPT, el vocabulario base son todos los caracteres posibles representados por Bytes&mdash;a esta técnica se le llama [Byte-Level BPE](https://research.facebook.com/publications/neural-machine-translation-with-byte-level-subwords/).

### ChatGPT genera un Token a la vez

ChatGPT es un modelo *Auto-Regresivo* con una gran cantidad de parámetros, 175 billones de parámetros para ser exacto. Un modelo Auto-Regresivo o *Deep Autoregressive Model* es un tipo especial de modelo generativo. El LLM es un [Transformer](https://arxiv.org/abs/1706.03762), una red neuronal encargada de la generar el siguiente Token&mdash;que podría ser una palabra completa&mdash;a partir de un texto previo. 

Antes de generar el nuevo Token, el modelo convierte el Token en un Positional Enbedding (PE). Un vector de números reales el cual da una identificación única al Token, teniendo en cuenta su orden en el texto de entrada. Enntonces la secuencia de palabras se convierte primero en una suencia de vectores indeficadores.

Esta secuencia de vectores se instertan en el Transformer, luego el modelo genera el siguiente Token&mdash;este debe ser la más probable. Al generar Token, estos tambien pueden incluir espacios, saltos de linea (para termina un parrafo), puntos (para terminar una oración) u otro tipo de símbolo. Cada vez que el Transformer genera un nuevo Token este formará parte de la entrada en la siguiente interación, de esta formar se puenten generar oraciones, un parrafos completo o incluso varios párrafos.

## Contextualiza tu instrucción

