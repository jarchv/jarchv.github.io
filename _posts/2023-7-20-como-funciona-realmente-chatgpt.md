---
layout: post
title: ¿Cómo funciona realmente ChatGPT?
---

| ![_config.yml]({{ site.baseurl }}/images/chatgpt.jpg) | 
|:--:| 
| *Figura 1. Imagen de <a href="https://unsplash.com/@maria_shalabaieva?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mariia Shalabaieva</a> en <a href="https://unsplash.com/es/fotos/nYSdjVD2ayo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>* |



  
A la fecha de hoy, la Inteligencia Artificial a escalado a un nivel tal, que es difícil de predecir sus futuras aplicaciones. Desde la predicción de la estructura tridimensional de proteínas con Alfafold, hasta la **generación** de contenido con [ChatGPT](https://openai.com/chatgpt) de [OpenAI](https://openai.com). Note el lector que he resaltado la palabra 'generación' debido a que este proceso es una característica especial de los modelos generativos, categoría a la cual pertenece ChatGPT.

ChatGPT fue diseñado para seguir instrucciones. En realidad, este *chatbot* es una modificación de [GPT-3](https://arxiv.org/abs/2005.14165)&mdash;el modelo base. GPT-3 es un modelo *Auto-Regresivo* con una gran cantidad de parámetros, 175 billones de parámetros para ser exacto. Un modelo Auto-Regresivo o *Deep Autoregressive Model* es un tipo especial de modelo Generativo, propuesto inicialmente en PixelCNN para generar los pixeles de una imagen de manera secuencial.

Pero retrocedamos un poco, hacia los modelos generativos. Para explicar que es un modelo generativo, me gustaría citar a un [paper](https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html) publicado en el 2001 por Andrew Ng y Michael Jordan. En este trabajo se habla de dos tipos de *Clasificadores*:

* Discriminativos 
* Generativos

Para esto, el objetivo de un Clasificador es predecir la etiqueta correcta para una determinada muestra. Por ejemplo, predecir que objeto se encuentra dentro de una imagen. Entonces, un clasificador discriminativo se encarga de predecir la etiqueta de una muestra de manera directa, a través de una red neuronal por ejemplo. Mientras que un clasificador generativo, primero aprende como las muestras y las etiquetas están distribuidas de manera conjunta, para luego clasificar una muestra en base a su etiqueta más probable.

| ![_config.yml]({{ site.baseurl }}/images/pixelcnn.png) | 
|:--:| 
| *Figura 2. El pixel en rojo es aquel que va a ser generado a partir de la información de los pixeles en azul, estos representan los pixeles previos.* |


Al aprender la distribución conjunta, el clasificador generativo es capaz de **generar** muestras nuevas a través de una etiqueta&mdash;es esta la caracteristica del modelo generativo. En el caso específico del modelo PixelCNN, cada pixel se genera a través de la información de los pixeles previos. Es decir, para generar un pixel determinado, primero tenemos que haber generado los previos (ver Figura 2). El pixel generado será el más probable entre todos los pixeles.

## ¿Cómo procesa una palabra GPT-3?

Predecir un pixel a través de los pixeles previos es posible debido a que cada pixel almacena tres números - uno por cada canal. En el caso de la generación de texto, el objetivo es predecir la siguiente palabra a partir de las anteriores. Entonces bastaría como dar como entrada a nuestro modelo auto-regresivo una secuencia de palabras para luego generar la siguiente más probable.

Sin embargo, una palabra no puede ser procesada por una Red Neuronal. Una red neuronal esta compuesta generalmente de operaciones matriciales y funciones no-lineales las cuales transforman un conjunto de números en otro número o conjunto de números. Debido a que no podemos hacer estas operaciones con palabras, es necesario primero hacer un pre-procesamiento. 
Este paso previo es llamado Tokenización, el cual implica transformar cada palabra a un identificador numérico (ID), el cual será utilizado como entrada en el modelo GPT-3. Este ID se obtiene al obtener la frecuencia de sub-palabras en cada palabra dentro de la base de datos. Supongamos que en nuestro Corpus existe la siguiente colección de palabras:

`{menos: 8}, {mes: 9}, {tres: 7}, {tren: 8}, {trenes, 4}`

Cada palabra tendrá asociada sus repeticiones dentro del Corpus. Entonces, como primer paso, separamos cada palabra en caracteres.

* `{m, e, n, o, s: 8}`
* `{m, e, s: 9}`
* `{t, r, e, s: 7}`
* `{t, r, e, n: 8}`
* `{t, r, e, n, e, s: 4}`

Ahora el Corpus en lugar de estar conformado por palabras, estará conformado por caracteres. Lo cual nos permite obtener el nuevo vocabulario, el cual estará representado por todos los caracteres utilizados en el Corpus:

`Vocabulario: {e, m, n, o, r, s, t}`

El siguiente paso es buscar pares de caracteres en el Corpus. Por ejemplo:

* `me` se repite en total 17 veces (8 + 9)
* `tr` se repite en total 19 veces (7 + 8 + 4)
* `es` se repite en total 20 veces (9 + 7 + 4)

Finalmente, escogemos el par con la mayor cantidad de repeticiones y lo añadimos a nuestro vocabulario. Esto también modificará nuestro Corpus:

* `Vocabulario: {e, m, n, o, r, s, t, es}`
* `Corpus: {m, e, n, o, s: 8}, {m, es: 9}, {t, r, es: 7}, {t, r, e, n: 8}, {t, r, e, n, es: 4}`

Repetimos el mismo proceso, ahora el par de caracteres con mayor repeticiones es `tr`, por lo tanto el vocabulario ahora estará conformado por los siguientes tokens o sub-palabras:

* `Vocabulario: {e, m, n, o, r, s, t, es}`
* `Corpus: {m, e, n, o, s: 8}, {m, es: 9}, {tr, es: 7}, {tr, e, n: 8}, {tr, e, n, es: 4}`

Este proceso se repite hasta alcanzar una determinada cantidad de tokens en nuestro vocabulario. Supongamos que ahora queremos tokenizar la palabra `resto`. Entonces el resultado sería:

<pre>
<code> 
 {r, es, t, o}
 {r, es, t, o}
</code>
</pre>

En GPT-3 el vocabulario base son todos los caracteres posibles representados por Bytes&mdash;a esta técnica se la llama [Byte-Level BPE](https://research.facebook.com/publications/neural-machine-translation-with-byte-level-subwords/). Una vez que podemos tokenizar un texto, el identificador  numérico para cada token se da en base al índice en el vocabulario.

## ¿Qué tipo de red neuronal es la encargada de generar texto?

En GPT-3, la red neuronal encargada de la generar la siguiente palabra a partir de un texto es el Transformer. Inicialmente el modelos transforma el identificador en un Positional Enbedding (PE). El PE representa un vector de números reales, el cual permite dar un vector identificador teniendo en cuenta el orden de cada palabra en el texto de entrada. GPT-3 utiliza únicamente el Decoder para generar la siguiente palabra en base a una secuencia previa (el bloque de la derecha en la Figura 3).