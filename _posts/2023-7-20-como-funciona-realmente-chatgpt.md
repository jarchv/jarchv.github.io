---
layout: post
title: ChatGPT puede razonar su respuesta con esta simple instrucción
---

| ![_config.yml]({{ site.baseurl }}/images/chatgpt.jpg) | 
|:--:| 
| *Figura 1. Imagen de <a href="https://unsplash.com/@maria_shalabaieva?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Mariia Shalabaieva</a> en <a href="https://unsplash.com/es/fotos/nYSdjVD2ayo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>.* |
  
Hoy en día, se escucha hablar de Inteligencia Artificial (IA) en múltiples plataformas, y no necesariamente por personas dentro del ambito científico. El torrente de aplicaciones que ha traido esta tecnología es realmente impactante. Irrumpiendo en campos como el diseño gráfico, programación, medicina, y la vida cotidiana misma. Sin embargo, podría asegurar que la IA más disruptiva fue [ChatGPT](https://openai.com/chatgpt) de [OpenAI](https://openai.com). Y es que he escuchado hablar de ChatGPT de personas que no son del ámbito científico, personas que simplemente le encontraron una utilidad en su trabajo, tarea, discurso o inclusivo para pregunta triviales.

Hace algunos años se jugaba con la idea de que la IA podía reemplazar el trabajo humano, pero no se tomó tan en serio como ahora. La industría, la educación y incluso la ciencia no pudo preveer este gigante tecnológico&mdash;descartemos a algunos investigadores dentro del campo del *Deep Learning*. Podemos agrupar las capacidades de ChatGPT en la de **generación** de contenido, note el lector que he resaltado la palabra 'generación' debido a que esta es una característica especial de la *IA Generativa*, categoría a la cual pertenece ChatGPT.

## ChatGPT es un modelo Generativo

ChatGPT fue desarrollo con el mismo objetivo que [InstructGPT](https://arxiv.org/abs/2203.02155)&mdash;para seguir instrucciones. El entrenamiento de ChatGPT involucró un grupo de personas que proporcionaron preguntas y diálogos para guiar al modelo en su aprendizaje. Estos datos de entrenamiento fueron utilizados para mejorar la capacidad de ChatGPT para responder de manera coherente, adecuada y versátil a diversas consultas.   

Pero retrocedamos un poco, ¿Qué es realmente ChatGPT?¿Qué tecnología esta detrás de ChatGPT? El esqueleto de ChatGPT es un Large Languaje Model (LLM), cuyo objetivo es generar la siguiente palabra a partir de las anteriores. Por lo tanto, podemos decir que ChatGPT es técnicamente un modelo *Generativo*. Entonces, ¿Que es un modelo Generativo?

Para explicar este modelo, me gustaría citar a un [paper](https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html)&mdash;publicado en el 2001 por Andrew Ng y Michael Jordan. En este trabajo se habla de dos tipos de *Clasificadores*, el Discriminativo y el Generativo. Para esto, el objetivo de un clasificador es predecir la etiqueta correcta para una determinada muestra; por ejemplo, predecir que objeto se encuentra dentro de una imagen. 

Entonces, un clasificador Discriminativo se encarga de predecir la etiqueta de una muestra de manera directa, a través de una red neuronal, por ejemplo. Mientras que un clasificador Generativo, primero aprende como las muestras y las etiquetas están distribuidas de manera conjunta, para luego clasificarlas en base a su etiqueta más probable. Al aprender la distribución conjunta, el clasificador generativo es capaz de **generar** muestras nuevas a través de una etiqueta&mdash;es esta la característica de un modelo Generativo. Si sabemos como están distribuidos los rostros humanos&mdash;donde va la boca, los ojos, la nariz el cabello&mdash;podemos dibujar un nuevo rostro en base a alguna etiqueta o categoría. Un hombre con lentes y barba, por ejemplo.

## ¿Cómo funciona ChatGPT?

El LLM detrás de ChatGPT recibe como entrada una intrucción, oración incompleta, pregunta, etc. Es decir una secuencia de palabras&mdash;o letras, o números o símbolos. Sin embargo, una palabra o secuencia de letras no puede ser procesada por el LLM de forma directa&mdash;se requieren números. Para realizar este proceso, es necesario realizar la *Tokenización*, proceso por el cual se transformar un texto en una secuencia de identificadores numéricos (IDs).

### Tokenización
Para explicar este proceso utilizaremos un ejemplo muy simple. Supongamos que en todo el universo de palabras que el modelo ha visto (Corpus) existen las siguientes:

`{menos: 8}, {mes: 9}, {tres: 7}, {tren: 8}, {trenes, 4}`

Cada palabra tiene asociada sus repeticiones dentro del Corpus. Entonces, como primer paso, separamos cada palabra en caracteres.

* `{m,e,n,o,s: 8}`
* `{m,e,s: 9}`
* `{t,r,e,s: 7}`
* `{t,r,e,n: 8}`
* `{t,r,e,n,e,s: 4}`

Ahora el Corpus en lugar de estar conformado por palabras, estará conformado por caracteres. Lo cual nos permite obtener el *Vocabulario*, el cual estará representado por todos los caracteres utilizados en el Corpus:

`Vocabulario: {e,m,n,o,r,s,t}`

El siguiente paso es buscar pares de caracteres en el Corpus. Por ejemplo:

* `me` se repite en total 17 veces (8 + 9)
* `tr` se repite en total 19 veces (7 + 8 + 4)
* `es` se repite en total 20 veces (9 + 7 + 4)

Finalmente, escogemos el par con la mayor cantidad de repeticiones y lo añadimos a nuestro vocabulario. Esto también modificará nuestro Corpus:

* `Vocabulario: {e,m,n,o,r,s,t,es}`
* `Corpus: {m,e,n,o,s: 8}, {m,es: 9}, {t,r,es: 7}, {t,r,e,n: 8}, {t,r,e,n,es: 4}`

Repetimos el mismo proceso, ahora el par de caracteres con mayor repeticiones es `tr`, por lo tanto el vocabulario ahora será el siguiente:

* `Vocabulario: {e,m,n,o,r,s,t,es,tr}`
* `Corpus: {m,e,n,o,s: 8}, {m,es: 9}, {tr,es: 7}, {tr,e,n: 8}, {tr,e,n,es: 4}`

Este proceso se repite hasta alcanzar una determinada cantidad de sub-palabras o *Tokens* en nuestro vocabulario. Supongamos que ahora queremos tokenizar la palabra `en estreno`. El resultado sería `{e,n,es,tr,e,n,o}`, obteniendo cinco tokens&mdash;así es como la red neuronal *visualiza* la palabra `estreno`. Cada token es único y tiene asignado un ID. En ChatGPT, este proceso ocurre de la misma forma cada vez que ingresamos una instrucción. Es decir, ChatGPT no procesa palabra por palabra, sino Token por Token. 

Para nuestro ejemplo, el vocabulario base o inicial estaba compuesto por siete caracteres (`e,m,n,o,r,s,t`). En ChatGPT, el vocabulario base son todos los caracteres posibles representados por Bytes&mdash;a esta técnica se le llama [Byte-Level BPE](https://research.facebook.com/publications/neural-machine-translation-with-byte-level-subwords/).

### ChatGPT genera un Token a la vez

ChatGPT es un modelo *Auto-Regresivo* con una gran cantidad de parámetros, 175 billones de parámetros para ser exacto. Un modelo Auto-Regresivo o *Deep Autoregressive Model* es un tipo especial de modelo generativo. El LLM es un [Transformer](https://arxiv.org/abs/1706.03762), una red neuronal encargada de la generar el siguiente Token&mdash;que podría ser una palabra completa&mdash;a partir de un texto previo. 

Antes de generar el nuevo Token, el modelo convierte el Token en un Positional Enbedding (PE). Un vector de números reales el cual da una identificación única al Token, teniendo en cuenta su orden en el texto de entrada. Enntonces la secuencia de palabras se convierte primero en una suencia de vectores indeficadores.

Esta secuencia de vectores se instertan en el Transformer, luego el modelo genera el siguiente Token&mdash;este debe ser la más probable. Al generar Token, estos tambien pueden incluir espacios, saltos de linea (para termina un parrafo), puntos (para terminar una oración) u otro tipo de símbolo. Cada vez que el Transformer genera un nuevo Token este formará parte de la entrada en la siguiente interación, de esta formar se puenten generar oraciones, un parrafos completo o incluso varios párrafos.

## Contextualiza tu instrucción

