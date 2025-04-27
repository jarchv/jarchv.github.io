---
layout: post
title: Modelo de Difusión:&nbsp;El modelo detrás de Sora
---

| ![_config.yml]({{ site.baseurl }}/images/sora.png) | 
|:--:| 
| *Imagen extraída de un video generado por  <a href="https://openai.com/sora">Sora</a> de OpenAI.* |
  
Convertir datos en ruido es fácil, Convertir ruido en datos es el trabajo de un modelo generativo. Estos modelos transforman el ruido aparente en información útil y procesable, abriendo nuevas oportunidades para la innovación en una amplia gama de disciplinas, desde el arte, la música hasta la medicina. 

Los modelos de difusión representan un tipo de modelo generativo, que se enfoca en modelar la distribución de datos de manera efectiva transformando gradualmente ruido en datos. Este enfoque no solo permite la generación de imágenes, texto o sonido convincentes, sino también la generación de videos realistas---Sora, el generador de videos de OpenAI, es un ejemplo de esto. En esencia, los modelos de difusión ofrecen una poderosa herramienta para convertir el ruido aparente en conocimiento significativo, abriendo nuevas fronteras en el mundo de la inteligencia artificial. En este post, exploraremos el funcionamiento y las aplicaciones de un modelo de difusión.

## ¿Qué es un modelo de difusión?

Un modelo de difusión es un modelo generativo que utiliza tres etapas para generar datos sintéticos. Para ello, ubiquémonos en el campo de las imágenes. La primera etapa consistiría en destruir la imagen original de manera gradual, es decir, poco a poco. La forma más común de destruir una imagen es agregando ruido. Un modelo de difusión agrega pequeñas cantidades de ruido a la imagen original de manera gradual hasta que esta se destruye por completo. 

| ![_config.yml]({{ site.baseurl }}/images/diffusion.png) | 
|:--:| 
| *Imagen extraída de un tutorial impartido en el CVPR 2022.* |

La segunda etapa consiste en entrenar un modelo para reconstruir la imagen original a partir de las versiones de la imagen original con ruido. Para ello, el modelo de difusión es entrenado para predecir los distintos ruidos aleatorios que se le agregaron a la imagen original hasta que esta se destruyó por completo. 

Una vez que el modelo ha sido entrenado, este es capaz de generar una imagen a partir de ruido puro. Finalmente, la última etapa consiste en predecir el ruido en cada versión de la imagen con ruido, incluso antes de destruirse por completo. De esta manera, poco a poco se va construyendo una imagen realista.

## Ahora de manera más formal

Supongamos que la imagen original es $x_0$. El proceso de difusión genera una secuencia de imágenes con ruido $x_1, x_2, x_3, \ldots, x_T$. Donde la imagen $x_{t}$ se genera a partir de la imagen $x_{t-1}$ y el ruido aleatorio $\epsilon$ de la siguiente manera:

$$x_{t} = x_{t-1} \cdot \sqrt{1 - \beta_t} + \beta_t \cdot \epsilon$$

Es decir, $x_{t}$ tendrá más ruido que $x_{t-1}$. En cada iteración $t$, $\beta_t$ se incrementa en pequeñas cantidades desde un valor $\beta_1=10^{-4}$ hasta el valor de $\beta_T=0.02$ (ver el [paper original](https://arxiv.org/abs/2006.11239) para más detalle). Note que el primer término de la ecuación anterior contiene la información de la imagen original, mientras que el segundo término contiene el ruido aleatorio---usualmente ruido gaussiano. Entonces, en cada iteración, la información de la imagen original se va perdiendo poco a poco. 

### Entrenamiento

Una vez generada la secuencia de imágenes con ruido, el modelo de difusión es entrenado para predecir $\epsilon$ a partir de la imagen $x_t$. El objetivo es predecir el ruido que se le agregó a la imagen $x_{t-1}$ para obtener $x_t$. El entrenamiento se realiza para distintas versiones de la imagen con ruido, es decir, para distintos valores de $t$. 

Para el entrenamiento se utiliza una red neuronal $\mathcal{F}$ que toma como entrada la imagen con ruido $x_t$ y la iteración $t$, para luego retornar el ruido $\epsilon$ minimizando la diferencia entre el ruido real y el ruido predicho:

$$||\epsilon-\mathcal{F}(x_t, t)||$$

En la práctica, se minimiza este error para un bloque de imágenes con ruido:

$$\mathbb{E}_{x_t,t,\epsilon}||\epsilon-\mathcal{F}(x_t, t)||$$

En la práctica, existe una manera rápida de calcular directamente $x_t$ a partir de $x_0$ sin tener que calcular todas las imágenes intermedias. Es decir no es necesario calcular $x_1, x_2, \ldots, x_{t-1}$ para obtener $x_t$. Esto permite generar cualquier versión de la imagen con ruido de manera rápida. Puede encontrar más información sobre este método en el [paper](https://arxiv.org/abs/2006.11239).

### Generación de datos

Una vez que el modelo de difusión ha aprendido a predecir $\epsilon$ a partir de una imagen con ruido $x_t$, ahora ya es capaz de generar una imagen. El proceso de generación de una imagen empieza con $x_T$---ruido puro---y el valor de $T$. El modelo captura estos dos valores como entrada y genera el ruido $\hat{\epsilon}$:

$$\hat{\epsilon}=\mathcal{F}(x_T, T)$$

Podemos considerar $\hat{\epsilon}$ como el ruido que se le agregó a la imagen $x_{T-1}$ para obtener la imagen con ruido $x_T$. Con lo cual, si quitamos el ruido $\hat{\epsilon}$ de $x_T$ obtenemos $x_{T-1}$---una versión con menos ruido. Poco a poco el modelo de difusión ira sustrayendo ruido y agregando información. Hasta finalmente generar una imagen $x_0$, realista y semánticamente significativa con respecto a los datos de entrenamiento.

## Modificaciones

En el año 2020, un [artículo](https://arxiv.org/abs/2006.11239) demostró que los modelos de difusión eran capaces de generar imágenes realistas. Desde entonces, el modelo ha sufrido varias modificaciones. Originalmente, el modelo operaba en el dominio de las imágenes, lo cual puede ser costoso si la imagen es de alta resolución. Un [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) (LDM) primero codifica la imagen en un espacio latente de menor dimensión---comprimiendo la imagen de manera espacial. Luego se aplica el modelo de difusión en el espacio latente. De esta manera, se reduce el costo computacional, permitiendo una generación eficiente de imágenes de alta resolución.

La arquitectura del modelo de difusión original es un red encoder-decoder, compuesta principalmente por redes convolucionales. Un [Diffusion Transformer](https://arxiv.org/abs/2212.09748) (DiT) utiliza un *Transformer* para predecir el ruido en el espacio latente de un LDM. Este modelo es la arquitectura detrás de [Sora](https://openai.com/sora), el generador de videos de OpenAI. En el caso de Sora, la compresión se realiza de manera espacial y también temporal. El DiT permite a Sora trabajar con videos de distintas resoluciones y duraciones. Permitiendo a Sora generar videos para múltiples tipos de dispositivos y aplicaciones.

## Ventajas y desventajas

En el año 2021, un [artículo](https://arxiv.org/abs/2105.05233) demostró que los modelos de difusión eran capaces de generar imágenes incluso más realistas que los modelos GANs. Ubicandose como el estado del arte en la generación de imágenes. Sin embargo, dado a que un modelo de difusión genera imágenes progresivamente, la etapa de generación es más lenta que la de un modelo GAN. Para hacernos una idea, un modelo de difusión puede requerir hasta 1000 iteraciones para generar una imagen realista. Mientras que un modelo GAN puede generar una imagen en una sola iteración.


 Esto hizo que los investigadores buscaran maneras de resolver este problema. Modelos como el [DDIM](https://arxiv.org/abs/2010.02502) o el [ADD](https://arxiv.org/abs/2311.17042) aceleran el tiempo durante la generación de imágenes. Reduciendo el número de iteraciones necesarias para generar una imagen a 10 o menos. Sin embargo, hay que tener en cuenta que estos modelos sacrifican algo de calidad en las imágenes generadas.