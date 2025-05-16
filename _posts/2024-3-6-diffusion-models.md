---
layout: post
title: Diffusion Model:&nbsp;El modelo detrás de Sora
---

| ![_config.yml]({{ site.baseurl }}/images/sora.png) | 
|:--:| 
| *Imagen extraída de un video generado por  <a href="https://openai.com/sora">Sora</a> de OpenAI.* |
  
Convertir datos en ruido es fácil, convertir ruido en datos es el trabajo de un modelo generativo. Este tipo de modelos transforman ruido en información útil y procesable, abriendo nuevas oportunidades para la innovación en una amplia gama de disciplinas, desde el arte, la música hasta la medicina. 

Los *Diffusion Models* representan un tipo de modelo generativo, que se enfoca en modelar la distribución de datos de manera efectiva transformando gradualmente ruido en datos. Este enfoque no solo permite la generación de imágenes, texto o sonido convincentes, sino también la generación de videos realistas. En este post, exploraremos el funcionamiento y las aplicaciones de uno de los modelos predilectos de las aplicaciones de Inteligencia Artificial Generativa.

## ¿Qué es un Diffusion Model?

Un Diffusion Model es un modelo generativo que utiliza tres etapas para generar datos sintéticos. Para ello, ubiquémonos en el campo de las imágenes. La primera etapa consiste en destruir la imagen original de manera gradual agregando ruido. 

| ![_config.yml]({{ site.baseurl }}/images/diffusion.png) | 
|:--:| 
| *Imagen extraída de un tutorial impartido en el CVPR 2022.* |

La segunda etapa consiste en entrenar un modelo para reconstruir la imagen original a partir de las versiones de la imagen original con ruido. Para ello, el modelo de difusión es entrenado para predecir los distintos ruidos aleatorios que se le agregaron a la imagen original hasta que esta se destruyó por completo. 

Finalmente, la última etapa consiste en predecir el ruido en cada versión de la imagen con ruido. Lo cual permite al modelo de difusión eliminar el ruido y recuperar la imagen original. Este proceso se realiza empezando desde ruido puro, y poco a poco el modelo de difusión va eliminando ruido y agregando información. Hasta finalmente generar una imagen realista y semánticamente significativa con respecto a los datos de entrenamiento.

## Vallamos a los detalles técnicos   

Supongamos que la imagen original es $x_0$. El proceso de difusión genera una secuencia de imágenes con ruido $x_1, x_2, x_3, \ldots, x_T$. Donde la imagen $x_{t}$ se genera a partir de la imagen $x_{t-1}$ y el ruido aleatorio $\epsilon$ de la siguiente manera:

$$x_{t} = x_{t-1}\sqrt{1 - \beta_t} + \beta_t \epsilon.$$

En cada iteración $t$, $\beta_t$ se incrementa en pequeñas cantidades desde un valor $\beta_1=10^{-4}$ hasta el valor de $\beta_T=0.02$ (ver el [paper original](https://arxiv.org/abs/2006.11239) para más detalle). Note que el primer término de la ecuación anterior contiene la información de la imagen original, mientras que el segundo término contiene el ruido aleatorio---usualmente ruido gaussiano. Entonces, en cada iteración, la información de la imagen original se va perdiendo poco a poco. 

### Entrenamiento

Una vez generada la secuencia de imágenes con ruido, el modelo de difusión es entrenado para predecir $\epsilon$ a partir de la imagen $x_t$. Es decir, la red neuronal tiene como objetivo predecir el ruido que fue agragado a la imagen $x_{t-1}$ para obtener $x_t$. 

El entrenamiento se realiza para distintas versiones de la imagen con ruido, es decir, para distintos valores de $t$. La red neuronal $\epsilon_{\theta}$ toma como entrada la imagen con ruido $x_t$ y la iteración $t$. Luego el objetivo de la función de pérdida es minimizar la diferencia entre la salida de  $\epsilon_{\theta}$ y el ruido $\epsilon$:

$$\mathcal{L}_{\theta}=||\epsilon-\epsilon_{\theta}(x_t, t)||=||\epsilon-\epsilon_{\theta}(x_{t-1}\sqrt{1 - \beta_t} + \beta_t \epsilon, t)||.$$

En la práctica, se minimiza este error para un bloque de imágenes con ruido:

$$\mathcal{L}_{\theta}=\mathbb{E}_{x_t,t,\epsilon}||\epsilon-\epsilon_{\theta}(x_t, t)||$$

En la práctica, existe una manera rápida de calcular directamente $x_t$ a partir de $x_0$ sin tener que calcular todas las imágenes intermedias. Es decir no es necesario calcular $x_1, x_2, \ldots, x_{t-1}$ para obtener $x_t$. Esto permite generar cualquier versión de la imagen con ruido de manera rápida.

### Generación de datos

Una vez que el modelo de difusión ha aprendido a predecir $\epsilon$ a partir de una imagen con ruido $x_t$, ahora ya es capaz de generar una imagen. El proceso de generación de una imagen empieza con $x_T$---ruido puro---y el valor de $T$. El modelo captura estos dos valores como entrada y genera el ruido $\hat{\epsilon}$:

$$\hat{\epsilon}=\epsilon_{\theta^*}(x_T, T)$$

Podemos considerar $\hat{\epsilon}$ como el ruido que se le agregó a la imagen $x_{T-1}$ para obtener la imagen con ruido $x_T$. Con lo cual, si quitamos el ruido $\hat{\epsilon}$ de $x_T$ obtenemos $x_{T-1}$---una versión con menos ruido. Poco a poco el modelo de difusión ira sustrayendo ruido y agregando información. Hasta finalmente generar una imagen $x_0$, una muestra similar a las imágenes de entrenamiento.

## Modificaciones

En el año 2020, un [artículo](https://arxiv.org/abs/2006.11239) demostró que los modelos de difusión eran capaces de generar imágenes realistas. Desde entonces, el modelo ha sufrido varias modificaciones. Originalmente, el modelo operaba en el dominio de las imágenes, lo cual puede ser costoso si estas son de alta resolución. Un [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) (LDM) primero codifica la imagen en un espacio latente de menor dimensión---comprimiendo la imagen de manera espacial. Luego al utilizar el Diffusion Model en el espacio latente reducimos el costo computacional, dado que el modelo procesa una imagen de menor tamaño

La arquitectura del modelo de difusión original es un red encoder-decoder, compuesta principalmente por *Redes Convolucionales*. Un [Diffusion Transformer](https://arxiv.org/abs/2212.09748) (DiT) utiliza un *Transformer* para predecir el ruido en el espacio latente de un LDM. Este modelo es la arquitectura detrás de [Sora](https://openai.com/sora), el generador de videos de OpenAI. En el caso de Sora, la compresión se realiza de manera espacial y también temporal. El DiT permite a Sora trabajar con videos de distintas resoluciones y duraciones. Permitiendo a Sora generar videos para múltiples tipos de dispositivos y aplicaciones.

## Ventajas y desventajas

En el año 2021, un [artículo](https://arxiv.org/abs/2105.05233) demostró que los modelos de difusión eran capaces de generar imágenes incluso más realistas que los modelos GANs. Ubicandose como el estado del arte en la generación de imágenes. Sin embargo, dado a que un modelo de difusión genera imágenes progresivamente, la etapa de generación es más lenta que en las GANs. Para hacernos una idea, un modelo de difusión puede requerir hasta 1000 iteraciones para generar una imagen realista. Mientras que un modelo GAN genera cada imagen en una sola iteración.

 Esto hizo que los investigadores buscaran maneras de resolver este problema. Modelos como el [DDIM](https://arxiv.org/abs/2010.02502) o el [ADD](https://arxiv.org/abs/2311.17042) reducen el número de iteraciones necesarias para generar una imagen a 10 o menos. Sin embargo, hay que tener en cuenta que estos modelos sacrifican algo de calidad a cambio de velocidad.