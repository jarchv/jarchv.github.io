---
layout: post
title: Modelo de Difusión:&nbsp;El modelo detrás de Sora
---

| ![_config.yml]({{ site.baseurl }}/images/sora.png) | 
|:--:| 
| *Figura 1. Imagen extraída de un video generador por  <a href="https://openai.com/sora">Sora</a> de OpenAI.* |
  
Destruir datos para generar nuevos datos. Esa es la premisa detrás de un modelo de difusión, el modelo detrás de Sora---el generador de videos a partir de texto, de OpenAI. En este post, exploraremos el funcionamiento y las aplicaciones de un modelo de difusión.

## ¿Qué es un modelo de difusión?

Un modelo de difusión es un modelo generativo que utiliza dos procesos para generar datos sinéticos. En campo de las imágenes, el primer proceso destruye los datos de la imagen original. El segundo proceso genera una nueva imagen a partir de los datos destruidos.

El proceso de destrucción es conocido como difusión. En el campo de las imágenes, la difusión se logra mediante la adición de ruido a la imagen original. Este proceso se realiza de manera gradual, es decir, se agrega ruido a la imagen original en pequeñas cantidades. Cada vez que se agrega ruido a la imagen original, se obtiene una imagen con ruido nueva. Con lo cual se obtiene varias versiones de la imagen original con ruido. El proceso de difusión se repite hasta que la imagen original se destruye por completo.

| ![_config.yml]({{ site.baseurl }}/images/diffusion.png) | 
|:--:| 
| *Figura 2. Imagen extraída de un tutorial impartirdo en el CVPR 2022.* |


Una vez que se tienen las versiones de la imagen original con ruido, el modelo es entrenado con el propósito reconstruir la imagen original a partir de las versiones con ruido.

## ¿Como se entrena un modelo de difusión?

Supongamos que la imagen original es $x_0$. El proceso de difusión genera una secuencia de imágenes con ruido $x_1, x_2, x_3, \ldots, x_T$. Donde la imagen $x_{t}$ se genera a partir de la imagen $x_{t-1}$ y el ruido $\epsilon_t$ de la siguiente manera:

$$x_{t} = x_{t-1} \cdot \sqrt{1 - \beta_t} + \beta_t \cdot \epsilon_t$$

Es decir, $x_{t}$ tendrá más ruido que $x_{t-1}$. En cada iteración $t$, $\beta_t$ se incrementa en pequeñas cantidades hasta el valor de 1. Note que el primer término de la ecuación anterios contiene la información de la imagen original. De esta manera, se obtiene una secuencia de imágenes con ruido que se destruye gradualmente.

Entonces, el modelo de difusión es entrenado para predecir el ruido $\epsilon_t$ a partir de la imagen con ruido $x_t$. Es decir, el modelo de difusión es entrenado para predecir el ruido que se le agregó a la imagen $x_{t-1}$. El modelo se entrena para distintas versiondes de la imagen con ruido, es decir, para distintos valores de $t$.

## ¿Como se genera una imagen en un modelo de difusión?

Una vez que el modelo de difusión $\mathcal{F}$ ha aprendido a predecir el ruido $\epsilon_t$ a partir de una imagen con ruido $x_t$, ahora ya es capaz de generar una imagen. El proceso de generación de una imagen empieza con $x_T$---ruido puro. El modelo empieza prediciendo el ruido $\epsilon_{T}$:

$$\epsilon_T=\mathcal{F}(x_T)$$

Luego, utilizando $\epsilon_T$ y $\mathcal{F}(x_T)$ se obtiene la imagen con ruido $x_{T-1}$. El proceso se repite hasta obtener la imagen original $x_0$.

## Modificaciones

El modelo de difusión que mostró un rendimiento sobresaliente en la generación de imágenes fue en 2020. Desde entonces, el modelo ha sufrido varias modificaciones. Originalmente el modelo opera en el dominio de las imágenes, lo cual puede ser costoso si la imagen es de alta resolución. Un [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) (LDM) primero codifica la imagen en un espacio latente---de menor dimensión---y luego aplica el modelo de difusión. El modelo de difusión es entrenado para predecir el ruido en el espacio latente. De esta manera, el modelo de difusión es capaz de generar imágenes de alta resolución de manera más eficiente.

El modelo de difusión es un red encoder-decoder, compuesta principalmente por redes convolucionales. Un [Diffusion Transformer](https://arxiv.org/abs/2212.09748) (DiT) utiliza un Transformer para predecir el ruido en el espacio latente en un LDM. Este modelo es la arquitectura detrás de [Sora](https://openai.com/sora), el generador de videos de OpenAI.