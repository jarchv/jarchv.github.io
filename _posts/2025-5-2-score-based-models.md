---
layout: post
title: Score Based Generative Models
date: 2025-04-12 10:00:00 -0500
author: Nombre del Autor

---

| ![_config.yml]({{ site.baseurl }}/images/smld.jpg)| 
|:--:| 
| *Score Based Generative Models. Imagen importada de [yang-song.net](https://yang-song.net/blog/2021/score).* |

En algunos casos, los modelos probabilísticos en *Machine Learning* vienen en forma de una densidad de probabilidad (p.d.f.) no normalizada. Es decir, estos modelos tiene una constante de normalización que no se puede calcular computacionalmente.

Supongamos que se tiene un vector aleatorio $x\in\mathbb{R}^n$ el cual tiene una distribución $p_{\text{data}}$ para el cual se tiene un modelo probabilístico parametrizado por $p_{\theta}$, donde $\theta$ son los parámetros del modelo. El objetivo es encontrar los parámetros $\hat{\theta}$ de tal forma que el $p_{\theta}$ se asemeje a $p_{\text{data}}$. El problema es que solo es posible calcular la $p_{\theta}$ multiplicada por la constante de normalización $Z(\theta)$:

$$
p_{\theta}(x) = \frac{1}{Z(\theta)}q_{\theta}(x)
$$

Es decir, $q_{\theta}(x)$ es conocido y $Z(\theta)$ es analíticamente intratable.

Existe un método para estimar modelos no normalizados el cual se basa en calcular el *Score*. El *Score* es el gradiente de la densidad logarítmica con respecto a la variable aleatoria $x$:

$$
\nabla_x \log p_{\text{data}}(x)
$$

El proposito de este método es que $  \nabla_x \log p_{\theta}(x) \approx \nabla_x\log p_{\text{data}}(x)$. Donde $\nabla_x \log p_{\theta}(x)=\nabla_x \log q_{\theta}(x)$ ya que $Z(\theta)$ no depende de $x$.

En el caso de un modelo generativo basado en el *Score*.El objetivo principal entrenar una red neuronal $s_{\theta}(x)$ para predecir directamente $\nabla_x \log p_{\text{data}}(x)$. Una vez que la red neuronal ha sido entrenada, se puede utilizar el *Score* para generar muestras de la distribución de datos.

# Noise Conditional Score Networks (NCSNs)

| ![_config.yml]({{ site.baseurl }}/images/celeba_large.gif) ![_config.yml]({{ site.baseurl }}/images/cifar10_large.gif)| 
|:--:| 
| *Generación de muestras para un Noise Conditional Score Networks (NCSNs) de paper "<a href="https://arxiv.org/abs/1907.05600">Generative Modeling by Estimating Gradients of the Data Distribution</a>".* |


En la práctica, una distribución de datos tiende a estar concentrada en una variedad topológica de dimensión menor que la del espacio de datos. En palabras simples, los datos no están distribuidos uniformemente en el espacio de datos $\mathbb{R}^n$, sino que están concentrados en una región de menor dimensión. 

Esto significa que el *Score* de la distribución de datos estaría indefinido en la mayoría de los puntos del espacio. Una solución a este problema es perturbar los datos con ruido gaussiano. Dado que el soporte del ruido gaussiano es todo el espacio, el *Score* de la distribución de datos perturbados estará definido para todos los puntos del espacio. Además, al agregar ruido, se puede rellenar regiones del espacio con baja densidad de probabilidad en la distribución de datos original. Esto permite que el *Score* de la distribución de datos perturbados sea más fácil de estimar, dado que van a existir más puntos en estas regiones.

Definimos $(\sigma_i)^L_{i=1}$ como una secuencia geométrica positiva la cual se rije la siguiente regla:

$$
\frac{\sigma_1}{\sigma_2}=\cdots=\frac{\sigma_{L-1}}{\sigma_L}>1.
$$

El objetivo es ahora entrenar una red neuronal que estime el *Score* de todos los datos perturbados $\tilde{x}$. Es decir $\forall \sigma\in (\sigma_i)^L_{i=1}:s_{\theta}(\tilde{x},\sigma)\approx \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}\|x)$. Donde $q_{\sigma}(\tilde{x}\|x)=\mathcal{N}(\tilde{x}\|x,\sigma^2I)$. Sin embargo, ¿Cómo llegamos a predecir $\nabla_x \log p_{\text{data}}(x)$ a partir de $s_{\theta}(\tilde{x},\sigma)$?

Buenas la respuesta a esta pregunta se resuelve con una identidad facilmente demostrable:

$$
\mathbb{E}_{q_{\sigma}(\tilde{x},x)} \left[ \left\| s_{\theta}(\tilde{x}, \sigma) - \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}\|x) \right\|^2_2 \right]=\mathbb{E}_{q_{\sigma}(\tilde{x})} \left[ \left\| s_{\theta}(\tilde{x}, \sigma) - \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}) \right\|^2_2 \right],
$$

donde para un valor óptimo de $\theta={\theta}^*$ y un $\sigma$ es lo suficientemente pequeño se cumple que 

$$s_{\theta}(x)=\nabla_x \log q_{\sigma}(x)\approx \nabla_x \log p_{\text{data}}(x).$$ 

Ahora que sabemos como estimar el *Score* de la distribución de datos, podemos simplificar el primer término en la equivalencia anterior. Dando como resultado la siguiente función de pérdida:

$$
\mathcal{L}(\theta;\sigma) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)} \mathbb{E}_{\tilde{x}\sim \mathcal{N}(x,\sigma^2I)} \left[ \left\| s_{\theta}(\tilde{x}, \sigma) + \frac{(\tilde{x}-x)}{\sigma^2} \right\|^2_2 \right]
$$

Al final, la función de pérdida se pondera para todos los valores de $\sigma_i$.

# El algoritmo de muestreo

Una vez que la red neuronal ha sido entrenada $s_{\theta}(x)\approx \nabla_x \log p_{\text{data}}(x)$. Dado un $\epsilon>0$, y un valor inicial $\tilde{x}_0\sim \mathcal{N}(0,I)$, el algoritmo de muestreo se realiza de la siguiente manera:

$$
\tilde{x}_t = \tilde{x}_{t-1} + \frac{\epsilon}{2} s_{\theta}(\tilde{x}_{t-1}) + \sqrt{\epsilon}z_t, z_t\sim \mathcal{N}(0,I)
$$

Donde la distribución de $\tilde{x}_T$ es igual a $p_x$ cuando $\epsilon\rightarrow 0$ y $T\rightarrow \infty$. Al final, $\tilde{x}_T$ se convierte en una muestra de la distribución de los datos de entrenamiento. 

Este método de muestreo es conocido como *Langevin Dynamics*, una técnica para simular el movimiento de partículas a través de una ecuación diferencial estocástica. En este caso, la red neuronal $s_{\theta}(x)$ actúa como el témino determinista, como una "fuerza" que guía la trayectoria para generar una muestra realista. Mientras que $z_t$ introduce aleatoriedad, asegurando la diversidad en las muestras generadas. La combinación de ambos términos permite que el modelo explore el espacio de datos y genere muestras sintéticas que se asemejan a los datos de entrenamiento.


