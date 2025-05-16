---
layout: post
title: Flow Matching:&nbsp; El modelo generativo definitivo
date: 2025-04-12 10:00:00 -0500
author: Nombre del Autor
---

En los últimos años, hemos sido testigos de un avance significativo en el campo de los modelos generativos, especialmente en la generación de imágenes y videos. El modelo de difusión ha sido uno de los protagonistas en esta revolución, pero ahora, un nuevo contendiente ha emergido: el *Flow Matching*. Este modelo, que combina la eficiencia de los *Normalizing Flows* con la capacidad de los *Diffusion Models*, promete llevar la generación de datos a un nuevo nivel. En este artículo, exploraremos cómo funciona este innovador paradigma de entrenamiento y generación de datos.

# Continuous Normalizing Flows

A diferencia de lo que muchos podrían pensar, cada avance en el campo de la inteligencia artificial no es necesariamente el resultado de un nuevo descubrimiento. En ocasiones, se trata de una combinación ingeniosa de ideas existentes que, al ser unidas, crean algo completamente nuevo. Este es el caso del *Flow Matching*, un modelo generativo que combina dos paradigmas: los *Normalizing Flows* y los *Diffusion Models*.

Como lo vimos en el artículo anterior, los [*Normalizing Flows*](https://jarchv.github.io/modelos-generativos/) son modelos generativos que utilizan una serie de transformaciones invertibles para mapear una distribución simple a una distribución compleja. Sin embargo, utilizando un conjunto continuo de transformaciones, en lugar de una cantidad discreta, el modelo no está limitado a utilizar redes neuronales invertibles---capturando distribuciones más complejas.

Dado un conjunto de datos donde cada muestra $x\in\mathbb{R}\in d$, la trayectoria de distribución de probabilidad $(p_t)_{0\leq t\leq 1}$, y un campo vectorial $u:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$. Un campo vectorial $u_t$ puede ser utilizado para construir una función de flujo $\psi:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$, definida a través de la siguiente ecuación diferencial ordinaria (ODE):

$$
\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x)), \quad \psi_0(x) = x.
$$

Integrando la solución de la ODE, obtenemos la evolución de $\psi_t(x)$ a lo largo del tiempo $t$:

$$
\psi_t(x)=x+\int_0^t u_s(\psi_s(x))ds.
$$

El objetivo de un *Continuous Normalizing Flow* (CNF) es aprender un campo vectorial $u_t$ con una **red neuronal** $u^{\theta}_t(x)$ para modificar un distribución de probabilidad conocida $p_0$ en una más compleja $p_1$. Esto se logra al utilizar la siguiente ecuación:

$$
[\psi_t]_{*}p_0(x)=p_t(x)=p_0(\psi^{-1}_t(x))|\det(\partial_x\psi^{-1}_t(x))|
$$

Si $\psi_t$ es invertible, y satisface esta ecuación, entonces el campo vectorial $u_t$ es capaz de generar la trayectoria de distribución de probabilidad $p_t$.

## La Ecuación de Continuidad

No podemos resolver la ecuanción diferencial ordinaria, pero podemos calcular la probabilidad de la solución utilizando la *Ecuación de Continuidad*:

$$
\partial_t p_t(x_t) = -\text{div}(p_t u_t)(x_t)=-(p_t(x_t)\text{div}(u_t)(x_t)+\nabla_{x_t} p_t(x_t)\cdot u_t(x_t)),
$$

donde $\text{div}$ es el operador **divergencia**, y $x_t=\psi_t(x)$. Luego la derivada total se puede expresar como:

$$
\frac{d}{dt}\log p_t(x_t) = \frac{1}{p_t(x_t)}\frac{d p_t(x_t)}{dt}=\frac{1}{p_t(x_t)}(\partial_t p_t(x_t) + \nabla_{x_t} p_t(x_t)\cdot \frac{d x_t}{d t})
$$

$$
\frac{d}{dt}\log p_t(x_t) =-\text{div}(u_t)(x_t)
$$

Finalmente, para simular la evolución del logaritmo de la probabilidad de $\psi_t(x)$, podemos integrar la ecuación anterior desde $t=0$ hasta $t=1$:

$$
\log p^{\theta}_1(\psi_1(x)) = \log p_0(\psi_0(x)) -\int_0^1 \text{div}(u^{\theta}_t)(\psi_t(x))dt,
$$

donde el témino de la izquierda es utilizado para ser maximizado.

# Otro
La meta del Flow Matching es aprender un flujo que transforme una muestra $X_0\sim p$ generada por una distribución $p$ en una muestra objetivo $X_1:= \psi(X_0)$ tal que $X_1\sim q$, donde $q$ es la distribución objetivo. 


Flow Matching es una técnica basada en estimar un *campo vectorial*. Cada campo define un *flujo* $\psi_t$ al resolver una Ecuación Diferencial Ordinaria (ODE). Un flujo es una transformación bijectiva determinista y continua en el tiempo $t$. La meta del Flow Matching es aprender un flujo que transforme una muestra $X_0\sim p$ generada por una distribución $p$ en una muestra objetivo $X_1:= \psi(X_0)$ tal que $X_1\sim q$, donde $q$ es la distribución objetivo.

