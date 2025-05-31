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

Dado un conjunto de datos donde cada muestra $x\in\mathbb{R}^d$, la trayectoria de distribución de probabilidad $(p_t)_{0\leq t\leq 1}$, y un campo vectorial $u:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$ podemos construir un flujo $\psi:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$. Este flujo está definido a través de la siguiente ecuación diferencial ordinaria (ODE):

$$
\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x)), \quad \psi_0(x) = x.
$$



Integrando la solución de la ODE, obtenemos la evolución de $\psi_t(x)$ a lo largo del tiempo $t$:

$$\textcolor{#9F2B68}{
\psi_t(x)=x+\int_0^t u_s(\psi_s(x))ds,
}$$


últil durante la simulación de la ODE. 

> Los campos vectoriales definen ODEs cuyas soluciones son flujos.
>
> -- <cite>Peter Holderrieth & Ezra Erives.</cite>

El objetivo de un *Continuous Normalizing Flow* (CNF) es aprender un campo vectorial $u_t$ con una **red neuronal** $u^{\theta}_{t}(x)$ para modificar una distribución de probabilidad conocida $p_0$ en una más compleja $p_1$. 

Esto se logra al utilizar del operador $[\psi_t]_{\*}p_0(x)=p_t(x)$, donde:	

$$
p_t(x)=p_0(\psi^{-1}_t(x))|\det(\partial_x\psi^{-1}_t(x))|
$$

Si $\psi_t$ es invertible, y satisface esta ecuación, entonces el campo vectorial $u_t$ es capaz de generar la trayectoria de distribución de probabilidad $p_t$.

## La Ecuación de Continuidad

No podemos resolver la ODE, pero podemos calcular la probabilidad de la solución utilizando la *Ecuación de Continuidad*:

$$
\partial_t p_t(x_t) = -\text{div}(p_t u_t)(x_t)=-(p_t(x_t)\text{div}(u_t)(x_t)+\nabla_{x_t} p_t(x_t)\cdot u_t(x_t)),
$$

donde $\text{div}$ es el operador **divergencia**, y $x_t=\psi_t(x)$. Luego la derivada total se puede expresar como:

$$
\frac{d}{dt}\log p_t(x_t) = \frac{1}{p_t(x_t)}\frac{d p_t(x_t)}{dt}=\frac{1}{p_t(x_t)}(\partial_t p_t(x_t) + \nabla_{x_t} p_t(x_t)\cdot \frac{d x_t}{d t})
$$

$$
\frac{d}{dt}\log p_t(x_t) =-\text{div}(u_t)(x_t)=-\text{Tr}(\partial_{x_t} u_t(x_t)),
$$

Finalmente, para simular la evolución del logaritmo de la probabilidad de $\psi_t(x)$, podemos integrar la ecuación anterior desde $t=0$ hasta $t=1$:

$$
\textcolor{#9F2B68}{
\log p^{\theta}_1(\psi_1(x)) = \log p_0(\psi_0(x)) -\int_0^1 \text{div}(u^{\theta}_t)(\psi_t(x))dt,
}
$$

donde $p^{\theta}_1$ es la distribución de $x_1=\psi^{\theta}_1(x_0)$. Finalmente, para entrenar el modelo, simplemente se minimiza la divergencia KL entre $p^{\theta}_1$ y $q$:

$$
\mathcal{L}(\theta) = D_{\text{KL}}(q,p^{\theta}_1)
$$

Sin embargo, para obtener esta divergencia es necesario simular de manera precisa la ODE durante el entrenamiento a través de un método numérico. Un método simple y comúnmente utilizado es el método de *Euler*. En este método, la ODE se discretiza en pasos pequeños, con lo que podemos aproximar la solución de la ODE a través de una serie de pasos discretos. Finalmente, en cada iteración de entrenamiento, ejecutaríamos el siguiente algoritmo:

<div class="example"><pre>
EULER_SOLVER(u_θ, x_0, logp_0, t_0, t_1, N_steps):
    Δt ← (t_1 - t_0) / N_steps
    x ← x_0
    logp ← logp_0
    t ← t_0

    for k = 1 to N_steps do:
        dx_dt ← u_θ(x, t)
        J ← Jacobiano de u_θ con respecto a x
        tr ← traza de J
        dlogp_dt ← -tr
        x ← x + Δt · dx_dt
        logp ← logp + Δt · dlogp_dt
        t ← t + Δt

    return x, logp

</pre></div>

Sim embargo, esto puede ser computacionalmente costoso, y producir errores de aproximación significativos. Es aquí donde entra en juego el *Flow Matching*. Un método sencillo y eficiente para **entrenar un CNF sin necesidad de simular la ODE durante el entrenamiento**.

# Flow Matching

Sea una variable aleatoria $x_1$ con distribución desconocida $q$, donde solo tenemos acceso a las muestras. Además, supongamos que $p_t$ es la trayectoria de distribución de probabilidad tal que $p_0=p$ es una distribución conocida---como la distribución gaussiana estándar---y $p_1$ es aproximadamente igual a $q$. El objetivo del *Flow Matching* es aprender el $p_t$ que nos permita fluir desde $p_0$ hasta $p_1$---el mismo que el CNF.

En lugar de simular la ODE, el *Flow Matching* busca aprender directamente el campo vectorial $u_t$ a partir de las muestras disponibles. Para ello se define el Flow Matching objetivo como:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_t\sim p_t}\left[u_t(x_t)-u^{\theta}_t(x_t)\right]
$$

donde $\theta$ son los parámetros del campo vectorial $u^{\theta}_t$  de la CNF, $t\sim \mathcal{U}(0,1)$ es un valor aleatorio y $x_t\sim p_t$ es una muestra de la distribución $p_t$. Cuando la función de pérdida se minimiza lo suficiente, el campo vectorial $u^{\theta}_t$ se convierte en una aproximación del campo vectorial $u_t$ que fluye desde $p_0$ hasta $p_1$.

Sin embargo, el problema ahora es que no tenemos acceso a un $u_t$ explícito. De hecho podrían existir múltiples trayectorias que transformen $p_0$ en $p_1\approx q$, lo que complica el aprendizaje del flujo. Para resolver este problema se utiliza el campo vectorial definido en cada muestra de entrenamiento. Esto permite construir una función de pérdida mucho más sencilla.


