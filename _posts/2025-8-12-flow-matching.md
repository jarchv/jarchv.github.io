---
layout: post
title: Flow Matching:&nbsp; Introducción
date: 2025-08-12 10:00:00 -0500
author: Nombre del Autor
---

En los últimos años, hemos sido testigos de un avance significativo en el campo de los modelos generativos, especialmente en la generación de imágenes y videos. El modelo de difusión ha sido uno de los protagonistas en esta revolución. Pero ahora, un nuevo *modelo* está tomando protagonismo: el *Flow Matching*. Este modelo, que en realidad es un método, ha llevado la escalabilidad de los *Normalizing Flows* a un nuevo nivel, y propone una manera sencilla de generar datos de alta dimensionalidad. En este artículo, exploramos los fundamentos del Flow Matching, la motivación detrás de su desarrollo y cómo se diferencia de otros modelos generativos.

# Continuous Normalizing Flows

A diferencia de lo que muchos podrían pensar, cada avance en el campo de la inteligencia artificial no es necesariamente el resultado de un nuevo descubrimiento. En ocasiones, se trata de una combinación ingeniosa de ideas existentes que, al ser unidas, crean algo completamente nuevo. Este es el caso del Flow Matching, un modelo generativo que combina los conceptos de dos modelos generativos: los Normalizing Flows y los *Diffusion Models*.

Como lo vimos en el artículo anterior, los [Normalizing Flows](https://jarchv.github.io/modelos-generativos/) son modelos que utilizan una serie de transformaciones invertibles para mapear una distribución simple a una distribución compleja. Sin embargo, utilizando un conjunto continuo de transformaciones, en lugar de una cantidad discreta, el modelo no está limitado a utilizar redes neuronales invertibles---capturando de esta manera distribuciones más complejas.

Dado un conjunto de datos donde cada muestra $y\in\mathbb{R}^d$ que sigue una distribución $y\sim p_{\text{data}}$, y una trayectoria de distribución de probabilidad $(p_t)_{0\leq t\leq 1}$. Un Continuous Normalizing Flow (CNF) se encarga de transformar una distribución de probabilidad conocida $p_0$ (Gaussiana) en una distribución más compleja $$p_1 \approx p_{\text{data}}$$ a través de un flujo continuo $\psi_t(x)$, donde $t\in[0,1]$ representa la evolución del flujo a lo largo del tiempo. 

Se define también un campo vectorial $u:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$, podemos construir un flujo $\psi:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$. Este flujo está definido a través de la siguiente ecuación diferencial ordinaria (ODE):

$$
\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x)), \quad \psi_1(x) = y.
$$

> Nótece que este es un caso especial de una **Ecuación Diferencial 
> Estocástica** (SDE). Ecuación utilizada en un Score Based Generative Model. Aquí,
> el término aleatorio no está presente.

Integrando la ODE, obtenemos la evolución de $\psi_t(x)$ a lo largo del tiempo $t$:

$$\textcolor{#9F2B68}{
\psi_t(x)=y+\int_t^1 u_s(\psi_s(x))ds=y-\int_1^t u_s(\psi_s(x))ds,
}$$


útil durante la simulación de la ODE. 

> Los campos vectoriales definen ODEs cuyas soluciones son flujos.
>
> -- <cite>Peter Holderrieth & Ezra Erives.</cite>

El objetivo de un CNF es aprender un campo vectorial $u_t$ a partir de una **red neuronal** $u^{\theta}_{t}(x)$ para modificar una distribución de probabilidad conocida $p_0$ en una más compleja $p_1$. 

Esto se logra al utilizar el operador $[\psi_t]_{\*}p_0(x)=p_t(x)$, donde:	

$$
p_t(x)=p_0(\psi^{-1}_t(x))|\det(\partial_x\psi^{-1}_t(x))|
$$

Si $\psi_t$ es invertible, y satisface esta ecuación, entonces el campo vectorial $u_t$ es capaz de generar la trayectoria de distribución de probabilidad $p_t$.

## La Ecuación de Continuidad

No podemos resolver la ODE, pero podemos calcular la probabilidad de la solución utilizando la *Ecuación de Continuidad*:

$$
\partial_t p_t(x_t) = -\text{div}(p_t u_t)(x_t)=-(p_t(x_t)\text{div}(u_t)(x_t)+\nabla_{x_t} p_t(x_t)\cdot u_t(x_t)),
$$

donde $\text{div}$ es el operador **divergencia**, y $x_t=\psi_t(x)$. Luego, la derivada del *log likelihood* se puede expresar como:

$$
\frac{d}{dt}\log p_t(x_t) = \frac{1}{p_t(x_t)}\frac{d p_t(x_t)}{dt}=\frac{1}{p_t(x_t)}(\partial_t p_t(x_t) + \nabla_{x_t} p_t(x_t)\cdot \frac{d x_t}{d t})
$$

$$
\frac{d}{dt}\log p_t(x_t) =-\text{div}(u_t)(x_t)=-\text{Tr}(\partial_{x_t} u_t(x_t)).
$$

Finalmente, para simular la evolución del logaritmo de la probabilidad de $\psi_t(x)$, podemos integrar la ecuación anterior desde $t=0$ hasta $t=1$:

$$
\textcolor{#9F2B68}{
\log p^{\theta}_1(\psi_1(x)) = \log p_0(\psi_0(x)) -\int_0^1 \text{div}(u^{\theta}_t)(\psi_t(x))dt,
}
$$

Finalmente, el objetivo para estimar el vector de parámetros $\theta$ es maximizar el log likelihood de la distribución de probabilidad $p_1$:

$$
\text{arg}\max_{\theta} \mathbb{E}_{x\sim p_{\text{data}}} \left[ \log p^{\theta}_1(\psi_1(x)) \right] = \mathbb{E}_{x\sim p_0} \left[ \log p_0(\psi_0(x)) -\int_0^1 \text{div}(u^{\theta}_t)(\psi_t(x))dt \right],
$$

$$
\log p_0(w) = -\frac{1}{2}\|w\|^2_2 - \frac{d}{2}\log(2\pi).
$$

Para ello es necesario simular de manera precisa $\log p^{\theta}_1(\psi_1(x))$ mediante la ODE durante el entrenamiento. Un método simple para este proceso es el método de *Euler*. En este método, la ODE se discretiza en pasos pequeños, con lo que podemos aproximar la solución de la ODE a través de una serie de pasos discretos. Finalmente, en cada iteración de entrenamiento, ejecutaríamos el siguiente algoritmo:

<div class="example"><pre>
EULER_SOLVER(u_θ, y, t_0, t_1, N_steps):
    Δt ← (t_1 - t_0) / N_steps
    x ← y
    div ← 0
    t ← t_1

    for k = 1 to N_steps do:
        dx_dt ← u_θ(x, t)
        tr ← Traza del Jacobiano de u_θ con respecto a x
        x ← x - Δt · dx_dt
        div ← div - Δt · tr
        t ← t - Δt

    log_p = 0.5 * (x @ x) - (len(x) / 2) * log(2 * PI) - div
    return x, logp

</pre></div>

Sin embargo, esto puede ser computacionalmente costoso sobre todo con datos de alta dimensionalidad. También abre la puerta a producir errores de aproximación significativos, lo que puede afectar la calidad de las muestras generadas. Es aquí donde entra en juego el Flow Matching. Un método sencillo y eficiente para **entrenar un CNF sin necesidad de simular la ODE durante el entrenamiento**.

# Flow Matching

Dada una variable aleatoria $x_1$ con distribución desconocida $p_{\text{data}}$, donde solo tenemos acceso a las muestras. Además, supongamos que $p_t$ es la trayectoria de distribución de probabilidad. Donde $p_0$ es una distribución conocida, como la distribución gaussiana estándar, y $p_1$ es aproximadamente igual a $p_{\text{data}}$. El objetivo del Flow Matching es aprender $p_t$, la trayectoria que nos permita fluir desde $p_1$ hasta $p_0$.

En lugar de simular la ecuación diferencial ordinaria durante el entrenamiento, el Flow Matching nos provee una manera sencilla de estimar directamente el campo vectorial $u_t$ a partir de las muestras disponibles. Para ello, el Flow Matching define la función de costo como:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t\sim \mathcal{U}(0,1), x_t\sim p_t}\left[\|u_t(x_t)-u^{\theta}_t(x_t)\|^2\right]
$$

donde $\theta$ es el vector de parámetros de la red $u^{\theta}_t$, $t\sim \mathcal{U}(0,1)$ es un número aleatorio y $x_t\sim p_t$ es una muestra de la distribución $p_t$. Cuando la función de pérdida se minimiza lo suficiente, $u^{\theta}_t$ se aproxima a $u_t$.

Sin embargo, el problema ahora es que no tenemos acceso a un $u_t$ explícito. De hecho, podrían existir múltiples trayectorias que transformen $p_1$ en $p_0$, lo que complica el aprendizaje del campo vectorial. Para resolver este problema, se condiciona la función de costo para cada muestra $x_1\sim p_1$. Y se define el flujo como una interpolación lineal entre $x_1$ y $x_0\sim p_0$:

$$
\psi_t(x_1) = x_t = x_0 + t(x_1 - x_0),
$$

> En los Diffusion Models las trayectorias son aleatorias. Mientras
> que en los Flow Matching son deterministas. Al menos, utilizando
> esta configuración.


Reemplazando en la ODE:

$$
u_t(\psi_t(x_1))=\frac{d}{dt}\psi_t(x_1)=x_1 - x_0.
$$

Podemos simplificar la notación del campo vectorial condicionado a $x_1$ como $u_t(x_t\|x_1):=u_t(\psi_t(x_1))$. Ahora la función de pérdida se puede definir como:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t\sim \mathcal{U}(0,1), x_t\sim p_{t|1}, x_1\sim p_{data}}\left[\|u_t(x_t|x_1)-u^{\theta}_t(x_t)\|^2\right],
$$

donde $x_1\sim p_{data}$, $x_0\sim p_0$ y $p_{t\|1}(x_t\|x_1)=\mathcal{N} (x_t\|tx_1,(1-t)^2 I)$. Finalmente reemplazando $u_t(x_t\|x_1)$:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t\sim \mathcal{U}(0,1), x_0\sim p_{0}, x_1\sim p_{data}}\left[\|(x_1-x_0)-u^{\theta}_t(x_t)\|^2\right].
$$

Una función de pérdida simple, sencilla de implementar, y con potencial de escalabilidad a datos de gran dimensión.

# Conclusiones Finales

Hemos recorrido la línea que va desde los CNFs hasta la formulación práctica del Flow Matching. El Flow Matching ofrece un método de tipo *simulación-free* para aprender un campo vectorial que transforme una distribución sencilla (Gaussiana) en la distribución de los datos, sin tener que integrar la ODE durante el entrenamiento. Esto reduce costes computacionales y abre la puerta a CNFs escalables a datos de mayor dimensión y complejidad.

Para continuar con el estudio de este método, recomiendo revisar los siguiente artículos:

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Neural ordinary differential equations](https://arxiv.org/abs/1806.07366)
- [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)
- [An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/abs/2506.02070)