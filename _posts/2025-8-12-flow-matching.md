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

Dado un conjunto de datos donde cada muestra $y\in\mathbb{R}^d$ que sigue una distribución $p_{\text{data}}$, y una trayectoria de distribución de probabilidad $X\sim (p_t)_{0\leq t\leq 1}$. Un Continuous Normalizing Flow (CNF) se encarga de transformar una distribución de probabilidad conocida $p_0$ (Gaussiana) en una distribución más compleja $$p_1 \approx p_{\text{data}}$$ a través de un flujo continuo $\psi_t$, donde $t\in[0,1]$ representa la evolución del flujo a lo largo del tiempo. 

Se define también un campo vectorial $u:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$, y por consiguiente un flujo $\psi:[0,1]\times\mathbb{R}^d\to\mathbb{R}^d$. Este flujo está definido a través de la siguiente ecuación diferencial ordinaria (ODE):

$$
\frac{d}{dt}\psi_t(z) = u_t(\psi_t(z)), \quad \psi_0(z) = z.
$$

donde $X_1\sim p_{\text{data}}$, $Z\sim p_0$, y $X:=\psi(Z)\sim p_t$.

> Nótece que este es un caso especial de una **Ecuación Diferencial 
> Estocástica** (SDE). Ecuación utilizada en un Score Based Generative Model. Aquí,
> el término aleatorio no está presente.

Integrando la ODE, obtenemos la evolución de $\psi_t(z)$ a lo largo del tiempo $t$:

$$\textcolor{#9F2B68}{
\psi_t(z)=\psi_1(z)-\int_t^1 u_s(\psi_s(z))ds,
}$$


útil durante la simulación de la ODE. 

> Los campos vectoriales definen ODEs cuyas soluciones son flujos.
>
> -- <cite>Peter Holderrieth & Ezra Erives.</cite>

El objetivo de un CNF es aprender un campo vectorial $u_t$ a partir de una **red neuronal** $v^{\theta}_{t}(x)$ para modificar una distribución de probabilidad conocida $p_0$ en una más compleja $p_1$. 

Esto se logra al utilizar el operador $[\psi_t]_{\*}p_0(x)=p_t(x)$, donde:	

$$
p_t(x)=p_0(\psi^{-1}_t(x))|\det(\partial_x\psi^{-1}_t(x))|,\quad x\in\mathbb{R}^d
$$

Si $\psi_t$ es invertible, y satisface esta ecuación, entonces el campo vectorial $u_t$ es capaz de generar la trayectoria de distribución de probabilidad $p_t$.

## La Ecuación de Continuidad

No podemos resolver la ODE, pero podemos calcular la probabilidad de la solución utilizando la *Ecuación de Continuidad*. Considerando $x:=\psi_t(z)$, la evolución de la distribución de probabilidad $p_t$ está dada por la siguiente ecuación en derivadas parciales:

$$
\partial_t p_t(x) = -\text{div}(p_t u_t)(x)=-(p_t(x)\text{div}(u_t)(x)+\nabla_{x} p_t(x)\cdot u_t(x)),
$$

donde $\text{div}$ es el operador **divergencia**. Luego, la derivada del *log likelihood* se puede expresar como:

$$
\frac{d}{dt}\log p_t(x) = \frac{1}{p_t(x)}\frac{d p_t(x)}{dt}=\frac{1}{p_t(x)}(\partial_t p_t(x) + \nabla_{x} p_t(x)\cdot \frac{d x}{d t})
$$


$$
\frac{d}{dt}\log p_t(x) =-\text{div}(u_t)(x)=-\text{Tr}(\partial_{x} u_t(x)),
$$

Esta ecuación es fundamental para aproximar el *log likelihood* de las muestras generadas por el CNF el cual es necesario para entrenar el modelo. Para entrenar el CNF, simplemente se maximaiza el *log likelihood* de las muestras:

$$
\text{arg}\max_{\theta} \mathbb{E}_{p_{\text{data}}} \left[ \log p^{\theta}_1(\psi_1(z)) \right] = \mathbb{E}_{p_0} \left[ \log p_0(\psi_0(z)) -\int_0^1 \text{div}(v^{\theta}_t)(\psi_t(z))dt \right],
$$

donde:

$$
\log p_0(w) = -\frac{1}{2}\|w\|^2_2 - \frac{d}{2}\log(2\pi).
$$

Para ello es necesario simular de manera precisa $\log p^{\theta}_1(\psi_1(x))$ mediante la ODE durante el entrenamiento. Un método simple para este proceso es el método de *Euler*. En este método, la ODE se discretiza en pasos pequeños, con lo que podemos aproximar la solución de la ODE a través de una serie de pasos discretos. Finalmente, en cada iteración de entrenamiento, ejecutaríamos el siguiente algoritmo:

<div class="example"><pre>
EULER_SOLVER(u_θ, y, N_steps):
    Δt ← 1 / N_steps
    x ← y
    div ← 0
    t ← 1

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
\mathcal{L}_{\text{FM}}(\theta)  = \mathbb{E}_{t\sim \mathcal{U}(0,1), p_t}\left[\|v^{\theta}_t(x)-u_t(x)\|^2\right]
$$

donde $\theta$ es el vector de parámetros de la red $v^{\theta}_t$, $t\sim \mathcal{U}(0,1)$ es un número aleatorio y $x$ es una muestra de la distribución $p_t$. Cuando la función de pérdida se minimiza lo suficiente, $v^{\theta}_t$ se aproxima a $u_t$.

Sin embargo, el problema ahora es que no tenemos acceso a un $u_t$ explícito. De hecho, podrían existir múltiples trayectorias que transformen $p_1$ en $p_0$, lo que complica el aprendizaje del campo vectorial. 

## Contruyendo un campo vectorial $u_t$

Para resolver este problema, se condiciona la función de costo para cada muestra $X_1\sim p_1$. Para ello se define la trayectoria de distribución de probabilidad condicionada $p_t(x\|x_1)$, de modo que $p_0(x\|x_1)=p(x)$ y $p_1(x\|x_1)=\mathcal{N}(x\|x_1,\sigma^2\mathbf{I})$---donde $\sigma>0$. Luego, podemos calcular la trayectoria de distribución de probabilidad mariginal $p_t(x)$ como:

$$
p_t(x) = \int \mathbb{P}(X=x\|X_1=x_1)\mathbb{P}(X_1=x_1)dx_1 = \int p_t(x\|x_1)p_{\text{data}}(x_1)dx_1,
$$

donde $p_1(x)=\int p_1(x\|x_1)p_{\text{data}}(x_1)dx_1\approx p_{\text{data}}(x)$. Luego, si consideramos $u_t(x\|x_1):=u_t(\psi_t(x_1))$, procedemos a calcular $\partial_t p_t(x)$:

$$
\partial_t p_t(x) = \int \partial_t p_t(x\|x_1)p_{\text{data}}(x_1)dx_1.
$$

Utilizando la Ecuación de Continuidad para la distribución condicionada $p_t(x\|x_1)$:

$$
\partial_t p_t(x)= -\int \text{div}(p_t u_t)(x\|x_1)p_{\text{data}}(x_1)dx_1,
$$

$$
\partial_t p_t(x)= -\text{div}(\int p_t(x\|x_1) u_t(x\|x_1) p_{\text{data}}(x_1)dx_1).
$$

Como $\partial_t p_t(x)=-\text{div}(p_t u_t)(x)$, podemos definir un campo vectorial marginal $u_t(x)$ que cumpla la igualdad:

$$
p_t(x) u_t(x) = \int p_t(x\|x_1) u_t(x\|x_1) p_{\text{data}}(x_1)dx_1,
$$

$$
u_t(x) = \frac{\int p_t(x\|x_1) u_t(x\|x_1) p_{\text{data}}(x_1)dx_1}{p_t(x)},
$$

donde $u_t(\cdot\|x_1):\mathbb{R}\rightarrow \mathbb{R}$ es un campo vectorial condicionado a $x_1$ el cual genera la trayectoria de distribución de probabilidad $p_t(\cdot\|x_1)$.

## Conditional Flow Matching (CFM)

Finalmente, asumiendo que $p_t(x)>$ para todo $x\in\mathbb{R}^d$ y $t\in[0,1]$, podemos redefinir la función de pérdida del Flow Matching como:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t\sim \mathcal{U}(0,1), p_{t|1}, p_{data}}\left[\|v^{\theta}_t(x)-u_t(x\|x_1)\|^2\right].
$$

Dejo como tarea al lector demostrar que: 

$$
\nabla_{\theta}\mathcal{L}_{\text{CFM}}(\theta) = \nabla_{\theta}\mathcal{L}_{\text{FM}}(\theta).
$$

Esta igualdad implica que podemos entrenar un CNF para generar muestras desde $p_t$ sin necesidad de acceder a $u_t$ explícitamente. Y particularmente, cuando $t=1$, nos permite approximar $p_{\text{data}}$. Lo único que faltaría es definir una trayectoria de distribución de probabilidad condicionada y el campo vectorial asociado más adecuado. Una posible opción es considerar el siguiente flujo lineal:

$$
X=\psi(Z)=t X_1 + (1-t)Z\sim p_t.
$$

donde 

$$
X|x_1=t x_1 + (1-t)Z \sim p_{t\|1}=\mathcal{N}(x\|tx_1,(1-t)^2 I).
$$

Este flujo transforma una muestra $x_1$ en una muestra $z$ a través de una trayectoria lineal. Ahora procedemos a calcular $u_t(X_{t\|1})$:


$$
u_t(X|x_1) = \frac{d X}{d t} = x_1 - Z=\frac{x_1-X}{1-t}.
$$

Entonces $u_t(x\|x_1)=x_1-z=\frac{x_1 - x}{1-t}$.

> En los Diffusion Models las trayectorias son aleatorias. Mientras
> que en los Flow Matching son deterministas. Al menos, utilizando
> esta configuración.

Ahora la función de pérdida del CFM se puede redefinir de la siguiente manera:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t\sim \mathcal{U}(0,1), p_{0}, p_{data}}\left[\|v^{\theta}_t(x)-(x_1-z)\|^2\right], z\sim p_0 = \mathcal{N}(0, I), x = t x_1 + (1-t) z.
$$

Una función de pérdida simple, sencilla de implementar, y con potencial de escalabilidad a datos de gran dimensión.

# Conclusiones Finales

En este artículo, hemos explorado los fundamentos matemáticos necesarios para formular un método eficiente para entrenar Continuous Normalizing Flows sin la necesidad de simular la ODE durante el entrenamiento. El Flow Matching, y en particular el Conditional Flow Matching, nos proporciona una manera sencilla de aproximar el campo vectorial necesario para transformar una distribución simple en una más compleja utilizando únicamente muestras de datos y ruido gaussiano. En comparación con otros modelos generativos, el Flow Matching ofrece una ventaja significativa en términos de eficiencia computacional y simplicidad de implementación. Además, abre nuevas posibilidades para la generación de datos de alta dimensionalidad, como imágenes y videos.

Si el lector está interesado en profundizar en este tema, le recomiendo revisar los siguiente artículos:

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Neural ordinary differential equations](https://arxiv.org/abs/1806.07366)
- [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)
- [An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/abs/2506.02070)