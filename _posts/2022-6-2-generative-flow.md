---
layout: paper
title: Generative Flows as a General Purpose Solution for Inverse Problems
---
<div align="center">
<h2> Autores </h2>
</div>
- José A. Chávez

<div align="center">
<h2> Abstract</h2>
</div>
Due to the success of generative flows to model data distributions, they have been explored in inverse problems. Given a pre-trained generative flow, previous work proposed to minimize the 2-norm of the latent variables as a regularization term. The intuition behind it was to ensure high likelihood latent variables that produce the closest restoration. However, high-likelihood latent variables may generate unrealistic samples as we show in our experiments. We therefore propose a solver to directly produce high-likelihood reconstructions. We hypothesize that our approach could make generative flows a general purpose solver for inverse problems. Furthermore, we propose 1 x 1 coupling functions to introduce permutations in a generative flow. It has the advantage that its inverse does not require to be calculated in the generation process. Finally, we evaluate our method for denoising, deblurring, inpainting, and colorization. We observe a compelling improvement of our method over prior works.

<div align="center">
<h2> Paper </h2>
<a href="https://arxiv.org/pdf/2110.13285.pdf">
<img src="{{ site.baseurl }}/images/paper-ambient.jpg" alt="paper-ambient" width="50%" height="50%">
</a>
</div>

<div align="center">
<h2> Implementación</h2>
<a href="https://github.com/jarchv/solverip">Repositorio de Github</a>
</div>

<div align="center">
<h2>Bibtex</h2>
</div>

<div class="example"><pre>
@InProceedings{Chavez_2022_CVPR,
    author    = {Ch\'avez, Jos\'e A.},
    title     = {Generative Flows as a General Purpose Solution for Inverse Problems},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1490-1498}
}
</pre></div>

```