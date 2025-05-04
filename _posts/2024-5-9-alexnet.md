---
layout: post
title: AlexNet:&nbsp; La Red Neuronal de 44 millones de dólares
---

| ![_config.yml]({{ site.baseurl }}/images/alexnet-post.png) | 
|:--:| 
| *De izquierda a derecha: Ilya Sutskever, Alex Krizhevsky y  Geoffrey Hinton. Los creadores de AlexNet.* |

Año 2009, en la *IEEE Conference on Computer Vision and Pattern Recognition*---una conferencia anual de visión por computador---se presentó un artículo titulado "ImageNet: A Large-Scale Hierarchical Image Database". Una base de datos con más de 14 millones de imágenes etiquetadas en más de 20000 categorías. Permitiendo a los investigadores entrenar modelos con un conjunto de datos mucho más grande y diverso.

ImageNet rápidamente se convirtió en una competencia anual---el desafío *ImageNet Large Scale Visual Recognition Challenge*. Donde equipos competían por presentar el mejor algoritmo capaz de clasificar entre 1000 categorías con el menor error posible. En el 2012, un equipo llamado *SuperVision* logró un porcentaje de acierto muy por encima de sus competidores. El modelo de SuperVision, era algo distinto, no era un modelo tradicional de aprendizaje automático, sino una red neuronal convolucional. Este modelo fue el primero en demostrar que las redes neuronales podían superar a los modelos tradicionales de aprendizaje automático en tareas de visión por computador.

Con dos tarjetas gráficas NVIDIA GTX 580, el equipo SuperVision conformado por Alex Krizhevsky, Ilya Sutskever y Geoffrey Hinton crearon *AlexNet*. No paso mucho tiempo para que los gigantes tecnológicos se dieran cuenta del potencial de AlexNet, y de sus creadores. Google había desarrollado una plataforma para la búsqueda de imágenes, y creían que esta red neuronal podía mejorar la precisión y velocidad de sus búsquedas---era evidente el interés de Google. Sin embargo, Baidu---el Google Chino---fue el que se adelantó y ofreció 12 millones de dólares al equipo de  Hinton por algunos años de trabajo en la compañía. Luego Google, Microsoft y DeepMind también mostraron interés en adquirir al grupo de investigadores. 

Hinton, al ver este escenario, declinó la oferta de Baidu y decidió fundar junto con sus dos estudiantes una startup llamada *DNNResearch*. Más tarde, después de consultar con sus abogados, ideó un plan. Una subasta que se llevaría a cabo en el *NIPS 2012 (The Twenty-sixth Annual Conference on Neural Information Processing Systems)*---una conferencia anual que reúne a investigadores en el campo de la Inteligencia Artificial.

Hinton y sus estudiantes acudirían a la conferencia, pero no solo para presentar su investigación, sino también para subastar su talento al mejor postor. La conferencia tuvo lugar en el *Harrah's Lake Tahoe Hotel & Casino*. Un lugar en Nevada rodeado de montañas y un lago, la vista era espectacular. Hinton llegó al hotel y tuvo un encuentro breve con los representantes de cada compañía. 

Acordaron realizar la subasta por correo electrónico. Los representantes de cada compañía debían enviar sus ofertas mientras Hinton, desde su habitación de hotel, las recibía. Debido a un problema con su espalda---producto de una lesión de adolescente---Hinton no podía estar sentado por mucho tiempo. Por lo que armó una especie de torre con un bote de basura y una mesa, colocó su laptop encima, y esperó junto a sus estudiantes las ofertas.

Las reglas eran simples: cuando una compañía enviaba su oferta, las otras tenían una hora para superarla con al menos un millón de dólares. Si ninguna de las compañías enviaba una oferta, la subasta terminaba, dando como ganador al último postor. DeepMind no podía competir, al ser una startup de un poco más de dos años, no tenía el capital de Google, Microsoft o Baidu---prontamente se retiraron de la subasta. 

La subasta continuó entre Google, Microsoft y Baidu. Las ofertas llegaron a 15, 20, 22 millones de dólares. Fue aquí cuando Microsoft decidió retirarse. Un dato curioso fue que se utilizó una cuenta de correo Gmail para realizar la subasta, por lo que Google tranquilamente podría haber estado interceptando las ofertas que le llegaban a Hinton y sus estudiantes.

Finalmente, quedaron Google y Baidu en la subasta---un duelo entre los dos motores de búsqueda más grandes del mundo. Google, a diferencia de Microsoft, estaba dispuesto a pagar el precio que fuera necesario para adquirir a DNNResearch. Entonces, Baidu subió la oferta a unos 25 millones de dólares, luego 30 millones y finalmente 35 millones de dólares. Las ofertas estaban subiendo ridículamente rápido por ambas partes, fue entonces que Hinton decidió reducir el tiempo de respuesta a 30 minutos. Las ofertas subieron aún más rápido: 40, 41, 42, 43 millones de dólares. En algún momento, el monto llegó a 44 millones de dólares. En este punto, Hinton decidió suspender la subasta. Necesitaban un descanso, habían estado recibiendo ofertas durante horas y la tensión era desgastante.

Al día siguiente, después de un merecido descanso y un análisis de las ofertas, Hinton decidió dar por terminada la subasta. Cuando esto sucedió, ambas compañías creyeron que era una broma, ellos estaban dispuestos a ofrecer mucho más dinero por adquirir el talento detrás de AlexNet. Sin embargo, para Hinton y sus estudiantes, 44 millones de dólares era más que suficiente. Vieron en Google una compañía con el mejor ambiente para continuar su trabajo---después de todo eran académicos, su prioridad era la investigación.

Microsoft y Baidu se quedaron con un sabor amargo, ahora Google tenía el talento detrás de AlexNet y por consiguiente, estaba un paso adelante en la carrera por la Inteligencia Artificial General.

## Referencias

1. [Genius Makers: The Mavericks Who Brought AI to Google, Facebook, and the World. Cade Metz, 2021.](https://www.amazon.com/Genius-Makers-Mavericks-Brought-Facebook/dp/1524742678)
2. [Ego, Fear and Money: How the A.I. Fuse Was Lit. Cade Metz et. al, 2023.](https://www.nytimes.com/2023/12/03/technology/ai-openai-musk-page-altman.html)
3. [The data that transformed AI research—and possibly the world. Dave Gershgorn, 2017.](https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world)
4. [ImageNet: A large-scale hierarchical image database. Deng et. al, 2009.](https://ieeexplore.ieee.org/document/5206848)
5. [ImageNet Classification with Deep Convolutional Neural Networks. Krizhevsky et. al, 2012.](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
6. [ImageNet Large Scale Visual Recognition Challenge (ILSVRC).](https://image-net.org/challenges/LSVRC/)