<h1>Simple ray tracing CPU/GPU.</h1>

---
<h3>Некоторые технические характеристики.</h3>

- **CPU**: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz
- **GPU**: NVIDIA GeForce GTX 950
***
Отрендеренная статическая demo сцена (100 шаров и 10 точечных источников света). Было проведено по 10 экспериментов на CPU и GPU для каждого разрешения экрана. Результаты и замеры времени выполнений представлены в таблице ниже.

Название формата | Разрешение изображения	|	Лучшее время выполнения на CPU (с.)	|	Лучшее время выполнения на GPU (с.)	|	CPU/GPU
---	| ---	|	---	|	---	|	---
8K | 7680×4320	|	105,623	|	2,09286	|	50,4683
4K | 3840×2160	|	26,4081	|	0,563472	|	46,8667
FHD | 1920×1080	|	6,62854	|	0,156388	|	42,3852
qHD | 960×540 | 1,6671 | 0,044934 | 37,1011

---
<h4>Компилировать командой:</h4>

`nvcc EasyBMP.cpp kernel.cu -O3 -use_fast_math -Xptxas -dlcm=ca`
***
<h4>Некоторые примеры.</h4>

![Image alt](https://github.com/JinRSA/CUDA/blob/master/Simple%20Ray%20Tracing/Images/Demo%20sample%201.jpg)

![Image alt](https://github.com/JinRSA/CUDA/blob/master/Simple%20Ray%20Tracing/Images/Randome%20sample%200.jpg)

(◉ܫ◉)
