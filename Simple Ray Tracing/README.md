<h1>Simple ray tracing CPU/GPU.</h1>

---
<h3>Некоторые технические характеристики.</h3>

- **CPU**: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz
- **GPU**: NVIDIA GeForce GTX 950

***
Отрендеренная статическая demo сцена (100 шаров и 10 точечных источников света). Результаты и замеры времени выполнений представлены в таблице ниже.

Название формата | Разрешение изображения	|	Лучшее время выполнения на CPU (с.)	|	Лучшее время выполнения на GPU (с.)	|	CPU/GPU
---	| ---	|	---	|	---	|	---
8K | 7680×4320	|	105,623	|	3,97068	|	26,6007
4K | 3840×2160	|	26,4081	|	1,12051	|	23,5679
FHD | 1920×1080	|	6,62854	|	0,32913	|	20,1396
qHD | 960×540 | 1,6671 | 0,10035 | 16,6129

---
<h4>Компилировать командой:</h4>

`nvcc EasyBMP.cpp kernel.cu -O3 -use_fast_math -Xptxas -dlcm=ca`
***
<h4>Некоторые примеры.</h4>

![Image alt](https://github.com/JinRSA/CUDA/blob/master/Simple%20Ray%20Tracing/Images/Demo%20sample%201.jpg)

![Image alt](https://github.com/JinRSA/CUDA/blob/master/Simple%20Ray%20Tracing/Images/Randome%20sample%200.jpg)

(◉ܫ◉)
