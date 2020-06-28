<h1>Simple ray tracing CPU/GPU.</h1>

---
<h3>Некоторые технические характеристики.</h3>

- **CPU**: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz
- **GPU**: NVIDIA GeForce GTX 950

***
Отрендеренная статическая demo сцена (100 шаров и 10 точечных источниками света). Результаты и замеры времени выполнений представлены в таблице ниже.

Разрешение изображения	|	Лучшее время выполнения на CPU (с.)	|	Лучшее время выполнения на GPU (с.)	|	CPU/GPU
---	|	---	|	---	|	---
3840×2160	|	26,443	|	23,2491	|	1,14
1920×1080	|	6,64199	|	5,90911	|	1,124

---
Компилировать командой:
`nvcc EasyBMP.cpp kernel.cu -O3`

***
Некоторые примеры.
![Image alt](https://github.com/JinRSA/CUDA/blob/master/Simple%20Ray%20Tracing/Images/Demo%20sample%201.jpg)

![Image alt](https://github.com/JinRSA/CUDA/blob/master/Simple%20Ray%20Tracing/Images/Randome%20sample%200.jpg)

(◉ܫ◉)
