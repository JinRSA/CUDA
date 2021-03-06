<h1>Умножение матриц на CPU/GPU.</h1>

---
<h3>Некоторые технические характеристики.</h3>

- **CPU**: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz
- **GPU**: NVIDIA GeForce GTX 950

***
__A__ * __B__ = __C__.
* Порядок матрицы __A__ [4096×8192] (33 554 432 элементов);
* Порядок матрицы __B__ [8192×6144] (50 331 648 элементов);
* Порядок матрицы __C__ [4096×6144] (25 165 824 элементов).

---
№ эксперимента	|	Время выполнения на CPU (с.)	|	Время выполнения на GPU (с.)	|	CPU/GPU
---	|	---	|	---	|	---
№1	|	189,752418	|	0,004323	|	43894
№2	|	189,946948	|	0,004315	|	44020
№3	|	189,988219	|	0,004353	|	43645
№4	|	190,009405	|	0,004507	|	42159
№5	|	189,817485	|	0,004274	|	44412
№6	|	189,840875	|	0,004371	|	43432
№7	|	189,774178	|	0,004376	|	43367
№8	|	189,793485	|	0,004314	|	43995
№9	|	189,682986	|	0,004269	|	44433
№10	|	189,715537	|	0,004352	|	43593
**Среднее**	|	**189,832154**	|	**0,004345**	|	**43690**

---
<h4>Компилировать командой:</h4>

`nvcc Main.cu Matrix.cpp -O3`

(◉ܫ◉)
