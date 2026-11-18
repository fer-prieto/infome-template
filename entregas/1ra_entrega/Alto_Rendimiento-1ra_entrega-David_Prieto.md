# Multiplicación de Matrices de Gran Escala: Comparación CPU vs GPU con CUDA

Multiplicación de matrices densas en precisión simple (FP32)

Alumno: David Fernando Prieto Cardoso Profesor: Henry Samuel Dueñas de la Cruz Curso: Arquitecturas de Alto Rendimiento (IF557ZIN)

Fecha de realización: 8 de noviembre de 2026 Cusco, Perú

## Resumen

Este reporte documenta la primera entrega del proyecto de multiplicación de matrices de gran escala _C_ = _A × B_ en precisión simple (FP32). Se implementa desde cero la versión secuencial en CPU en C/C++ con dos órdenes de bucles —la ingenua _i_ - _j_ - _k_ y la optimizada _i_ - _k_ - _j_ — y se miden el tiempo de cómputo, el speedup y los GFLOPS para matrices de _N_ = 1024, 2048 y 4096. La versión _i_ - _k_ - _j_ alcanza un speedup de aproximadamente 36 _×_ frente a _i_ - _j_ - _k_ y entre el 11 % y el 25 % del techo práctico de la CPU (OpenBLAS a 1 hilo). Estos resultados constituyen el denominador de referencia ( _T_ CPU) para los speedups frente a los kernels CUDA de las Unidades 2 y 3.

## Índice de Contenidos

|**1.**|**Introducción y objetivos**|**1**|
|---|---|---|
||1.1.<br>Antecedentes . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . .<br>1|
|**2.**|**Descripción del algoritmo**|**1**|
||2.1.<br>Producto de matrices y FLOPs<br>. . . . . . . . . . . . . .|. . . . . . . . . .<br>1|
||2.2.<br>Disposición en memoria . . . . . . . . . . . . . . . . . . . <br>|. . . . . . . . . .<br>2|
||2.3.<br>Órdenes de bucles: i-j-k frente a i-k-j<br>. . . . . . . . . . .|. . . . . . . . . .<br>2|
|**3.**|**Entorno de prueba**|**3**|
|**4.**|**Metodología**|**3**|
|**5.**|**Resultados base**|**4**|
||5.1.<br>Tiempo de cómputo y GFLOPS . . . . . . . . . . . . . .|. . . . . . . . . .<br>4|
||5.2.<br>Speedup y porcentaje de OpenBLAS<br>. . . . . . . . . . .|. . . . . . . . . .<br>5|
||5.3.<br>Validación de correctitud . . . . . . . . . . . . . . . . . .|. . . . . . . . . .<br>5|
|**6.**|**Observaciones y limitaciones**|**5**|
||6.1.<br>Siguientes pasos . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . . .<br>6|
|**Re**|**ferencias**|**7**|
|**Ín**|**dice de Tablas**||

|1.|Entorno de prueba de la Unidad 1.<br>. . . . . . . . . . . . . . . . . . . . . . . .<br>3|
|---|---|
|2.|Tiempo de cómputo (mediana) y GFLOPS por implementación y tamaño.<br>. .<br>4|
|3.|Speedup de i-k-j frente a i-j-k y porcentaje respecto a OpenBLAS (1 hilo).<br>. .<br>5|
|4.|Validación de _i_-_k_-_j_ frente a la referencia OpenBLAS (1 hilo). . . . . . . . . . .<br>5|

## 1. Introducción y objetivos

La multiplicación de matrices densas (GEMM) es una de las operaciones más estudiadas en computación de alto rendimiento y el bloque de construcción de numerosos algoritmos numéricos y de aprendizaje automático. El objetivo de la Unidad 1 es establecer una **línea base secuencial en CPU** correcta, medible y reproducible que sirva de:

1. **Referencia de correctitud** para las versiones en GPU.

2. **Denominador de speedup** ( _S_ = _T_ CPU _/T_ GPU) en las Unidades 2 y 3.

3. **Punto de partida** para justificar la optimización con memoria compartida (tiling) y para cuantificar la distancia al techo práctico de la CPU (OpenBLAS).

El alcance se limita a precisión simple (FP32), matrices cuadradas de 1024 _×_ 1024, 2048 _×_ 2048 y 4096 _×_ 4096, y a un análisis cuantitativo de la jerarquía de memoria.

### 1.1. Antecedentes

GEMM es probablemente la operación más optimizada de la computación: las bibliotecas BLAS —y en particular el trabajo de Goto y van de Geijn sobre empaquetado de datos, tiling multinivel y micro-kernels en ensamblador— definen el estado del arte en CPU [1]. En este trabajo la implementación es **propia y escrita desde cero** , porque es precisamente lo que evalúa la consigna; las bibliotecas optimizadas (OpenBLAS) se usan únicamente como **referencia comparativa** para mostrar el techo práctico de la CPU y como referencia de validación, nunca como objeto de reimplementación. El material educativo consultado se estudia y se cita en docs/referencias.md.

## 2. Descripción del algoritmo

### 2.1. Producto de matrices y FLOPs

Dadas dos matrices _A_ y _B_ de _N × N_ , el producto _C_ = _A × B_ se define como:

<!-- Start of picture text -->
N − 1<br>C [ i ][ j ] = � A [ i ][ k ] · B [ k ][ j ] (1)<br>k =0<br><!-- End of picture text -->

Cada elemento _C_ [ _i_ ][ _j_ ] requiere _N_ multiplicaciones y _N −_ 1 sumas. Como hay _N_<sup>2</sup> elementos, el número total de operaciones en punto flotante es:

<!-- Start of picture text -->
FLOPs = 2 N 3 (2)<br><!-- End of picture text -->

El rendimiento se expresa en GFLOPS (1 GFLOP = 10<sup>9</sup> operaciones de punto flotante por segundo):

_Unidad 1: Línea base secuencial en CPU_

_IF557ZIN Arquitecturas de Alto Rendimiento_

<!-- Start of picture text -->
GFLOPS = 2 N 3 (3)<br>t · 10 9<br><!-- End of picture text -->

### 2.2. Disposición en memoria

Las matrices se almacenan como **arreglos unidimensionales en orden por filas** ( _rowmajor_ ): el elemento ( _i, j_ ) está en _m_ [ _i·N_ + _j_ ]. Esta es la misma disposición que usará CUDA, de modo que los kernels futuros y la versión CPU comparten la convención. Los datos se alinean a 64 bytes (una línea de caché) y se generan con un generador pseudoaleatorio reproducible (SplitMix64, semilla fija) en el intervalo [ _−_ 1 _,_ 1].

### 2.3. Órdenes de bucles: i-j-k frente a i-k-j

Se implementan dos variantes que recorren los mismos datos en distinto orden. _i_ **-** _j_ **-** _k_ **(ingenua, referencia de correctitud).** El bucle interno recorre _k_ :

for (i) for (j) { suma = 0; for (k) suma += A[i][k]*B[k][j]; C[i][j] = suma; }

Cada iteración lee _B_ [ _k_ ][ _j_ ] saltando una fila completa en memoria (acceso _strided_ ), lo que produce **fallos de caché por conflicto** cuando _N_ es potencia de 2 y, al ser una reducción (suma += ...), el compilador no puede reordenar las sumas en punto flotante sin -ffast-math. Resultado: apenas se vectoriza y es la versión más lenta.

_i_ **-** _k_ **-** _j_ **(optimizada, línea base oficial).** El bucle interno recorre _j_ :

for (i) for (k) { aik = A[i][k]; for (j) C[i][j] += aik * B[k][j]; }

- **Localidad espacial:** el bucle interno recorre una fila de _B_ y una fila de _C_ de forma contigua (elementos consecutivos en memoria).

- **Localidad temporal:** _aik_ queda en un registro durante todo el bucle _j_ , y la fila _i_ de _C_ se reutiliza en cada iteración de _k_ , permaneciendo en L1/L2.

- **Vectorización:** las iteraciones de _j_ son independientes, por lo que el compilador vectoriza el bucle con SIMD (en esta CPU, AVX2 con FMA: 8 _floats_ por instrucción y multiply-add fusionado).

La implementación está en include/matmul_cpu.h y src/cpu/matmul_cpu.cpp, escrita desde cero. Las funciones con interfaz C (extern Ç") reciben los punteros a _A_ , _B_ y _C_ (con __restrict), el tamaño _N_ en 64 bits, ponen _C_ a cero antes de acumular y no reservan memoria (la gestiona el llamador).

## 3. Entorno de prueba

Los valores de esta tabla provienen de results/unidad1/env_<run>.json, generado automáticamente por el arnés antes de la corrida.

Tabla 1: Entorno de prueba de la Unidad 1.

|**Componente**|**Valor**|
|---|---|
|CPU|Intel Core i7-12700KF (Alder Lake)|
|Núcleos / hilos|10 núcleos / 20 hilos (vista de WSL2)|
|SIMD soportado|SSE4.2, AVX, AVX2, FMA (sin AVX-512)|
|Caché L1d|48 KiB|
|Caché L2|1.25 MiB (por núcleo)|
|Caché L3|25 MiB (compartida)|
|RAM|15.5 GiB|
|Sistema operativo|Ubuntu 24.04.5 LTS (WSL2, kernel 6.18.54-microsoft)|
|Compilador|g++ (GCC) 13.3.0|
|Flags de compilación<br>-march=native resuelve a|-O3 -march=native -std=c++17 -Wall -Wextra -fPIC<br>alderlake (AVX2, FMA)|
|Rust / make|rustc 1.99.1 _·_ GNU Make 4.3|
|Gobernador de CPU|no disponible (WSL2)|
|OpenBLAS|0.3.26 DYNAMIC_ARCH (MAX_THREADS=64)|

**Nota sobre WSL2:** la medición se ejecutó dentro de WSL2, que no expone los sensores de temperatura ni el gobernador de frecuencias del host. Por ello el reporte no incluye temperatura ni throttling térmico (ver Sección 6).

## 4. Metodología

- **Generación de datos:** matrices en [ _−_ 1 _,_ 1] con semilla fija 42 (streams independientes para _A_ y _B_ ), idénticas en el binario C++ y en el arnés.

- **Medición de tiempo:** con std::chrono::steady_clock **dentro de C++** , rodeando exclusivamente el triple bucle (o la llamada a cblas_sgemm). No se incluye la generación de datos ni la reserva de memoria.

- **Tiempo de cómputo puro vs. total:** en CPU el tiempo reportado es el de cómputo puro. La separación «cómputo vs. transferencia Host _↔_ Device» (cudaEventRecord/cudaMemcpy) corresponde a la Unidad 2, donde sí habrá copias por PCIe.

- **Calentamiento y repeticiones:** 1 warm-up no medido + 5 repeticiones medidas por

(implementación, _N_ ); para _N_ = 4096 se usan 3 repeticiones. Se reporta la **mediana** (robusta a valores atípicos), además del mínimo, la media y la desviación estándar.

- **GFLOPS y speedup:** GFLOPS = 2 _N_<sup>3</sup> _/_ ( _t_ mediana _·_ 10<sup>9</sup> ); el speedup de referencia de esta unidad es _S_ = _Tijk/Tikj_ (efecto de la localidad de caché).

- **Validación de correctitud:** comparación contra una referencia calculada en _f_ 64 (error relativo de Frobenius y error absoluto máximo) y verificación probabilística de Freivalds; tolerancia 10<sup>_−_4</sup> . La referencia preferida es cpu_openblas_1t.

- **Control de hilos de OpenBLAS:** se mide cpu_openblas_1t (1 hilo, la comparación justa con el código secuencial) y cpu_openblas_mt (20 hilos, solo informativa como techo absoluto). El porcentaje frente a OpenBLAS se calcula contra la variante de 1 hilo.

## 5. Resultados base

Los valores provienen de results/unidad1/summary_<run>.csv. Los GFLOPS usan la mediana.

### 5.1. Tiempo de cómputo y GFLOPS

Tabla 2: Tiempo de cómputo (mediana) y GFLOPS por implementación y tamaño.

|**Imp**|**lementación**|**Tipo**|**N**|**t_mediana (s)**|**GFLOPS**|
|---|---|---|---|---|---|
|cpu|_ijk|propia|1024|2.212|0.97|
|cpu|_ikj|propia|1024|0.0614|34.95|
|cpu|_ijk|propia|2048|24.207|0.71|
|cpu|_ikj|propia|2048|0.654|26.26|
|cpu|_ijk|propia|4096|—|—|
|cpu|_ikj|propia|4096|8.882|15.47|
|cpu|_openblas_1t|referencia|1024|0.0157|137.01|
|cpu|_openblas_1t|referencia|2048|0.122|140.61|
|cpu|_openblas_1t|referencia|4096|0.959|143.38|
|cpu|_openblas_mt|referencia|1024|0.0035|611.35|
|cpu|_openblas_mt|referencia|2048|0.0237|724.88|
|cpu|_openblas_mt|referencia|4096|0.175|784.37|

cpu_ijk se omite en _N_ = 4096 por ser demasiado lenta para _N >_ 2048 (tarda _≈_ 24 s solo en _N_ = 2048, y en 4096 la matriz de 64 MiB no cabe en la caché L3 de 25 MiB).

### 5.2. Speedup y porcentaje de OpenBLAS

Tabla 3: Speedup de i-k-j frente a i-j-k y porcentaje respecto a OpenBLAS (1 hilo).

|**N**|_S_ =|_Tijk/Tikj_|**% **|**de OpenBLAS**|
|---|---|---|---|---|
|1024||36.0||25.5 %|
|2048||37.0||18.7 %|
|4096||—||10.8 %|

El speedup de _N_ = 4096 no se reporta porque cpu_ijk se omitió en ese tamaño. El porcentaje de OpenBLAS se calcula como GFLOPS _ikj/_ GFLOPS _openblas_ _1 _t ×_ 100.

### 5.3. Validación de correctitud

Tabla 4: Validación de _i_ - _k_ - _j_ frente a la referencia OpenBLAS (1 hilo).

|**Implementación**|**N**|**Error relativo**<br>**Fre**|**ivalds**|
|---|---|---|---|
|cpu_ikj|1024|5_._85_×_10<sup>_−_7</sup><br>5_._41|_×_10<sup>_−_7</sup>|
|cpu_ikj|2048|8_._45_×_10<sup>_−_7</sup><br>8_._48|_×_10<sup>_−_7</sup>|
|cpu_ikj|4096|1_._19_×_10<sup>_−_6</sup><br>1_._16|_×_10<sup>_−_6</sup>|

Todos los errores quedan varios órdenes por debajo de la tolerancia 10<sup>_−_4</sup> (el error absoluto máximo es de _≈_ 3 _._ 7 _×_ 10<sup>_−_4</sup> en _N_ = 4096).

## 6. Observaciones y limitaciones

- **Entorno WSL2 sin sensores:** la máquina de prueba es WSL2 sobre Windows, por lo que **no se pudo registrar temperatura, frecuencia ni throttling** térmico (el arnés los deja vacíos). No es una limitación del código sino del entorno virtualizado.

- **Estabilidad de la medición:** el coeficiente de variación (CV) se mantiene bajo ( _≤_ 6 _._ 4 % en casi todos los casos), lo que indica una medición estable.

- **Speedup y localidad de caché:** _i_ - _k_ - _j_ es _≈_ 36–37 _×_ más rápido que _i_ - _j_ - _k_ gracias a la localidad espacial/temporal y a la vectorización AVX2+FMA.

- **Distancia al techo de OpenBLAS:** _i_ - _k_ - _j_ alcanza 25.5 % ( _N_ = 1024), 18.7 % ( _N_ = 2048) y 10.8 % ( _N_ = 4096) del techo secuencial. La brecha **crece con** _N_ porque _i_ - _k_ - _j_ no aplica tiling multinivel, empaquetado de datos ni micro-kernels en ensamblador; cuando

_N_ excede la caché L2 (1.25 MiB), la matriz ya no cabe en caché y el rendimiento cae de _≈_ 35 a _≈_ 15 GFLOPS. Esta diferencia es justamente la motivación para el tiling en las Unidades 2 y 3, y es consistente con el modelo Roofline [2].

- **Precisión:** el análisis se limita a FP32; otras precisiones quedan como extensión.

### 6.1. Siguientes pasos

1. Implementar el **kernel CUDA naive** (un hilo por elemento, acceso directo a memoria global) en la máquina con GPU NVIDIA (RTX 4060 Ti) [3].

2. Instrumentar con cudaEventRecord (tiempo de kernel) y cudaMemcpy (tiempo de transferencia Host _↔_ Device) para separar cómputo y PCIe.

3. Comparar _S_ = _T_ CPU _/T_ GPU contra la línea base aquí establecida.

4. En la Unidad 3, optimizar con **memoria compartida (tiling)** siguiendo la técnica clásica de bloques y sincronización [4], y contrastar con cuBLAS como techo en GPU.

## Referencias

- [1] Goto, K. y van de Geijn, R. A., “Anatomy of high-performance matrix multiplication”, ACM Transactions on Mathematical Software, vol. 34, no. 3, pp. 12:1–12:25, 2008, doi: 10.1145/1356052.1356053.

- [2] Williams, S., Waterman, A., y Patterson, D. A., “Roofline: An insightful visual performance model for multicore architectures”, Communications of the ACM, vol. 52, no. 4, pp. 65–76, 2009, doi:10.1145/1498765.1498785.

- [3] Corporation, N., “Cuda c++ programming guide”, 2025, https://docs.nvidia.com/cuda /cuda-c-programming-guide/.

- [4] Kirk, D. B. y Hwu, W.-m. W., Programming Massively Parallel Processors: A Hands-on Approach. Morgan Kaufmann, 3rd ed., 2017.

- [5] Hennessy, J. L. y Patterson, D. A., Computer Architecture: A Quantitative Approach. Morgan Kaufmann, 6th ed., 2019.
