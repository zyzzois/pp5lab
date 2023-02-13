/*Параллельная, CPU.
#include <iostream>
#include <time.h>
#include <math.h>
#include <cstdlib>
#include <random>
#include <omp.h>
#define n 256

using namespace std;

// реализация матричного умножения с использованием OpenMP (Open Multi-Processing) для параллельной обработки

int main()
{
    double* matrix1 = new double[n * n];
    double* matrix2 = new double[n * n];
    double* matrix3 = new double[n * n];

    cout << "N = " << n << endl;

    int q = 1;

    srand(time(NULL));

    for (int i = 0; i < n * n; i++)
    {
        matrix1[i] = (double)rand() / RAND_MAX - 0.5;
        matrix2[i] = (double)rand() / RAND_MAX - 0.5;
    }

    double sum = 0;
    double start_time = omp_get_wtime();

	
	// основной цикл
	// элементы matrix3 вычисляются как скалярное произведение строки matrix1 и столбца matrix2
	// директива OpenMP «#pragma omp parallel for reduce(+:sum) num_threads(6)» позволяет 
	// выполнять самый внутренний цикл параллельно с 6 потоками
    for (int count = 0; count < q; count++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                sum = 0;
#pragma omp parallel for reduction(+:sum) num_threads(6)
                for (int k = 0; k < n; k++)
                {
                    sum += matrix1[i * n + k] * matrix2[j + n * k];
                }
                matrix3[i * n + j] = sum;
            }
        }
    }

	// result - норма результируюшей матрицы
    double result = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result = 0;
            result += matrix3[i * n + j] * matrix3[i * n + j];
        }
    }
    double end_time = omp_get_wtime(); // конечное время 
    cout << "Program execution time = " << (end_time - start_time) << endl;
    cout << "Norm = " << sqrt(result) << endl;

    free(matrix1);
    free(matrix2);
    free(matrix3);
    return 0;
}
*/

/*На CUDA C = AB
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//размер массива 32*32
constexpr int N = 1024;

// Размер блока = 32
constexpr int BS = 32;

constexpr int BLOCK_SIZE = 32;

#include <cstdlib>
#include <iostream>
#include <math.h>


// вычисление произведения двух матриц на ядре mul_shared.
// mul_shared использует разделяемую память для повышения производительности.
// расчеты с использованием разделяемой памяти GPU являются наиболее производительными !!!
__global__ void mul_shared(float* a, float* b, float* c, int N) {
	int i, j,
		bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		ix = bx * blockDim.x + tx, iy = by * blockDim.y + ty;
	float s = 0.;
	__shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

	for (i = 0; i < N / BLOCK_SIZE; i++) {
		as[ty][tx] = a[(by * BLOCK_SIZE + ty) * N + i * BLOCK_SIZE + tx];
		bs[ty][tx] = b[(i * BLOCK_SIZE + ty) * N + bx * BLOCK_SIZE + tx];

		__syncthreads();

		for (j = 0; j < BLOCK_SIZE; j++)
			s += as[ty][j] * bs[j][tx];

		__syncthreads();
	}
	c[iy * N + ix] = s;
}



// вычисление произведения двух матриц на ядре kernel.
// kernel выполняет умножение матриц без разделяемой памяти.

__global__ void kernel(float* a, float* b, float* c, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float s = 0.;
	for (int i = 0; i < n; i++)
		s += a[row * n + i] * b[i * n + col];
	c[row * n + col] = s;
}

// Функция генерация случайных чисел с плавающей точкой и инициализации матриц. 
void random_floats(float* a, int n) {
	for (long int i = 0; i < n; i++)
		a[i] = (float)rand() / RAND_MAX;
}

int main() {
	// если размер матрицы N не кратен 32 то программа завершается с ошибкой (по условию)
	if (N % 32 != 0) {
		std::cout << "N is not correct\n";
		exit(1);
	}
	printf("N = %d\n", N);

	srand(0);
	//переменные матриц
	float* d_a, * d_b, * d_c;
	float* a, * b, * c;
	float norma = 0.;

	// количество потоков
	dim3 threads(BS, BS);
	// количество блоков 
	dim3 blocks(N / BS, N / BS);
	int size = N * N * sizeof(float);

	// создание ивентов начала и конца (для измерения)
	cudaEvent_t start, stop;
	// переменная для подсчета времени
	float time = 0;

	// заготовка матриц размера N*N
	a = new float[N * N];
	b = new float[N * N];
	c = new float[N * N];

	//Выделение памяти
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	//Генерация матриц случайным образом
	random_floats(a, N * N);
	random_floats(b, N * N);

	// создание ивентов
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// запись ивента start.
	cudaEventRecord(start, 0);

	// передача данных между хостом (CPU) и устройством (GPU)
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// подсчет времени между событиями start и stop
	// в данном случае мы измеряли время передачи данных между хостом и устройством
	cudaEventElapsedTime(&time, start, stop);

	printf("Memory copy time: %.10f ms\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// создаем новые ивенты
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// начинаем запись 
	cudaEventRecord(start, 0);

	// << <blocks, threads >> > - конфигурация запуска ядра
	// blocks - количество блоков 
	// threads - количество потоков на блок
	mul_shared << <blocks, threads >> > (d_a, d_b, d_c, N);
	//kernel <<<blocks, threads>>> (d_a, d_b, d_c, N);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// в данном случае мы измеряли время подсчета на ядре mul_shared или на kernel
	cudaEventElapsedTime(&time, start, stop);

	printf("Computing time : %.10f ms\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// проверяем есть ли ошибка
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) 
		std::cout << "Cuda error: " << cudaGetErrorString(error) << std::endl;
	

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			norma += c[N * i + j] * c[N * i + j];
	}
	std::cout << "norma = " << sqrt(norma) << std::endl;

	//Освобождение памяти
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}*/

/*На CUDA C = AA ^ T
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr int N = 1024;
constexpr int BS = 32;

constexpr int BLOCK_SIZE = 32;

using namespace std;

#include <cstdlib>
#include <iostream>

__global__ void mul_transp_shared_1(float* a, float* c, int N) //no bank conflict, no coalescing
{
	int i, j,
		bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		ix = bx * blockDim.x + tx, iy = by * blockDim.y + ty;
	float s = 0.;
	__shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

	for (i = 0; i < N / BLOCK_SIZE; i++) {
		as[ty][tx] = a[(by * BLOCK_SIZE + ty) * N + i * BLOCK_SIZE + tx];
		bs[ty][tx] = a[(bx * BLOCK_SIZE + tx) * N + i * BLOCK_SIZE + ty];

		__syncthreads();

		for (j = 0; j < BLOCK_SIZE; j++)
			s += as[ty][j] * bs[j][tx];

		__syncthreads();
	}
	c[iy * N + ix] = s;
}

__global__ void mul_transp_shared_2(float* a, float* c, int N) { //bank conflict, coalescing
	int i, j,
		bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		ix = bx * blockDim.x + tx, iy = by * blockDim.y + ty;
	float s = 0.;
	__shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

	for (i = 0; i < N / BLOCK_SIZE; i++) {
		as[ty][tx] = a[(i * BLOCK_SIZE + ty) * N + bx * BLOCK_SIZE + tx];
		bs[tx][ty] = a[(i * BLOCK_SIZE + ty) * N + bx * BLOCK_SIZE + tx];

		__syncthreads();

		for (j = 0; j < BLOCK_SIZE; j++)
			s += as[ty][j] * bs[j][tx];

		__syncthreads();
	}
	c[iy * N + ix] = s;
}

__global__ void mul_transp_shared_3(float* a, float* c, int N) //no bank conflict, coalescing
{
	int i, j,
		bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		ix = bx * blockDim.x + tx, iy = by * blockDim.y + ty;
	float s = 0.;
	__shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float bs[BLOCK_SIZE][BLOCK_SIZE + 1];

	for (i = 0; i < N / BLOCK_SIZE; i++) {
		as[ty][tx] = a[(i * BLOCK_SIZE + ty) * N + bx * BLOCK_SIZE + tx];
		bs[tx][ty] = a[(i * BLOCK_SIZE + ty) * N + bx * BLOCK_SIZE + tx];

		__syncthreads();

		for (j = 0; j < BLOCK_SIZE; j++)
			s += as[ty][j] * bs[j][tx];

		__syncthreads();
	}
	c[iy * N + ix] = s;
}

void random_floats(float* a, int n) {
	for (long int i = 0; i < n; i++)
		a[i] = (float)rand() / RAND_MAX;
}

int main() {
	if (N % 32 != 0) {
		cout << "N is not correct\n";
		exit(1);
	}
	printf("N = %d\n", N);

	srand(0);
	float* d_a, * d_b, * d_c;
	float* a, * b, * c;
	float norma = 0.;
	dim3 threads(BS, BS);
	dim3 blocks(N / BS, N / BS);
	int size = N * N * sizeof(float);

	cudaEvent_t start, stop;
	float time = 0;

	a = new float[N * N];
	b = new float[N * N];
	c = new float[N * N];

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	random_floats(a, N * N);
	random_floats(b, N * N);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Memory copy time: %.10f ms\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	mul_transp_shared_1 «<blocks, threads»>(d_a, d_c, N);
	//mul_transp_shared_2 «<blocks, threads»>(d_a, d_c, N);
	//mul_transp_shared_3 «<blocks, threads»> (d_a, d_c, N);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Computing time : %.10f ms\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
		cout << "Cuda error: " << cudaGetErrorString(error) << endl;
	

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			norma += c[N * i + j] * c[N * i + j];
	}
	cout << "norma = " << sqrt(norma) << endl;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}*/