#include <iostream>
#include <time.h>
#include "Matrix.h"
#define COUNT 10
#define HANDLE_ERROR(cudaError) HandleError(cudaError, __FILE__, __LINE__)

static void HandleError(const cudaError_t err, const char* const in_file, const int line)
{
	if (err != cudaSuccess)
	{
		printf("%s,\n� ����� %s,\n� ������: %d.\n", cudaGetErrorString(err), in_file, line);
		exit(EXIT_FAILURE);
	}
}
//#define __global__

// [r1 x c1] * [r2 x c2] = [r1 x c2].
// c1 == r2!
__global__ void MatrixMultiplication(const el_t* const in_A, const el_t* const in_B, el_t* const out_Res, const uint rowsA, const uint colsA, const uint colsB)
{
	if (blockIdx.y < rowsA && blockIdx.x < colsB)
	{
		el_t summ = 0;
		for (uint i = 0; i < colsA; ++i)
			summ += in_A[colsA * blockIdx.y + i] * in_B[colsB * i + blockIdx.x];
		out_Res[colsB * blockIdx.y + blockIdx.x] = summ;
	}
}

int main()
{
	using namespace std;
	const int q = 8192, n = 4096, m = 6144;
	Matrix A = Init(n, q);	// �������������� ������� A.
	for (el_t i = 0; i < n * q; ++i)
		A.El[i] = i + 1;
	Matrix B = Init(q, m);	// �������������� ������� B.
	for (el_t i = 0; i < q * m; ++i)
		B.El[i] = i + 1;
	Matrix MatResCPU = InitZeros(A.Row, B.Col);	// �������������� ������ ������� ��������� ��������� �� CPU.
	Matrix MatResGPU = InitZeros(A.Row, B.Col);	// �������������� ������ ������� ��������� ��������� �� GPU.
	el_t* dev_A, *dev_B, *dev_MatRes;
	// �������� ������ �� GPU ��� ������.
	HANDLE_ERROR(cudaMalloc((void**)&dev_A, A.Row * A.Col * sizeof(el_t)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_B, B.Row * B.Col * sizeof(el_t)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_MatRes, MatResGPU.Row * MatResGPU.Col * sizeof(el_t)));
	// ����������� ������� �� GPU.
	HANDLE_ERROR(cudaMemcpy(dev_A, A.El, A.Row * A.Col * sizeof(el_t), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_B, B.El, B.Row * B.Col * sizeof(el_t), cudaMemcpyHostToDevice));
	double TimesMultiplyByRow[COUNT], TimesMatrixMultiplication[COUNT];	// ������� ��� �������� ��������� ������ �� ���������� ���������.
	for (int i = 0; i < COUNT; ++i)	// �������� ����� ���������� ����� ���������� COUNT ���.
	{
		FillZeros(&MatResCPU);
		clock_t Start = clock();
		MultiplyByRow(&A, &B, &MatResCPU);	// ��������� ��������� ������ �� CPU.
		clock_t End = clock();
		TimesMultiplyByRow[i] = ((double)End - Start) / CLOCKS_PER_SEC;
		Start = clock();
		MatrixMultiplication <<<dim3(m, n), 1 >>>(dev_A, dev_B, dev_MatRes, A.Row, A.Col, B.Col);	// ��������� ��������� ������ �� GPU.
		End = clock();
		TimesMatrixMultiplication[i] = ((double)End - Start) / CLOCKS_PER_SEC;
	}
	HANDLE_ERROR(cudaMemcpy(MatResGPU.El, dev_MatRes, MatResGPU.Row * MatResGPU.Col * sizeof(el_t), cudaMemcpyDeviceToHost));	// ����������� ��������� � ���.
	// ���������� GPU ������.
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_MatRes);
	free(A.El);
	free(B.El);
	double Sum = 0;
	for (int i = 0; i < COUNT; ++i)
	{
		printf("��������� �� CPU %i:\t%f:\n", i + 1, TimesMultiplyByRow[i]);
		Sum += TimesMultiplyByRow[i];
	}
	printf("��������� �� CPU (avr.):\t%f:\n", Sum / COUNT);
	Sum = 0;
	for (int i = 0; i < COUNT; ++i)
	{
		printf("��������� �� GPU %i:\t%f:\n", i + 1, TimesMatrixMultiplication[i]);
		Sum += TimesMatrixMultiplication[i];
	}
	printf("��������� �� GPU (avr.):\t%f:\n", Sum / COUNT);
	const auto exitCode = !VerifyMatrix(&MatResCPU, &MatResGPU);
	free(MatResCPU.El);
	free(MatResGPU.El);
	return exitCode;
}

//nvcc Main.cu Matrix.cpp - O3