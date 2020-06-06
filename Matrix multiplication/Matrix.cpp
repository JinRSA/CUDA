#include "Matrix.h"
#include <malloc.h>
#include <iostream>
#include <cstring>

Matrix InitZeros(const uint in_Row, const uint in_Col)
{
	Matrix Mat(in_Row, in_Col);
	Mat.El = reinterpret_cast<el_t*>(calloc(in_Row * in_Col, sizeof(el_t)));
	return Mat;
}

Matrix Init(const uint in_Row, const uint in_Col)
{
	Matrix Mat(in_Row, in_Col);
	Mat.El = reinterpret_cast<el_t*>(malloc(in_Row * in_Col * sizeof(el_t)));
	return Mat;
}

void FillZeros(Matrix* const in_A)
{
	memset(in_A->El, 0, in_A->Row * in_A->Col * sizeof(el_t));
}

void MultiplyByRow(const Matrix* __restrict const in_A, const Matrix* __restrict const in_B, Matrix* __restrict const out_Res)
{
	for (uint k = 0; k < in_A->Col; ++k)
	{
		for (uint i = 0; i < in_A->Row; ++i)
		{
			for (uint j = 0; j < in_B->Col; ++j)
			{
				out_Res->El[i * out_Res->Col + j] += in_A->El[i * in_A->Col + k] * in_B->El[k * in_B->Col + j];
			}
		}
	}
}

bool VerifyMatrix(const Matrix* __restrict const in_A, const Matrix* __restrict const in_B, const uint maxErrorCount)
{
	using namespace std;
	uint errorCounter = 0;
	for (uint i = 0; i < in_A->Row; ++i)
	{
		for (uint j = 0; j < in_A->Col; ++j)
		{
			const auto id = i * in_A->Col + j;
			if (in_A->El[id] != in_B->El[id])
			{
				cout << in_A->El[id] << "!=" << in_B->El[id] << '\t' << "Row:" << i << ", Col:" << j << endl;
				if (++errorCounter >= maxErrorCount)
					return false;
			}
		}
	}
	return errorCounter == 0 ? true : false;
}