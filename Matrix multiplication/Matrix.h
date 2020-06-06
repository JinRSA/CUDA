#ifndef MATRIX_H
#define MATRIX_H

typedef int el_t;
typedef unsigned uint;
struct Matrix
{
	const uint Row, Col;
	el_t* __restrict El;
	Matrix(const uint in_Row, const uint in_Col) : Row(in_Row), Col(in_Col) {}
};

Matrix InitZeros(const uint in_Row, const uint in_Col);
Matrix Init(const uint in_Row, const uint in_Col);
void FillZeros(Matrix* const in_A);
void MultiplyByRow(const Matrix* __restrict const in_A, const Matrix* __restrict const in_B, Matrix* __restrict const out_Res);
bool VerifyMatrix(const Matrix* __restrict const in_A, const Matrix* __restrict const in_B, const uint maxErrorCount = 16);
#endif // !MATRIX_H