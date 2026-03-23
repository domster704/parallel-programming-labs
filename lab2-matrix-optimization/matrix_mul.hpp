#pragma once
#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

double* allocateMatrix(int N);
void freeMatrix(double* M);
void zeroMatrix(double* M, int N);
void fillMatrix(double* M, int N);
bool compareMatrices(const double* A, const double* B, int N, double eps);
void transposeMatrix(const double* B, double* BT, int N);
double calcGFLOPS(int N, double ms);

double measure(
	void (*func)(const double*, const double*, double*, int, int),
	const double* A,
	const double* B,
	double* C,
	int N,
	int param
);

void multiplyClassic(const double* A, const double* B, double* C, int N, int unused);
void multiplyTransposed(const double* A, const double* BT, double* C, int N, int unused);
void multiplyBuffered(const double* A, const double* B, double* C, int N, int M);
void multiplyBlocked(const double* A, const double* B, double* C, int N, int S);

extern int blockUnrollM;

#endif