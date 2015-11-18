#pragma once
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/array1d.h>

#define CAST(x) thrust::raw_pointer_cast(&x[0])
#define SINGLE

typedef cusp::csr_matrix<int, double, cusp::host_memory> Matrix_CSR_h;
typedef cusp::csr_matrix<int, double, cusp::device_memory> Matrix_CSR_d;

typedef cusp::array1d<double, cusp::host_memory> Vector_h;
typedef cusp::array1d<double, cusp::device_memory> Vector_d;

typedef cusp::array1d<int, cusp::host_memory> IdxVector_h;
typedef cusp::array1d<int, cusp::device_memory> IdxVector_d;

typedef cusp::array1d<bool, cusp::device_memory> BoolVector_d;
typedef cusp::array1d<bool, cusp::host_memory> BoolVector_h;

typedef cusp::array1d<char, cusp::device_memory> CharVector_d;
typedef cusp::array1d<char, cusp::host_memory> CharVector_h;

typedef cusp::ell_matrix<int, double, cusp::device_memory> Matrix_ELL_d;
typedef cusp::ell_matrix<int, double, cusp::host_memory> Matrix_ELL_h;

typedef cusp::coo_matrix<int, double, cusp::device_memory> Matrix_COO_d;
typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix_COO_h;

typedef cusp::hyb_matrix<int, double, cusp::device_memory> Matrix_HYB_d;
typedef cusp::hyb_matrix<int, double, cusp::host_memory> Matrix_HYB_h;


