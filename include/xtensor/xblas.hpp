/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBLAS_HPP
#define XBLAS_HPP

#include <flens/flens.cxx>

namespace xt
{

namespace blas
{

    /**
     * Calculate the dot product between two vectors
     * @param a vector of n elements
     * @param b vector of n elements
     * @returns scalar result
     */
	template <class T>
	xscalar<T> dot(const xexpression<T>& a, const xexpression<T>& b)
	{
		T res;
		cxxblas::dot(a.data().size(), 
					 a.data().data() + a.offset(), a.strides()[0], 
					 b.data().data() + b.offset(), b.strides()[0], 
					 res);
		return res;
	}

	/**
	 * Calculate the matrix-vector product of matrix @a and vector @b
	 * @param a matrix of m-by-n elements
	 * @param b vector of n elements
	 * @returns vector of n elements
	 */
	template <class E1, class E2>
	E1 gemv(const xexpression<E1>& A, const xexpression<E2>& y)
	{
		const E1& dA = A.derived_cast();
		const E2& dy = y.derived_cast();
		xt::xarray<typename E1::value_type> res(dy.shape());

		cxxblas::gemv(
			cxxblas::StorageOrder::RowMajor, 
			cxxblas::Transpose::NoTrans, 
			dA.shape()[0], dA.shape()[1],
			1.f, // alpha
			dA.data().data(), dA.strides()[0],
			dy.data().data(), dy.strides()[0], 
			0.f, // beta 
			res.data().data(), 1ul);

		return res;
	}

	/**
	 * Calculate the matrix-matrix product of matrix @A and matrix @B
	 * @param A matrix of m-by-n elements
	 * @param B matrix of n-by-k elements
	 * @returns matrix of m-by-k elements
	 */
	template <class E1, class E2>
	E1 gemm(const xexpression<E1>& A, const xexpression<E2>& B)
	{
		const E1& da = A.derived_cast();
		const E2& db = B.derived_cast();
		
		E2 res(db.shape());
		cxxblas::gemm(
			cxxblas::StorageOrder::RowMajor, 
			cxxblas::Transpose::NoTrans, cxxblas::Transpose::NoTrans, 
			da.shape()[0], da.shape()[1], db.shape()[0],
			1.f, // alpha
			da.data().data(), da.strides()[0],
			db.data().data(), db.strides()[0], 
			0.f, // beta
			res.data().data(), res.strides()[0]);
		return res;
	}
}

}
#endif