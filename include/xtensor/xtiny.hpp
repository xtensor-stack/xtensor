/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XTINY_HPP
#define XTENSOR_XTINY_HPP

#include "xconcepts.hpp"
#include "xexception.hpp"
#include <iosfwd>   // ostream
#include <algorithm>
#include <memory>
#include <iterator>
#include <utility>
#include <complex>
#include <cmath>
#include <cstdlib>

#ifdef XTENSOR_CHECK_BOUNDS
    #define XTENSOR_ASSERT_INSIDE(array, diff) \
      xtensor_precondition(diff >= 0 && diff < array.size(), "Index out of bounds")
#else
    #define XTENSOR_ASSERT_INSIDE(array, diff)
#endif

namespace xt {

// FIXME: move these to the appropriate places

using std::swap;
using std::sqrt;
using std::abs;

using ArrayIndex = std::ptrdiff_t;
static const int runtime_size  = -1;
enum SkipInitialization { DontInit };
enum ReverseCopyTag { ReverseCopy };

///////////////////////////////////////////////////////////////
// UninitializedMemoryTraits

template<class T>
struct UninitializedMemoryTraits
{
    static const bool value = std::is_scalar<T>::value || std::is_pod<T>::value;
};

template <class VALUETYPE, int M=runtime_size, int ... N>
class TinyArray;

template <class VALUETYPE, int M=runtime_size, int ... N>
class TinyArrayView;

/**********************************************************/
/*                                                        */
/*                         abs()                          */
/*                                                        */
/**********************************************************/

// define the missing variants of abs() to avoid 'ambiguous overload'
// errors in template functions
#define XTENSOR_DEFINE_UNSIGNED_ABS(T) \
    inline T abs(T t) { return t; }

XTENSOR_DEFINE_UNSIGNED_ABS(bool)
XTENSOR_DEFINE_UNSIGNED_ABS(unsigned char)
XTENSOR_DEFINE_UNSIGNED_ABS(unsigned short)
XTENSOR_DEFINE_UNSIGNED_ABS(unsigned int)
XTENSOR_DEFINE_UNSIGNED_ABS(unsigned long)
XTENSOR_DEFINE_UNSIGNED_ABS(unsigned long long)

#undef XTENSOR_DEFINE_UNSIGNED_ABS

#define XTENSOR_DEFINE_MISSING_ABS(T) \
    inline T abs(T t) { return t < 0 ? static_cast<T>(-t) : t; }

XTENSOR_DEFINE_MISSING_ABS(signed char)
XTENSOR_DEFINE_MISSING_ABS(signed short)

#if defined(_MSC_VER) && _MSC_VER < 1600
XTENSOR_DEFINE_MISSING_ABS(signed long long)
#endif

#undef XTENSOR_DEFINE_MISSING_ABS


template <class T1, class T2 = T1>
using PromoteType = decltype((*((T1*)0)) * (*((T2*)0)));

///////////////////////////////////////////////////////////////
// BoolPromote

    // replace 'bool' with 'unsigned char'
template <class T>
using BoolPromote = typename std::conditional<std::is_same<T, bool>::value, unsigned char, T>::type;

template <class T>
using RealPromoteType = decltype(sqrt(*((T*)0)));

///////////////////////////////////////////////////////////////
// NormTraits

template<class T>
struct NormTraits;

namespace detail {

template <class T, bool scalar = std::is_arithmetic<T>::value>
struct NormOfScalarImpl;

template <class T>
struct NormOfScalarImpl<T, false>
{
    static const bool value = false;
    typedef void *            NormType;
    typedef void *            SquaredNormType;
};

template <class T>
struct NormOfScalarImpl<T, true>
{
    static const bool value = true;
    typedef T                 NormType;
    typedef typename
        std::conditional<std::is_integral<T>::value,
            typename std::conditional<sizeof(T) >= 4,
                uint64_t,
                uint32_t>::type,
            T>::type
        SquaredNormType;
};

template <class T, bool integral = std::is_integral<T>::value,
                   bool floating = std::is_floating_point<T>::value>
struct NormOfArrayElementsImpl;

template <>
struct NormOfArrayElementsImpl<void *, false, false>
{
    typedef void *  NormType;
    typedef void *  SquaredNormType;
};

template <class T>
struct NormOfArrayElementsImpl<T, false, false>
{
    typedef typename NormTraits<T>::NormType         NormType;
    typedef typename NormTraits<T>::SquaredNormType  SquaredNormType;
};

template <class T>
struct NormOfArrayElementsImpl<T, true, false>
{
    static_assert(!std::is_same<T, char>::value,
       "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

    typedef double              NormType;
    typedef uint64_t            SquaredNormType;
};

template <class T>
struct NormOfArrayElementsImpl<T, false, true>
{
    typedef double              NormType;
    typedef double              SquaredNormType;
};

template <>
struct NormOfArrayElementsImpl<long double, false, true>
{
    typedef long double         NormType;
    typedef long double         SquaredNormType;
};

template <class ARRAY>
struct NormOfVectorImpl
{
    static void * test(...);

    template <class U>
    static typename U::value_type test(U*, typename U::value_type * = 0);

    typedef decltype(test((ARRAY*)0)) T;

    static const bool value = !std::is_same<T, void*>::value;

    typedef typename NormOfArrayElementsImpl<T>::NormType         NormType;
    typedef typename NormOfArrayElementsImpl<T>::SquaredNormType  SquaredNormType;
};


} // namespace detail

    /* NormTraits<T> implement the following default rules, which are
       designed to minimize the possibility of overflow:
        * T is an integer type:
               NormType is T itself,
               SquaredNormType is 'uint32_t' (sizeof(T) < 4) or 'uint64_t' (otherwise)
        * T is a floating-point type:
               NormType and SquaredNormType are T itself,
        * T is a container of 'long double' elements:
               NormType and SquaredNormType are 'long double'
        * T is a container of another floating-point type:
               NormType and SquaredNormType are 'double',
        * T is a container of integer elements:
               NormType is 'double',
               SquaredNormType is 'uint64_t'
        * T is a container of some other type:
               NormType is the element's norm type,
               SquaredNormType is the element's squared norm type
       Containers are recognized by having an embedded typedef 'value_type'.

       To change the behavior for a particular case or extend it to cases
       not covered here, simply specialize the NormTraits template.
    */
template<class T>
struct NormTraits
{
    static_assert(!std::is_same<T, char>::value,
       "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

    typedef detail::NormOfScalarImpl<T>   NormOfScalar;
    typedef detail::NormOfVectorImpl<T>   NormOfVector;

    static const bool value = NormOfScalar::value || NormOfVector::value;

    static_assert(value, "NormTraits<T> are undefined for type T.");

    typedef typename std::conditional<NormOfVector::value,
                typename NormOfVector::NormType,
                typename NormOfScalar::NormType>::type
            NormType;

    typedef typename std::conditional<NormOfVector::value,
                typename NormOfVector::SquaredNormType,
                typename NormOfScalar::SquaredNormType>::type
            SquaredNormType;
};

template <class T>
using SquaredNormType = typename NormTraits<T>::SquaredNormType;

template <class T>
using NormType = typename NormTraits<T>::NormType;

namespace mathfunctions {

using std::abs;
using std::fabs;

using std::cos;
using std::sin;
using std::tan;
// using std::sin_pi;
// using std::cos_pi;
using std::acos;
using std::asin;
using std::atan;

using std::cosh;
using std::sinh;
using std::tanh;
using std::acosh;
using std::asinh;
using std::atanh;

using std::sqrt;
using std::cbrt;
// using std::sqrti;
// using std::sq;
// using std::elementwiseNorm;
// using std::elementwiseSquaredNorm;

using std::exp;
using std::exp2;
using std::expm1;
using std::log;
using std::log2;
using std::log10;
using std::log1p;
using std::logb;
using std::ilogb;

using std::ceil;
using std::floor;
using std::trunc;
using std::round;
using std::lround;
using std::llround;
// using std::roundi;
// using std::even;
// using std::odd;
// using std::sign;
// using std::signi;

using std::erf;
using std::erfc;
using std::erfc;
using std::tgamma;
using std::lgamma;
// using std::loggamma;

using std::conj;
using std::real;
using std::imag;
using std::arg;

using std::atan2;
using std::copysign;
using std::fdim;
using std::fmax;
using std::fmin;
using std::fmod;
using std::hypot;
using std::pow;

template <class V1, class V2>
inline
std::enable_if_t<std::is_floating_point<PromoteType<V1, V2> >::value, bool>
isclose(V1 v1, V2 v2, double rtol = 1e-05, double atol = 1e-08)
{
    using P = PromoteType<V1, V2>;
    return abs(v1-v2) <= (atol + rtol * std::max(abs(static_cast<P>(v1)), abs(static_cast<P>(v2))));
}

    /** \brief The square function.

        <tt>sq(x) = x*x</tt> is needed so often that it makes sense to define it as a function.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T,
          XTENSOR_REQUIRE<std::is_arithmetic<T>::value> >
inline PromoteType<T>
sq(T t)
{
    return t*t;
}

/**********************************************************/
/*                                                        */
/*                         min()                          */
/*                                                        */
/**********************************************************/

    /** \brief A proper minimum function.

        The <tt>std::min</tt> template matches everything -- this is way too
        greedy to be useful. VIGRA implements the basic <tt>min</tt> function
        only for arithmetic types and provides explicit overloads for everything
        else. Moreover, VIGRA's <tt>min</tt> function also computes the minimum
        between two different types, as long as they have a <tt>std::common_type</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2,
          XTENSOR_REQUIRE<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value> >
inline std::common_type_t<T1, T2>
min(T1 const & t1, T2 const & t2)
{
    using T = std::common_type_t<T1, T2>;
    return std::min(static_cast<T>(t1), static_cast<T>(t2));
}

template <class T,
          XTENSOR_REQUIRE<std::is_arithmetic<T>::value> >
inline T const &
min(T const & t1, T const & t2)
{
    return std::min(t1, t2);
}

/**********************************************************/
/*                                                        */
/*                         max()                          */
/*                                                        */
/**********************************************************/

    /** \brief A proper maximum function.

        The <tt>std::max</tt> template matches everything -- this is way too
        greedy to be useful. VIGRA implements the basic <tt>max</tt> function
        only for arithmetic types and provides explicit overloads for everything
        else. Moreover, VIGRA's <tt>max</tt> function also computes the maximum
        between two different types, as long as they have a <tt>std::common_type</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2,
          XTENSOR_REQUIRE<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value> >
inline std::common_type_t<T1, T2>
max(T1 const & t1, T2 const & t2)
{
    using T = std::common_type_t<T1, T2>;
    return std::max(static_cast<T>(t1), static_cast<T>(t2));
}

template <class T,
          XTENSOR_REQUIRE<std::is_arithmetic<T>::value> >
inline T const &
max(T const & t1, T const & t2)
{
    return std::max(t1, t2);
}

/**********************************************************/
/*                                                        */
/*                   floor(), ceil()                      */
/*                                                        */
/**********************************************************/

    // add missing floor() and ceil() overloads for integral types

#define XTENSOR_DEFINE_INTEGER_FLOOR_CEIL(T) \
    inline T floor(signed T t) { return t; } \
    inline T floor(unsigned T t) { return t; } \
    inline T ceil(signed T t) { return t; } \
    inline T ceil(unsigned T t) { return t; }

XTENSOR_DEFINE_INTEGER_FLOOR_CEIL(char)
XTENSOR_DEFINE_INTEGER_FLOOR_CEIL(short)
XTENSOR_DEFINE_INTEGER_FLOOR_CEIL(int)
XTENSOR_DEFINE_INTEGER_FLOOR_CEIL(long)
XTENSOR_DEFINE_INTEGER_FLOOR_CEIL(long long)

#undef XTENSOR_DEFINE_INTEGER_FLOOR_CEIL

/**********************************************************/
/*                                                        */
/*                           dot()                        */
/*                                                        */
/**********************************************************/

// scalar dot is needed for generic functions that should work with
// scalars and vectors alike

#define XTENSOR_DEFINE_SCALAR_DOT(T) \
    inline PromoteType<T> dot(T l, T r) { return l*r; }

XTENSOR_DEFINE_SCALAR_DOT(unsigned char)
XTENSOR_DEFINE_SCALAR_DOT(unsigned short)
XTENSOR_DEFINE_SCALAR_DOT(unsigned int)
XTENSOR_DEFINE_SCALAR_DOT(unsigned long)
XTENSOR_DEFINE_SCALAR_DOT(unsigned long long)
XTENSOR_DEFINE_SCALAR_DOT(signed char)
XTENSOR_DEFINE_SCALAR_DOT(signed short)
XTENSOR_DEFINE_SCALAR_DOT(signed int)
XTENSOR_DEFINE_SCALAR_DOT(signed long)
XTENSOR_DEFINE_SCALAR_DOT(signed long long)
XTENSOR_DEFINE_SCALAR_DOT(float)
XTENSOR_DEFINE_SCALAR_DOT(double)
XTENSOR_DEFINE_SCALAR_DOT(long double)

#undef XTENSOR_DEFINE_SCALAR_DOT

/**********************************************************/
/*                                                        */
/*                           pow()                        */
/*                                                        */
/**********************************************************/

// support 'double' exponents for all floating point versions of pow()
inline float pow(float v, double e)
{
    return std::pow(v, (float)e);
}

inline long double pow(long double v, double e)
{
    return std::pow(v, (long double)e);
}

/**********************************************************/
/*                                                        */
/*                         roundi()                       */
/*                                                        */
/**********************************************************/

    // round and cast to nearest long int
inline long int roundi(double t)
{
     return lround(t);
}

/**********************************************************/
/*                                                        */
/*                       even(), odd()                    */
/*                                                        */
/**********************************************************/

template <class T,
          XTENSOR_REQUIRE<std::is_integral<T>::value> >
inline bool
even(T const t)
{
    return (t&1) == 0;
}

template <class T,
          XTENSOR_REQUIRE<std::is_integral<T>::value> >
inline bool
odd(T const t)
{
    return (t&1) != 0;
}

/**********************************************************/
/*                                                        */
/*                   sin_pi(), cos_pi()                   */
/*                                                        */
/**********************************************************/

    /** \brief sin(pi*x).

        Essentially calls <tt>std::sin(M_PI*x)</tt> but uses a more accurate implementation
        to make sure that <tt>sin_pi(1.0) == 0.0</tt> (which does not hold for
        <tt>std::sin(M_PI)</tt> due to round-off error), and <tt>sin_pi(0.5) == 1.0</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class REAL,
          XTENSOR_REQUIRE<std::is_floating_point<REAL>::value> >
REAL sin_pi(REAL x)
{
    if(x < 0.0)
        return -sin_pi(-x);
    if(x < 0.5)
        return std::sin(numeric_constants<REAL>::PI * x);

    bool invert = false;
    if(x < 1.0)
    {
        invert = true;
        x = -x;
    }

    REAL rem = std::floor(x);
    if(odd((int)rem))
        invert = !invert;
    rem = x - rem;
    if(rem > 0.5)
        rem = 1.0 - rem;
    if(rem == 0.5)
        rem = REAL(1);
    else
        rem = std::sin(numeric_constants<REAL>::PI * rem);
    return invert
              ? -rem
              : rem;
}

    /** \brief cos(pi*x).

        Essentially calls <tt>std::cos(M_PI*x)</tt> but uses a more accurate implementation
        to make sure that <tt>cos_pi(1.0) == -1.0</tt> and <tt>cos_pi(0.5) == 0.0</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class REAL,
          XTENSOR_REQUIRE<std::is_floating_point<REAL>::value> >
REAL cos_pi(REAL x)
{
    return sin_pi(x+0.5);
}

/**********************************************************/
/*                                                        */
/*                     gamma(), loggamma()                */
/*                                                        */
/**********************************************************/

inline double gamma(double x)
{
    xtensor_precondition(x <= 171.0,
        "gamma(): argument cannot exceed 171.0.");

    xtensor_precondition(x > 0.0 || fmod(x, 1.0) != 0,
         "gamma(): gamma function is undefined for 0 and negative integers.");

    return tgamma(x);
}

inline double loggamma(double x)
{
    xtensor_precondition(x > 0.0,
        "loggamma(): argument must be positive.");

    xtensor_precondition(x <= 1.0e307,
        "loggamma(): argument must not exceed 1e307.");

    return lgamma(x);
}

} // namespace mathfunctions

using mathfunctions::sq;

/**********************************************************/
/*                                                        */
/*                     squaredNorm()                      */
/*                                                        */
/**********************************************************/

template <class T>
inline SquaredNormType<std::complex<T> >
squaredNorm(std::complex<T> const & t)
{
    return sq(t.real()) + sq(t.imag());
}

/**********************************************************/
/*                                                        */
/*                          norm()                        */
/*                                                        */
/**********************************************************/

    /** \brief The norm of a numerical object.

        For scalar types: implemented as <tt>abs(t)</tt><br>
        otherwise: implemented as <tt>sqrt(squaredNorm(t))</tt>.

        <b>\#include</b> \<vigra2/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T>
inline auto
norm(T const & t) -> decltype(sqrt(squaredNorm(t)))
{
    return sqrt(squaredNorm(t));
}

#define XTENSOR_DEFINE_NORM(T) \
    inline SquaredNormType<T> squaredNorm(T t) { return sq(t); } \
    inline NormType<T> norm(T t) { return abs(t); } \
    inline SquaredNormType<T> sizeDividedSquaredNorm(T t) { return sq(t); } \
    inline NormType<T> sizeDividedNorm(T t) { return abs(t); }

XTENSOR_DEFINE_NORM(signed char)
XTENSOR_DEFINE_NORM(unsigned char)
XTENSOR_DEFINE_NORM(short)
XTENSOR_DEFINE_NORM(unsigned short)
XTENSOR_DEFINE_NORM(int)
XTENSOR_DEFINE_NORM(unsigned int)
XTENSOR_DEFINE_NORM(long)
XTENSOR_DEFINE_NORM(unsigned long)
XTENSOR_DEFINE_NORM(long long)
XTENSOR_DEFINE_NORM(unsigned long long)
XTENSOR_DEFINE_NORM(float)
XTENSOR_DEFINE_NORM(double)
XTENSOR_DEFINE_NORM(long double)

#undef XTENSOR_DEFINE_NORM

namespace tags {

/********************************************************/
/*                                                      */
/*                       tags::size                     */
/*                                                      */
/********************************************************/

    // Support for tags::size keyword argument
    // to disambiguate array sizes from initial values.
struct SizeProxy
{
    ArrayIndex value;
};

struct SizeTag
{
    SizeProxy operator=(ArrayIndex s) const
    {
        return {s};
    }

    SizeProxy operator()(ArrayIndex s) const
    {
        return {s};
    }
};

namespace {

SizeTag size;

}

} // namespace tags

// mask cl.exe shortcomings [begin]
// #if defined(_MSC_VER)
// #pragma warning( push )
// #pragma warning( disable : 4503 )
// #endif


// FIXME: document this
template <ArrayIndex LEVEL, int ... N>
struct TinyShapeImpl;

template <ArrayIndex LEVEL, int N, int ... REST>
struct TinyShapeImpl<LEVEL, N, REST...>
{
    static_assert(N >= 0, "TinyArrayBase(): array must have non-negative shape.");
    using NextType = TinyShapeImpl<LEVEL+1, REST...>;

    static const ArrayIndex level      = LEVEL;
    static const ArrayIndex stride     = NextType::total_size;
    static const ArrayIndex total_size = N * stride;
    static const ArrayIndex alloc_size = total_size;

    static ArrayIndex offset(ArrayIndex const * coord)
    {
        return stride*coord[level] + NextType::offset(coord);
    }

    template <class ... V>
    static ArrayIndex offset(ArrayIndex i, V...rest)
    {
        return stride*i + NextType::offset(rest...);
    }
};

template <ArrayIndex LEVEL, int N>
struct TinyShapeImpl<LEVEL, N>
{
    static_assert(N >= 0, "TinyArrayBase(): array must have non-negative shape.");
    static const ArrayIndex level      = LEVEL;
    static const ArrayIndex stride     = 1;
    static const ArrayIndex total_size = N;
    static const ArrayIndex alloc_size = total_size;

    static ArrayIndex offset(ArrayIndex const * coord)
    {
        return coord[level];
    }

    static ArrayIndex offset(ArrayIndex i)
    {
        return i;
    }
};

template <ArrayIndex LEVEL>
struct TinyShapeImpl<LEVEL, 0>
{
    static const ArrayIndex level      = LEVEL;
    static const ArrayIndex stride     = 1;
    static const ArrayIndex total_size = 0;
    static const ArrayIndex alloc_size = 1;

    static ArrayIndex offset(ArrayIndex const * coord)
    {
        return coord[level];
    }

    static ArrayIndex offset(ArrayIndex i)
    {
        return i;
    }
};

template <int ... N>
struct TinySize
{
    static const ArrayIndex value = TinyShapeImpl<0, N...>::total_size;
    static const ArrayIndex ndim  = sizeof...(N);
};

namespace detail {

template <int N0, int ... N>
struct TinyArrayIsStatic
{
    static const int ndim = sizeof...(N)+1;
    static const bool value = ndim > 1 || N0 != runtime_size;
};

} // namespace detail

#define XTENSOR_ASSERT_RUNTIME_SIZE(SHAPE, PREDICATE, MESSAGE) \
    if(detail::TinyArrayIsStatic<SHAPE>::value) {} else \
        xtensor_precondition(PREDICATE, MESSAGE)


/********************************************************/
/*                                                      */
/*                    TinyArrayBase                     */
/*                                                      */
/********************************************************/

/** \brief Base class for fixed size vectors and matrices.

    This class contains functionality shared by
    \ref TinyArray and \ref TinyArrayView, and enables these classes
    to be freely mixed within expressions. It is typically not used directly.

    <b>\#include</b> \<vigra/tinyarray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, class DERIVED, int ... N>
class TinyArrayBase
: public TinyArrayTag
{
  protected:
    using ShapeHelper = TinyShapeImpl<0, N...>;

    static const bool derived_is_array = std::is_same<DERIVED, TinyArray<VALUETYPE, N...> >::value;
    using data_array_type = typename std::conditional<derived_is_array,
                                                VALUETYPE[ShapeHelper::alloc_size],
                                                VALUETYPE *>::type;

    template <int LEVEL, class ... V2>
    void initImpl(VALUETYPE v1, V2... v2)
    {
        data_[LEVEL] = v1;
        initImpl<LEVEL+1>(v2...);
    }

    template <int LEVEL>
    void initImpl(VALUETYPE v1)
    {
        data_[LEVEL] = v1;
    }

    template <class ITERATOR,
              XTENSOR_REQUIRE<IteratorConcept<ITERATOR>::value>>
    void initImpl(ITERATOR i)
    {
        for(ArrayIndex k=0; k < static_size; ++k, ++i)
            data_[k] = static_cast<VALUETYPE>(*i);
    }

  public:

    template <class NEW_VALUETYPE>
    using AsType = TinyArray<NEW_VALUETYPE, N...>;

    using value_type             = VALUETYPE;
    using const_value_type       = typename std::add_const<VALUETYPE>::type;
    using reference              = value_type &;
    using const_reference        = const_value_type &;
    using pointer                = value_type *;
    using const_pointer          = const_value_type *;
    using iterator               = value_type *;
    using const_iterator         = const_value_type *;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using index_type             = TinyArray<ArrayIndex, sizeof...(N)>;

    static constexpr ArrayIndex static_ndim  = sizeof...(N);
    static constexpr ArrayIndex static_size  = ShapeHelper::total_size;
    static const bool may_use_uninitialized_memory =
                                   UninitializedMemoryTraits<VALUETYPE>::value;

    // constructors

    constexpr TinyArrayBase(TinyArrayBase const &) = default;

  protected:

    TinyArrayBase(SkipInitialization)
    {}

    // constructors to be used by TinyArray

    template <class OTHER, class OTHER_DERIVED>
    TinyArrayBase(TinyArrayBase<OTHER, OTHER_DERIVED, N...> const & other)
    {
        xtensor_precondition(size() == other.size(),
                      "TinyArrayBase(): shape mismatch.");
        for(int i=0; i<static_size; ++i)
            data_[i] = static_cast<value_type>(other[i]);
    }

    // constructor for zero or one argument
    explicit TinyArrayBase(value_type v = value_type())
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
    }

    // // constructor for two or more arguments
    // template <class ... V>
    // constexpr TinyArrayBase(value_type v0, value_type v1, V ... v)
    // : data_{VALUETYPE(v0), VALUETYPE(v1), VALUETYPE(v)...}
    // {
        // static_assert(sizeof...(V)+2 == static_size,
                      // "TinyArrayBase(): number of constructor arguments contradicts size().");
    // }

    template <class U>
    explicit TinyArrayBase(U const * u)
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = static_cast<value_type>(u[i]);
    }

    template <class U>
    TinyArrayBase(U const * u, ReverseCopyTag)
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = static_cast<value_type>(u[static_size-1-i]);
    }

        // for compatibility with TinyArrayBase<..., runtime_size>
    template <class U>
    TinyArrayBase(U const * u, U const * /* end */, ReverseCopyTag)
    : TinyArrayBase(u, ReverseCopy)
    {}

  public:

    // assignment

    TinyArrayBase & operator=(TinyArrayBase const &) = default;

    TinyArrayBase & operator=(value_type v)
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
        return *this;
    }

    TinyArrayBase & operator=(value_type const (&v)[static_size])
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v[i];
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED>
    TinyArrayBase & operator=(TinyArrayBase<OTHER, OTHER_DERIVED, N...> const & other)
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = static_cast<value_type>(other[i]);
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED>
    TinyArrayBase & operator=(TinyArrayBase<OTHER, OTHER_DERIVED, runtime_size> const & other)
    {
        xtensor_precondition(size() == other.size(),
            "TinyArrayBase::operator=(): size mismatch.");
        for(int i=0; i<size(); ++i)
            data_[i] = static_cast<value_type>(other[i]);
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED, int ... M>
    constexpr bool
    sameShape(TinyArrayBase<OTHER, OTHER_DERIVED, M...> const & other) const
    {
        return false;
    }

    template <class OTHER, class OTHER_DERIVED>
    constexpr bool
    sameShape(TinyArrayBase<OTHER, OTHER_DERIVED, N...> const & other) const
    {
        return true;
    }

    template <class OTHER, class OTHER_DERIVED>
    bool
    sameShape(TinyArrayBase<OTHER, OTHER_DERIVED, runtime_size> const & other) const
    {
        return sizeof...(N) == 1 && size() == other.size();
    }

    DERIVED & init(value_type v = value_type())
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
        return static_cast<DERIVED &>(*this);
    }

    template <class ... V>
    DERIVED & init(value_type v0, value_type v1, V... v)
    {
        static_assert(sizeof...(V)+2 == static_size,
                      "TinyArrayBase::init(): wrong number of arguments.");
        initImpl<0>(v0, v1, v...);
        return static_cast<DERIVED &>(*this);
    }

    template <class Iterator>
    DERIVED & init(Iterator first, Iterator end)
    {
        ArrayIndex range = std::distance(first, end);
        if(static_size < range)
            range = static_size;
        for(ArrayIndex i=0; i<range; ++i, ++first)
            data_[i] = static_cast<value_type>(*first);
        return static_cast<DERIVED &>(*this);
    }

    // index access

    reference operator[](ArrayIndex i)
    {
        return data_[i];
    }

    constexpr const_reference operator[](ArrayIndex i) const
    {
        return data_[i];
    }

    reference at(ArrayIndex i)
    {
        if(i < 0 || i >= static_size)
            throw std::out_of_range("TinyArrayBase::at()");
        return data_[i];
    }

    const_reference at(ArrayIndex i) const
    {
        if(i < 0 || i >= static_size)
            throw std::out_of_range("TinyArrayBase::at()");
        return data_[i];
    }

    reference operator[](ArrayIndex const (&i)[static_ndim])
    {
        return data_[ShapeHelper::offset(i)];
    }

    constexpr const_reference operator[](ArrayIndex const (&i)[static_ndim]) const
    {
        return data_[ShapeHelper::offset(i)];
    }

    reference at(ArrayIndex const (&i)[static_ndim])
    {
        return at(ShapeHelper::offset(i));
    }

    const_reference at(ArrayIndex const (&i)[static_ndim]) const
    {
        return at(ShapeHelper::offset(i));
    }

    reference operator[](index_type const & i)
    {
        return data_[ShapeHelper::offset(i.data())];
    }

    constexpr const_reference operator[](index_type const & i) const
    {
        return data_[ShapeHelper::offset(i.data())];
    }

    reference at(index_type const & i)
    {
        return at(ShapeHelper::offset(i.data()));
    }

    const_reference at(index_type const & i) const
    {
        return at(ShapeHelper::offset(i.data()));
    }

    template <class ... V>
    reference operator()(V...v)
    {
        static_assert(sizeof...(V) == static_ndim,
                      "TinyArrayBase::operator(): wrong number of arguments.");
        return data_[ShapeHelper::offset(v...)];
    }

    template <class ... V>
    constexpr const_reference operator()(V...v) const
    {
        static_assert(sizeof...(V) == static_ndim,
                      "TinyArrayBase::operator(): wrong number of arguments.");
        return data_[ShapeHelper::offset(v...)];
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= static_size</tt>.
            Only available if <tt>static_ndim == 1</tt>.
        */
    template <int FROM, int TO>
    TinyArrayView<value_type, TO-FROM>
    subarray() const
    {
        static_assert(sizeof...(N) == 1,
            "TinyArrayBase::subarray(): array must be 1-dimensional.");
        static_assert(FROM >= 0 && FROM < TO && TO <= static_size,
            "TinyArrayBase::subarray(): range out of bounds.");
        return TinyArrayView<value_type, TO-FROM>(const_cast<VALUETYPE*>(data_)+FROM);
    }

    TinyArrayView<value_type, runtime_size>
    subarray(ArrayIndex FROM, ArrayIndex TO) const
    {
        xtensor_precondition(FROM >= 0 && FROM < TO && TO <= static_size,
                      "TinyArrayBase::subarray(): range out of bounds.");
        return TinyArrayView<value_type, runtime_size>(TO-FROM, const_cast<VALUETYPE*>(data_)+FROM);
    }

    template<int M = static_ndim>
    TinyArray<value_type, static_size-1>
    erase(ArrayIndex m) const
    {
        static_assert(sizeof...(N) == 1,
            "TinyArrayBase::erase(): array must be 1-dimensional.");
        xtensor_precondition(m >= 0 && m < static_size, "TinyArray::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+").");
        TinyArray<value_type, static_size-1> res(static_size-1, DontInit);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        for(ArrayIndex k=m; k<static_size-1; ++k)
            res[k] = data_[k+1];
        return res;
    }

    template<int M = static_ndim>
    TinyArray<value_type, static_size-1>
    pop_front() const
    {
        static_assert(sizeof...(N) == 1,
            "TinyArrayBase::pop_front(): array must be 1-dimensional.");
        return erase(0);
    }

    template<int M = static_ndim>
    TinyArray<value_type, static_size-1>
    pop_back() const
    {
        static_assert(sizeof...(N) == 1,
            "TinyArrayBase::pop_back(): array must be 1-dimensional.");
        return erase(size()-1);
    }

    template<int M = static_ndim>
    TinyArray<value_type, static_size+1>
    insert(ArrayIndex m, value_type v) const
    {
        static_assert(sizeof...(N) == 1,
            "TinyArrayBase::insert(): array must be 1-dimensional.");
        xtensor_precondition(m >= 0 && m <= static_size, "TinyArray::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+"].");
        TinyArray<value_type, static_size+1> res(DontInit);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        res[m] = v;
        for(ArrayIndex k=m; k<static_size; ++k)
            res[k+1] = data_[k];
        return res;
    }

    template <class V, class D, int M>
    inline
    TinyArray<value_type, static_size>
    transpose(TinyArrayBase<V, D, M> const & permutation) const
    {
        static_assert(sizeof...(N) == 1,
            "TinyArray::transpose(): only allowed for 1-dimensional arrays.");
        static_assert(M == static_size || M == runtime_size,
            "TinyArray::transpose(): size mismatch.");
        XTENSOR_ASSERT_RUNTIME_SIZE(M, size() == 0 || size() == permutation.size(),
            "TinyArray::transpose(): size mismatch.");
        TinyArray<value_type, static_size> res(DontInit);
        for(int k=0; k < size(); ++k)
        {
            XTENSOR_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < size(),
                "transpose():  Permutation index out of bounds");
            res[k] = (*this)[permutation[k]];
        }
        return res;
    }

    // boiler plate

    iterator begin() { return data_; }
    iterator end()   { return data_ + static_size; }
    const_iterator begin() const { return data_; }
    const_iterator end()   const { return data_ + static_size; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend()   const { return data_ + static_size; }

    reverse_iterator rbegin() { return reverse_iterator(data_ + static_size); }
    reverse_iterator rend()   { return reverse_iterator(data_); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(data_ + static_size); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(data_); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(data_ + static_size); }
    const_reverse_iterator crend()   const { return const_reverse_iterator(data_); }

    pointer data() { return data_; }
    const_pointer data() const { return data_; }

    reference front() { return data_[0]; }
    reference back()  { return data_[static_size-1]; }
    constexpr const_reference front() const { return data_[0]; }
    constexpr const_reference back()  const { return data_[static_size-1]; }

    constexpr bool       empty() const { return static_size == 0; }
    constexpr ArrayIndex size()  const { return static_size; }
    constexpr ArrayIndex max_size()  const { return static_size; }
    constexpr index_type shape() const { return index_type{ N... }; }
    constexpr ArrayIndex ndim()  const { return static_ndim; }

    TinyArrayBase & reverse()
    {
        ArrayIndex i=0, j=size()-1;
        while(i < j)
             xt::swap(data_[i++], data_[j--]);
        return *this;
    }

    void swap(TinyArrayBase & other)
    {
        for(int k=0; k<static_size; ++k)
        {
            xt::swap(data_[k], other[k]);
        }
    }

    template <class OTHER, class OTHER_DERIVED>
    void swap(TinyArrayBase<OTHER, OTHER_DERIVED, N...> & other)
    {
        for(int k=0; k<static_size; ++k)
        {
            PromoteType<value_type, OTHER> t = data_[k];
            data_[k] = static_cast<value_type>(other[k]);
            other[k] = static_cast<OTHER>(t);
        }
    }

        /// factory function for fixed-size unit matrix
    template <int SIZE>
    static inline
    TinyArray<value_type, SIZE, SIZE>
    eye()
    {
        TinyArray<value_type, SIZE, SIZE> res;
        for(int k=0; k<SIZE; ++k)
            res(k,k) = 1;
        return res;
    }

        /// factory function for the fixed-size k-th unit vector
    template <int SIZE=static_size>
    static inline
    TinyArray<value_type, SIZE>
    unitVector(ArrayIndex k)
    {
        TinyArray<value_type, SIZE> res;
        res(k) = 1;
        return res;
    }

        /// factory function for the k-th unit vector
        // (for compatibility with TinyArray<..., runtime_size>)
    static inline
    TinyArray<value_type, static_size>
    unitVector(tags::SizeProxy const & size, ArrayIndex k)
    {
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "TinyArray::unitVector(): size mismatch.");
        TinyArray<value_type, static_size> res;
        res(k) = 1;
        return res;
    }

        /// factory function for fixed-size linear sequence starting at <tt>start</tt> with stepsize <tt>step</tt>
    static inline
    TinyArray<value_type, N...>
    linearSequence(value_type start = value_type(), value_type step = value_type(1))
    {
        TinyArray<value_type, N...> res(DontInit);
        for(int k=0; k < static_size; ++k, start += step)
            res[k] = start;
        return res;
    }

        /// factory function for fixed-size linear sequence ending at <tt>end-1</tt>
    static inline
    TinyArray<value_type, N...>
    range(value_type end)
    {
        value_type start = end - static_size;
        TinyArray<value_type, N...> res(DontInit);
        for(int k=0; k < static_size; ++k, ++start)
            res[k] = start;
        return res;
    }

  protected:
    data_array_type data_;
};

/********************************************************/
/*                                                      */
/*                TinyArrayBase output                  */
/*                                                      */
/********************************************************/

template <class T, class DERIVED, int ... N>
std::ostream & operator<<(std::ostream & o, TinyArrayBase<T, DERIVED, N...> const & v)
{
    o << "{";
    if(v.size() > 0)
        o << PromoteType<T>(v[0]);
    for(int i=1; i < v.size(); ++i)
        o << ", " << PromoteType<T>(v[i]);
    o << "}";
    return o;
}

template <class T, class DERIVED, int N1, int N2>
std::ostream & operator<<(std::ostream & o, TinyArrayBase<T, DERIVED, N1, N2> const & v)
{
    o << "{";
    for(int i=0; N2>0 && i<N1; ++i)
    {
        if(i > 0)
            o << ",\n ";
        o << PromoteType<T>(v(i,0));
        for(int j=1; j<N2; ++j)
        {
            o << ", " << PromoteType<T>(v(i, j));
        }
    }
    o << "}";
    return o;
}

/********************************************************/
/*                                                      */
/*         TinyArrayBase<..., runtime_size>             */
/*                                                      */
/********************************************************/

/** \brief Specialization of TinyArrayBase for dynamic arrays.

    This class contains functionality shared by
    \ref TinyArray and \ref TinyArrayView, and enables these classes
    to be freely mixed within expressions. It is typically not used directly.

    <b>\#include</b> \<vigra/tinyarray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, class DERIVED>
class TinyArrayBase<VALUETYPE, DERIVED, runtime_size>
: public TinyArrayTag
{
  public:

    template <class NEW_VALUETYPE>
    using AsType = TinyArray<NEW_VALUETYPE, runtime_size>;

    using value_type             = VALUETYPE;
    using const_value_type       = typename std::add_const<VALUETYPE>::type;
    using reference              = value_type &;
    using const_reference        = const_value_type &;
    using pointer                = value_type *;
    using const_pointer          = const_value_type *;
    using iterator               = value_type *;
    using const_iterator         = const_value_type *;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using index_type             = ArrayIndex;

    static const ArrayIndex static_size  = runtime_size;
    static const ArrayIndex static_ndim  = 1;
    static const bool may_use_uninitialized_memory =
                                   UninitializedMemoryTraits<VALUETYPE>::value;

  protected:

    template <int LEVEL, class ... V2>
    void initImpl(VALUETYPE v1, V2... v2)
    {
        data_[LEVEL] = v1;
        initImpl<LEVEL+1>(v2...);
    }

    template <int LEVEL>
    void initImpl(VALUETYPE v1)
    {
        data_[LEVEL] = v1;
    }

  public:

    TinyArrayBase(ArrayIndex size=0, pointer data=0)
    : size_(size)
    , data_(data)
    {}

    TinyArrayBase(TinyArrayBase const &) = default;

    // assignment

    TinyArrayBase & operator=(value_type v)
    {
        for(int i=0; i<size_; ++i)
            data_[i] = v;
        return *this;
    }

    TinyArrayBase & operator=(TinyArrayBase const & rhs)
    {
        xtensor_precondition(size_ == rhs.size(),
            "TinyArrayBase::operator=(): size mismatch.");
        for(int i=0; i<size_; ++i)
            data_[i] = rhs[i];
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED, int N>
    TinyArrayBase & operator=(TinyArrayBase<OTHER, OTHER_DERIVED, N> const & other)
    {
        xtensor_precondition(size_ == other.size(),
            "TinyArrayBase::operator=(): size mismatch.");
        for(int i=0; i<size_; ++i)
            data_[i] = static_cast<value_type>(other[i]);
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED, int ... M>
    bool
    sameShape(TinyArrayBase<OTHER, OTHER_DERIVED, M...> const & other) const
    {
        return sizeof...(M) == 1 && size() == other.size();;
    }

    template <class OTHER, class OTHER_DERIVED>
    bool
    sameShape(TinyArrayBase<OTHER, OTHER_DERIVED, runtime_size> const & other) const
    {
        return size() == other.size();
    }

    DERIVED & init(value_type v = value_type())
    {
        for(int i=0; i<size_; ++i)
            data_[i] = v;
        return static_cast<DERIVED &>(*this);
    }

    template <class ... V>
    DERIVED & init(value_type v0, value_type v1, V... v)
    {
        xtensor_precondition(sizeof...(V)+2 == size_,
                      "TinyArrayBase::init(): wrong number of arguments.");
        initImpl<0>(v0, v1, v...);
        return static_cast<DERIVED &>(*this);
    }

    template <class Iterator>
    DERIVED & init(Iterator first, Iterator end)
    {
        ArrayIndex range = std::distance(first, end);
        if(size_ < range)
            range = size_;
        for(ArrayIndex i=0; i<range; ++i, ++first)
            data_[i] = static_cast<value_type>(*first);
        return static_cast<DERIVED &>(*this);
    }

    template <class V>
    DERIVED & init(std::initializer_list<V> l)
    {
        return init(l.begin(), l.end());
    }

    // index access

    reference operator[](ArrayIndex i)
    {
        return data_[i];
    }

    constexpr const_reference operator[](ArrayIndex i) const
    {
        return data_[i];
    }

    reference at(ArrayIndex i)
    {
        if(i < 0 || i >= size_)
            throw std::out_of_range("TinyArrayBase::at()");
        return data_[i];
    }

    const_reference at(ArrayIndex i) const
    {
        if(i < 0 || i >= size_)
            throw std::out_of_range("TinyArrayBase::at()");
        return data_[i];
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= size_</tt>.
        */
    template <int FROM, int TO>
    TinyArrayView<value_type, TO-FROM>
    subarray() const
    {
        static_assert(FROM >= 0 && FROM < TO,
            "TinyArrayBase::subarray(): range out of bounds.");
        xtensor_precondition(TO <= size_,
            "TinyArrayBase::subarray(): range out of bounds.");
        return TinyArrayView<value_type, TO-FROM>(data_+FROM);
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= size_</tt>.
        */
    TinyArrayView<value_type, runtime_size>
    subarray(ArrayIndex FROM, ArrayIndex TO) const
    {
        xtensor_precondition(FROM >= 0 && FROM < TO && TO <= size_,
            "TinyArrayBase::subarray(): range out of bounds.");
        return TinyArrayView<value_type, runtime_size>(TO-FROM, const_cast<pointer>(data_)+FROM);
    }


    TinyArray<value_type, runtime_size>
    erase(ArrayIndex m) const
    {
        xtensor_precondition(m >= 0 && m < size(), "TinyArray::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+").");
        TinyArray<value_type, runtime_size> res(size()-1, DontInit);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        for(ArrayIndex k=m+1; k<size(); ++k)
            res[k-1] = data_[k];
        return res;
    }

    TinyArray<value_type, runtime_size>
    pop_front() const
    {
        return erase(0);
    }

    TinyArray<value_type, runtime_size>
    pop_back() const
    {
        return erase(size()-1);
    }

    TinyArray<value_type, runtime_size>
    insert(ArrayIndex m, value_type v) const
    {
        xtensor_precondition(m >= 0 && m <= size(), "TinyArray::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+"].");
        TinyArray<value_type, runtime_size> res(size()+1, DontInit);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        res[m] = v;
        for(ArrayIndex k=m; k<size(); ++k)
            res[k+1] = data_[k];
        return res;
    }

    template <class V, class D, int M>
    inline
    TinyArray<value_type, runtime_size>
    transpose(TinyArrayBase<V, D, M> const & permutation) const
    {
        xtensor_precondition(size() == 0 || size() == permutation.size(),
            "TinyArray::transpose(): size mismatch.");
        TinyArray<value_type, runtime_size> res(size(), DontInit);
        for(ArrayIndex k=0; k < size(); ++k)
        {
            XTENSOR_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < size(),
                "transpose():  Permutation index out of bounds");
            res[k] = (*this)[permutation[k]];
        }
        return res;
    }

    // boiler plate

    iterator begin() { return data_; }
    iterator end()   { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end()   const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend()   const { return data_ + size_; }

    reverse_iterator rbegin() { return reverse_iterator(data_ + size_); }
    reverse_iterator rend()   { return reverse_iterator(data_); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(data_ + size_); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(data_); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(data_ + size_); }
    const_reverse_iterator crend()   const { return const_reverse_iterator(data_); }

    pointer data() { return data_; }
    const_pointer data() const { return data_; }

    reference front() { return data_[0]; }
    reference back()  { return data_[size_-1]; }
    const_reference front() const { return data_[0]; }
    const_reference back()  const { return data_[size_-1]; }

    bool       empty() const { return size_ == 0; }
    ArrayIndex size()  const { return size_; }
    ArrayIndex max_size()  const { return size_; }
    ArrayIndex ndim()  const { return static_ndim; }

    TinyArrayBase & reverse()
    {
        ArrayIndex i=0, j=size_-1;
        while(i < j)
             xt::swap(data_[i++], data_[j--]);
        return *this;
    }

    void swap(TinyArrayBase & other)
    {
        xt::swap(size_, other.size_);
        xt::swap(data_, other.data_);
    }

        /// factory function for the fixed-size k-th unit vector
    static inline
    TinyArray<value_type, runtime_size>
    unitVector(tags::SizeProxy const & size, ArrayIndex k)
    {
        TinyArray<value_type, runtime_size> res(size.value);
        res[k] = 1;
        return res;
    }

        /// factory function for a linear sequence from <tt>begin</tt> to <tt>end</tt>
        /// (exclusive) with stepsize <tt>step</tt>
    static inline
    TinyArray<value_type, runtime_size>
    range(value_type begin,
          value_type end,
          value_type step = value_type(1))
    {
        xtensor_precondition(step != 0,
            "TinyArray::range(): step must be non-zero.");
        xtensor_precondition((step > 0 && begin <= end) || (step < 0 && begin >= end),
            "TinyArray::range(): sign mismatch between step and (end-begin).");
        ArrayIndex size = mathfunctions::floor((abs(end-begin+step)-1)/abs(step));
        TinyArray<value_type, runtime_size> res(size, DontInit);
        for(ArrayIndex k=0; k < size; ++k, begin += step)
            res[k] = begin;
        return res;
    }

        /// factory function for a linear sequence from 0 to <tt>end</tt>
        /// (exclusive) with stepsize 1
    static inline
    TinyArray<value_type, runtime_size>
    range(value_type end)
    {
        xtensor_precondition(end >= 0,
            "TinyArray::range(): end must be non-negative.");
        TinyArray<value_type, runtime_size> res(end, DontInit);
        auto begin = value_type();
        for(ArrayIndex k=0; k < res.size(); ++k, ++begin)
            res[k] = begin;
        return res;
    }

  protected:
    ArrayIndex size_;
    pointer data_;
};

/********************************************************/
/*                                                      */
/*                       TinyArray                      */
/*                                                      */
/********************************************************/

/** \brief Class for fixed size arrays.
    \ingroup RangesAndPoints

    This class contains an array of the specified VALUETYPE with
    (possibly multi-dimensional) shape given by the sequence <tt>ArrayIndex ... N</tt>.
    The interface conforms to STL vector, except that there are no functions
    that change the size of a TinyArray.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinyArrays are defined as component-wise applications of these
    operations.

    See also:<br>
    <UL style="list-style-image:url(documents/bullet.gif)">
        <LI> \ref vigra::TinyArrayBase
        <LI> \ref vigra::TinyArrayView
        <LI> \ref TinyArrayOperators
    </UL>

    <b>\#include</b> \<vigra/TinyArray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, int M, int ... N>
class TinyArray
: public TinyArrayBase<VALUETYPE, TinyArray<VALUETYPE, M, N...>, M, N...>
{
  public:
    using base_type = TinyArrayBase<VALUETYPE, TinyArray<VALUETYPE, M, N...>, M, N...>;

    typedef typename base_type::value_type value_type;
    static const ArrayIndex static_ndim = base_type::static_ndim;
    static const ArrayIndex static_size = base_type::static_size;

    explicit constexpr
    TinyArray()
    : base_type(value_type())
    {}

    explicit
    TinyArray(SkipInitialization)
    : base_type(DontInit)
    {}

        // This constructor would allow construction with round brackets, e.g.:
        //     TinyArray<int, 1> a(2);
        // However, this may lead to bugs when fixed-size arrays are mixed with
        // runtime_size arrays, where
        //     TinyArray<int, runtime_size> a(2);
        // constructs an array of length 2 with initial value 0. To avoid such bugs,
        // construction is restricted to curly braces:
        //     TinyArray<int, 1> a{2};
        //
    // template <class ... V>
    // constexpr TinyArray(value_type v0, V... v)
    // : base_type(v0, v...)
    // {}

    template <class V>
    TinyArray(std::initializer_list<V> v)
    : base_type(DontInit)
    {
        if(v.size() == 1)
            base_type::init(static_cast<value_type>(*v.begin()));
        else if(v.size() == static_size)
            base_type::initImpl(v.begin());
        else
            xtensor_precondition(false,
                "TinyArray(std::initializer_list<V>): wrong initialization size.");
    }

        // for compatibility with TinyArray<VALUETYPE, runtime_size>
    explicit
    TinyArray(tags::SizeProxy const & size,
              value_type const & v = value_type())
    : base_type(v)
    {
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "TinyArray(size): size argument conflicts with array length.");
    }

        // for compatibility with TinyArray<VALUETYPE, runtime_size>
    TinyArray(tags::SizeProxy const & size, SkipInitialization)
    : base_type(DontInit)
    {
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "TinyArray(size): size argument conflicts with array length.");
    }

        // for compatibility with TinyArray<VALUETYPE, runtime_size>
    TinyArray(ArrayIndex size, SkipInitialization)
    : base_type(DontInit)
    {
        XTENSOR_ASSERT_MSG(size == static_size,
            "TinyArray(size): size argument conflicts with array length.");
    }

    constexpr TinyArray(TinyArray const &) = default;

    template <class OTHER, class DERIVED>
    TinyArray(TinyArrayBase<OTHER, DERIVED, M, N...> const & other)
    : base_type(other)
    {}

    template <class OTHER, class DERIVED>
    TinyArray(TinyArrayBase<OTHER, DERIVED, runtime_size> const & other)
    : base_type(DontInit)
    {
        if(other.size() == 0)
        {
            this->init(value_type());
        }
        else if(this->size() != 0)
        {
            xtensor_precondition(this->size() == other.size(),
                "TinyArray(): shape mismatch.");
            this->initImpl(other.begin());
        }
    }

    explicit TinyArray(value_type const (&u)[1])
    : base_type(*u)
    {}

    template <class U>
    explicit TinyArray(U const (&u)[1])
    : base_type(static_cast<value_type>(*u))
    {}

    template <class U, int S=static_size,
              XTENSOR_REQUIRE<S != 1>>
    explicit TinyArray(U const (&u)[static_size])
    : base_type(u)
    {}

    template <class U,
              XTENSOR_REQUIRE<IteratorConcept<U>::value> >
    explicit TinyArray(U u, U end = U())
    : base_type(u)
    {}

    template <class U,
              XTENSOR_REQUIRE<IteratorConcept<U>::value> >
    TinyArray(U u, ReverseCopyTag)
    : base_type(u, ReverseCopy)
    {}

        // for compatibility with TinyArray<..., runtime_size>
    template <class U,
              XTENSOR_REQUIRE<IteratorConcept<U>::value> >
    TinyArray(U u, U end, ReverseCopyTag)
    : base_type(u, ReverseCopy)
    {}

    TinyArray & operator=(TinyArray const &) = default;

    TinyArray & operator=(value_type v)
    {
        base_type::operator=(v);
        return *this;
    }

    TinyArray & operator=(value_type const (&v)[static_size])
    {
        base_type::operator=(v);
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED>
    TinyArray & operator=(TinyArrayBase<OTHER, OTHER_DERIVED, M, N...> const & other)
    {
        base_type::operator=(other);
        return *this;
    }

    constexpr bool       empty() const { return static_size == 0; }
    constexpr ArrayIndex size()  const { return static_size; }
    constexpr ArrayIndex max_size()  const { return static_size; }
    constexpr typename base_type::index_type shape() const { return typename base_type::index_type{ M, N... }; }
    constexpr ArrayIndex ndim()  const { return static_ndim; }
};

template<class T, int M, int ... N>
struct UninitializedMemoryTraits<TinyArray<T, M, N...>>
{
    static const bool value = UninitializedMemoryTraits<T>::value;
};

/********************************************************/
/*                                                      */
/*            TinyArray<..., runtime_size>              */
/*                                                      */
/********************************************************/

/** \brief Specialization of TinyArray for dynamic arrays.
    \ingroup RangesAndPoints

    This class contains an array of the specified VALUETYPE with
    size specified at runtim.
    The interface conforms to STL vector, except that there are no functions
    that change the size of a TinyArray.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinyArrays are defined as component-wise applications of these
    operations.

    See also:<br>
    <UL style="list-style-image:url(documents/bullet.gif)">
        <LI> \ref vigra::TinyArrayBase
        <LI> \ref vigra::TinyArrayView
        <LI> \ref TinyArrayOperators
    </UL>

    <b>\#include</b> \<vigra/TinyArray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE>
class TinyArray<VALUETYPE, runtime_size>
: public TinyArrayBase<VALUETYPE, TinyArray<VALUETYPE, runtime_size>, runtime_size>
{
  public:
    using base_type = TinyArrayBase<VALUETYPE, TinyArray<VALUETYPE, runtime_size>, runtime_size>;

    using value_type = VALUETYPE;

    TinyArray()
    : base_type()
    {}

    explicit
    TinyArray(ArrayIndex size,
              value_type const & initial = value_type())
    : base_type(size)
    {
        this->data_ = alloc_.allocate(this->size_);
        std::uninitialized_fill(this->begin(), this->end(), initial);
    }

    explicit
    TinyArray(tags::SizeProxy const & size,
              value_type const & initial = value_type())
    : TinyArray(size.value, initial)
    {}

    TinyArray(ArrayIndex size, SkipInitialization)
    : base_type(size)
    {
        this->data_ = alloc_.allocate(this->size_);
        if(!base_type::may_use_uninitialized_memory)
            std::uninitialized_fill(this->begin(), this->end(), value_type());
    }

    TinyArray(TinyArray const & rhs )
    : base_type(rhs.size())
    {
        this->data_ = alloc_.allocate(this->size_);
        std::uninitialized_copy(rhs.begin(), rhs.end(), this->begin());
    }

    TinyArray(TinyArray && rhs)
    : base_type()
    {
        this->swap(rhs);
    }

    template <class U, class D, int ... N>
    TinyArray(TinyArrayBase<U, D, N...> const & other)
    : TinyArray(other.begin(), other.end())
    {}

    template <class U,
              XTENSOR_REQUIRE<IteratorConcept<U>::value> >
    TinyArray(U begin, U end)
    : base_type(std::distance(begin, end))
    {
        this->data_ = alloc_.allocate(this->size_);
        for(int i=0; i<this->size_; ++i, ++begin)
            new(this->data_+i) value_type(static_cast<value_type>(*begin));
    }

    template <class U,
              XTENSOR_REQUIRE<IteratorConcept<U>::value> >
    TinyArray(U begin, U end, ReverseCopyTag)
    : base_type(std::distance(begin, end))
    {
        this->data_ = alloc_.allocate(this->size_);
        for(int i=0; i<this->size_; ++i, --end)
            new(this->data_+i) value_type(static_cast<value_type>(*(end-1)));
    }

    template <class U, size_t SIZE>
    TinyArray(const U (&u)[SIZE])
    : TinyArray(u, u+SIZE)
    {}

    template <class U>
    TinyArray(std::initializer_list<U> rhs)
    : TinyArray(rhs.begin(), rhs.end())
    {}

    TinyArray & operator=(value_type const & v)
    {
        base_type::operator=(v);
        return *this;
    }

    TinyArray & operator=(TinyArray && rhs)
    {
        if(this->size_ != rhs.size())
            rhs.swap(*this);
        else
            base_type::operator=(rhs);
        return *this;
    }

    TinyArray & operator=(TinyArray const & rhs)
    {
        if(this == &rhs)
            return *this;
        if(this->size_ != rhs.size())
            TinyArray(rhs).swap(*this);
        else
            base_type::operator=(rhs);
        return *this;
    }

    template <class U, class D, int ... N>
    TinyArray & operator=(TinyArrayBase<U, D, N...> const & rhs)
    {
        if(this->size_ == 0 || rhs.size() == 0)
            TinyArray(rhs).swap(*this);
        else
            base_type::operator=(rhs);
        return *this;
    }

    ~TinyArray()
    {
        if(!base_type::may_use_uninitialized_memory)
        {
            for(ArrayIndex i=0; i<this->size_; ++i)
                (this->data_+i)->~value_type();
        }
        alloc_.deallocate(this->data_, this->size_);
    }

#if 0
    // FIXME: hacks to use TinyArray as shape in xtensor
    using base_type::erase;

    void erase(base_type::pointer p)
    {
        --this->size_;
        for(; p < this->data_ + this->size_; ++p)
            *p = p[1];
    }

    void resize(size_t s)
    {
        if(s > this->size_)
        {
            alloc_.deallocate(this->data_, this->size_);
            this->data_ = alloc_.allocate(s);
        }
        this->size_ = s;
    }

    template <class U>
    TinyArray(std::vector<U> const & rhs)
    : TinyArray(rhs.cbegin(), rhs.cend())
    {}

    template <class U>
    TinyArray & operator=(std::vector<U> const & rhs)
    {
        resize(rhs.size());
        for(int k=0; k < this->size_; ++k)
            this->data_[k] = rhs[k];
        return *this;
    }

    template <class U>
    bool operator==(std::vector<U> const & rhs) const
    {
        if(this->size_ != (int)rhs.size())
            return false;
        for(int k=0; k < this->size_; ++k)
            if(this->data_[k] != rhs[k])
                return false;
        return true;
    }
#endif

  private:
    // FIXME: implement an optimized allocator
    // FIXME: (look at Alexandrescu's Loki library or Kolmogorov's code)
    std::allocator<value_type> alloc_;
};


template <class U>
bool operator==(std::vector<U> const & l, TinyArray<U, -1> const & r)
{
    if((int)l.size() != r.size())
        return false;
    for(int k=0; k < (int)l.size(); ++k)
        if(l[k] != r[k])
            return false;
    return true;
}


template<class T>
struct UninitializedMemoryTraits<TinyArray<T, runtime_size>>
{
    static const bool value = false;
};



/********************************************************/
/*                                                      */
/*                    TinyArrayView                     */
/*                                                      */
/********************************************************/

/** \brief Wrapper for fixed size arrays.

    This class wraps the memory of an array of the specified VALUETYPE
    with (possibly multi-dimensional) shape given by <tt>ArrayIndex....N</tt>.
    Thus, the array can be accessed with an interface similar to
    that of std::vector (except that there are no functions
    that change the size of a TinyArrayView). The TinyArrayView
    does <em>not</em> assume ownership of the given memory.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinyArrayViews are defined as component-wise applications of these
    operations.

    <b>See also:</b>
    <ul>
        <li> \ref vigra::TinyArrayBase
        <li> \ref vigra::TinyArray
        <li> \ref vigra::TinySymmetricView
        <li> \ref TinyArrayOperators
    </ul>

    <b>\#include</b> \<vigra/tinyarray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, int M, int ... N>
class TinyArrayView
: public TinyArrayBase<VALUETYPE, TinyArrayView<VALUETYPE, M, N...>, M, N...>
{
  public:
    using base_type = TinyArrayBase<VALUETYPE, TinyArrayView<VALUETYPE, M, N...>, M, N...>;

    typedef typename base_type::value_type value_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    static const ArrayIndex static_size  = base_type::static_size;
    static const ArrayIndex static_ndim  = base_type::static_ndim;

    TinyArrayView()
    : base_type(DontInit)
    {
        base_type::data_ = nullptr;
    }

        /** Construct view for given data array
        */
    TinyArrayView(const_pointer data)
    : base_type(DontInit)
    {
        base_type::data_ = const_cast<pointer>(data);
    }

        /** Copy constructor (shallow copy).
        */
    TinyArrayView(TinyArrayView const & other)
    : base_type(DontInit)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Construct view from other TinyArray.
        */
    template <class OTHER_DERIVED>
    TinyArrayView(TinyArrayBase<value_type, OTHER_DERIVED, M, N...> const & other)
    : base_type(DontInit)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Reset to the other array's pointer.
        */
    template <class OTHER_DERIVED>
    void reset(TinyArrayBase<value_type, OTHER_DERIVED, M, N...> const & other)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Copy the data (not the pointer) of the rhs.
        */
    TinyArrayView & operator=(TinyArrayView const & r)
    {
        for(int k=0; k<static_size; ++k)
            base_type::data_[k] = static_cast<value_type>(r[k]);
        return *this;
    }

        /** Copy the data of the rhs with cast.
        */
    template <class U, class OTHER_DERIVED>
    TinyArrayView & operator=(TinyArrayBase<U, OTHER_DERIVED, M, N...> const & r)
    {
        for(int k=0; k<static_size; ++k)
            base_type::data_[k] = static_cast<value_type>(r[k]);
        return *this;
    }

    constexpr bool       empty() const { return static_size == 0; }
    constexpr ArrayIndex size()  const { return static_size; }
    constexpr ArrayIndex max_size()  const { return static_size; }
    constexpr typename base_type::index_type shape() const { return typename base_type::index_type{ M, N... }; }
    constexpr ArrayIndex ndim()  const { return static_ndim; }
};

template <class VALUETYPE>
class TinyArrayView<VALUETYPE, runtime_size>
: public TinyArrayBase<VALUETYPE, TinyArrayView<VALUETYPE, runtime_size>, runtime_size>
{
  public:
    using base_type = TinyArrayBase<VALUETYPE, TinyArrayView<VALUETYPE, runtime_size>, runtime_size>;

    using base_type::base_type;
    using base_type::operator=;
};

/********************************************************/
/*                                                      */
/*               TinyArraySymmetricView                 */
/*                                                      */
/********************************************************/

/** \brief Wrapper for fixed size arrays.

    This class wraps the memory of an 1D array of the specified VALUETYPE
    with size <tt>N*(N+1)/2</tt> and interprets this array as a symmetric
    matrix. Specifically, the data are interpreted as the row-wise
    representation of the upper triangular part of the symmetric matrix.
    All index access operations are overloaded such that the view appears
    as if it were a full matrix. The TinySymmetricView
    does <em>not</em> assume ownership of the given memory.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinySymmetricView are defined as component-wise applications of these
    operations.

    <b>See also:</b>
    <ul>
        <li> \ref vigra::TinyArrayBase
        <li> \ref vigra::TinyArray
        <li> \ref vigra::TinyArrayView
        <li> \ref TinyArrayOperators
    </ul>

    <b>\#include</b> \<vigra/tinyarray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, int N>
class TinySymmetricView
: public TinyArrayBase<VALUETYPE, TinySymmetricView<VALUETYPE, N>, N*(N+1)/2>
{
  public:
    using base_type = TinyArrayBase<VALUETYPE, TinySymmetricView<VALUETYPE, N>, N*(N+1)/2>;

    typedef typename base_type::value_type value_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;
    using index_type = TinyArray<ArrayIndex, 2>;

    static const ArrayIndex static_size = base_type::static_size;
    static const ArrayIndex static_ndim = 2;

    TinySymmetricView()
    : base_type(DontInit)
    {
        base_type::data_ = nullptr;
    }

        /** Construct view for given data array
        */
    TinySymmetricView(const_pointer data)
    : base_type(DontInit)
    {
        base_type::data_ = const_cast<pointer>(data);
    }

        /** Copy constructor (shallow copy).
        */
    TinySymmetricView(TinySymmetricView const & other)
    : base_type(DontInit)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Construct view from other TinyArray.
        */
    template <class OTHER_DERIVED>
    TinySymmetricView(TinyArrayBase<value_type, OTHER_DERIVED, N*(N+1)/2> const & other)
    : base_type(DontInit)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Reset to the other array's pointer.
        */
    template <class OTHER_DERIVED>
    void reset(TinyArrayBase<value_type, OTHER_DERIVED, N*(N+1)/2> const & other)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Copy the data (not the pointer) of the rhs.
        */
    TinySymmetricView & operator=(TinySymmetricView const & r)
    {
        for(int k=0; k<static_size; ++k)
            base_type::data_[k] = static_cast<value_type>(r[k]);
        return *this;
    }

        /** Copy the data of the rhs with cast.
        */
    template <class U, class OTHER_DERIVED>
    TinySymmetricView & operator=(TinyArrayBase<U, OTHER_DERIVED, N*(N+1)/2> const & r)
    {
        for(int k=0; k<static_size; ++k)
            base_type::data_[k] = static_cast<value_type>(r[k]);
        return *this;
    }

    // index access

    reference operator[](ArrayIndex i)
    {
        return base_type::operator[](i);
    }

    constexpr const_reference operator[](ArrayIndex i) const
    {
        return base_type::operator[](i);
    }

    reference at(ArrayIndex i)
    {
        return base_type::at(i);
    }

    const_reference at(ArrayIndex i) const
    {
        return base_type::at(i);
    }

    reference operator[](ArrayIndex const (&i)[2])
    {
        return this->operator()(i[0], i[1]);
    }

    constexpr const_reference operator[](ArrayIndex const (&i)[2]) const
    {
        return this->operator()(i[0], i[1]);
    }

    reference at(ArrayIndex const (&i)[static_ndim])
    {
        return this->at(i[0], i[1]);
    }

    const_reference at(ArrayIndex const (&i)[static_ndim]) const
    {
        return this->at(i[0], i[1]);
    }

    reference operator[](index_type const & i)
    {
        return this->operator()(i[0], i[1]);
    }

    constexpr const_reference operator[](index_type const & i) const
    {
        return this->operator()(i[0], i[1]);
    }

    reference at(index_type const & i)
    {
        return this->at(i[0], i[1]);
    }

    const_reference at(index_type const & i) const
    {
        return this->at(i[0], i[1]);
    }

    reference operator()(ArrayIndex i, ArrayIndex j)
    {
        return (i > j)
            ? base_type::data_[i + N*j - (j*(j+1) >> 1)]
            : base_type::data_[N*i + j - (i*(i+1) >> 1)];
    }

    constexpr const_reference operator()(ArrayIndex const i, ArrayIndex const j) const
    {
        return (i > j)
            ? base_type::data_[i + (j*((2 * N - 1) - j) >> 1)]
            : base_type::data_[j + (i*((2 * N - 1) - i) >> 1)];
    }

    reference at(ArrayIndex i, ArrayIndex j)
    {
        ArrayIndex k = (i > j)
                           ? i + (j*((2*N-1) - j) >> 1)
                           : j + (i*((2*N-1) - i) >> 1);
        if(k < 0 || k >= static_size)
            throw std::out_of_range("TinySymmetricView::at()");
        return base_type::data_[k];
    }

    const_reference at(ArrayIndex i, ArrayIndex j) const
    {
        ArrayIndex k = (i > j)
                           ? i + N*j - (j*(j+1) >> 1)
                           : N*i + j - (i*(i+1) >> 1);
        if(k < 0 || k >= static_size)
            throw std::out_of_range("TinySymmetricView::at()");
        return base_type::data_[k];
    }

    constexpr index_type shape() const { return index_type{N, N}; }
    constexpr ArrayIndex ndim () const { return static_ndim; }
};

/********************************************************/
/*                                                      */
/*            TinyArraySymmetricView output             */
/*                                                      */
/********************************************************/

template <class T, int N>
std::ostream & operator<<(std::ostream & o, TinySymmetricView<T, N> const & v)
{
    o << "{";
    for(int i=0; i<N; ++i)
    {
        if(i > 0)
            o << ",\n ";
        o << PromoteType<T>(v(i,0));
        for(int j=1; j<N; ++j)
        {
            o << ", " << PromoteType<T>(v(i, j));
        }
    }
    o << "}";
    return o;
}


/********************************************************/
/*                                                      */
/*                TinyArray Comparison                  */
/*                                                      */
/********************************************************/

/** \addtogroup TinyArrayOperators Functions for TinyArray

    \brief Implement basic arithmetic and equality for TinyArray.

    These functions fulfill the requirements of a Linear Space (vector space).
    Return types are determined according to \ref PromoteType or \ref RealPromoteType.

    <b>\#include</b> \<vigra/TinyArray.hxx\><br>
    Namespace: vigra
*/
//@{

    /// element-wise equal
template <class V1, class D1, class V2, class D2, int ...M, int ... N>
inline bool
operator==(TinyArrayBase<V1, D1, M...> const & l,
           TinyArrayBase<V2, D2, N...> const & r)
{
    if(!l.sameShape(r))
        return false;
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r[k])
            return false;
    return true;
}

    /// element-wise equal to a constant
template <class V1, class D1, class V2, int ...M,
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
operator==(TinyArrayBase<V1, D1, M...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r)
            return false;
    return true;
}

    /// element-wise equal to a constant
template <class V1, class V2, class D2, int ...M,
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
operator==(V1 const & l,
           TinyArrayBase<V2, D2, M...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if(l != r[k])
            return false;
    return true;
}

    /// element-wise not equal
template <class V1, class D1, class V2, class D2, int ... M, int ... N>
inline bool
operator!=(TinyArrayBase<V1, D1, M...> const & l,
           TinyArrayBase<V2, D2, N...> const & r)
{
    if(!l.sameShape(r))
        return true;
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r[k])
            return true;
    return false;
}

    /// element-wise not equal to a constant
template <class V1, class D1, class V2, int ... M,
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
operator!=(TinyArrayBase<V1, D1, M...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r)
            return true;
    return false;
}

    /// element-wise not equal to a constant
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
operator!=(V1 const & l,
           TinyArrayBase<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if(l != r[k])
            return true;
    return false;
}

    /// lexicographical comparison
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
operator<(TinyArrayBase<V1, D1, N...> const & l,
          TinyArrayBase<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "TinyArrayBase::operator<(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
    {
        if(l[k] < r[k])
            return true;
        if(r[k] < l[k])
            return false;
    }
    return false;
}

    /// lexicographical comparison
template <class V1, class D, int N, class V2, class A>
inline bool
operator<(TinyArrayBase<V1, D, N> const & l,
          std::vector<V2, A> const & r)
{
    ArrayIndex size = r.size();
    bool equal_res  = false;
    if(l.size() < size)
    {
        size      = l.size();
        equal_res = true;
    }
    for(int k=0; k < size; ++k)
    {
        if(l[k] < r[k])
            return true;
        if(r[k] < l[k])
            return false;
    }
    return equal_res;
}

template <class V1, class A, class V2, class D, int N>
inline bool
operator<(std::vector<V1, A> const & l,
          TinyArrayBase<V2, D, N> const & r)
{
    ArrayIndex size = r.size();
    bool equal_res  = false;
    if(l.size() < size)
    {
        size      = l.size();
        equal_res = true;
    }
    for(int k=0; k < size; ++k)
    {
        if(l[k] < r[k])
            return true;
        if(r[k] < l[k])
            return false;
    }
    return equal_res;
}

template <class V1, class D, int N, class V2, class A>
inline bool
operator>(TinyArrayBase<V1, D, N> const & l,
          std::vector<V2, A> const & r)
{
    return r < l;
}

template <class V1, class A, class V2, class D, int N>
inline bool
operator>(std::vector<V1, A> const & l,
          TinyArrayBase<V2, D, N> const & r)
{
    return r < l;
}

    /// check if all elements are non-zero (or 'true' if V is bool)
template <class V, class D, int ... N>
inline bool
all(TinyArrayBase<V, D, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] == V())
            return false;
    return true;
}

    /// check if at least one element is non-zero (or 'true' if V is bool)
template <class V, class D, int ... N>
inline bool
any(TinyArrayBase<V, D, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] != V())
            return true;
    return false;
}

    /// check if all elements are zero (or 'false' if V is bool)
template <class V, class D, int ... N>
inline bool
allZero(TinyArrayBase<V, D, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] != V())
            return false;
    return true;
}

    /// pointwise less-than
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
allLess(TinyArrayBase<V1, D1, N...> const & l,
        TinyArrayBase<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "TinyArrayBase::allLess(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] >= r[k])
            return false;
    return true;
}

    /// pointwise less than a constant
    /// (typically used to check negativity with `r = 0`)
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
allLess(TinyArrayBase<V1, D1, N...> const & l,
        V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] >= r)
            return false;
    return true;
}

    /// constant pointwise less than the vector
    /// (typically used to check positivity with `l = 0`)
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
allLess(V1 const & l,
        TinyArrayBase<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l >= r[k])
            return false;
    return true;
}

    /// pointwise less-equal
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
allLessEqual(TinyArrayBase<V1, D1, N...> const & l,
             TinyArrayBase<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "TinyArrayBase::allLessEqual(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] > r[k])
            return false;
    return true;
}

    /// pointwise less-equal with a constant
    /// (typically used to check non-positivity with `r = 0`)
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
allLessEqual(TinyArrayBase<V1, D1, N...> const & l,
             V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] > r)
            return false;
    return true;
}

    /// pointwise less-equal with a constant
    /// (typically used to check non-negativity with `l = 0`)
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
allLessEqual(V1 const & l,
             TinyArrayBase<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l > r[k])
            return false;
    return true;
}

    /// pointwise greater-than
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
allGreater(TinyArrayBase<V1, D1, N...> const & l,
           TinyArrayBase<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "TinyArrayBase::allGreater(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] <= r[k])
            return false;
    return true;
}

    /// pointwise greater-than with a constant
    /// (typically used to check positivity with `r = 0`)
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
allGreater(TinyArrayBase<V1, D1, N...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] <= r)
            return false;
    return true;
}

    /// constant pointwise greater-than a vector
    /// (typically used to check negativity with `l = 0`)
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
allGreater(V1 const & l,
           TinyArrayBase<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l <= r[k])
            return false;
    return true;
}

    /// pointwise greater-equal
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
allGreaterEqual(TinyArrayBase<V1, D1, N...> const & l,
                TinyArrayBase<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "TinyArrayBase::allGreaterEqual(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] < r[k])
            return false;
    return true;
}

    /// pointwise greater-equal with a constant
    /// (typically used to check non-negativity with `r = 0`)
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
allGreaterEqual(TinyArrayBase<V1, D1, N...> const & l,
                V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] < r)
            return false;
    return true;
}

    /// pointwise greater-equal with a constant
    /// (typically used to check non-positivity with `l = 0`)
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value &&
                        std::is_convertible<V2, V1>::value> >
inline bool
allGreaterEqual(V1 const & l,
                TinyArrayBase<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l < r[k])
            return false;
    return true;
}

template <class V1, class D1, class V2, class D2, int ... N>
inline bool
isclose(TinyArrayBase<V1, D1, N...> const & l,
        TinyArrayBase<V2, D2, N...> const & r,
        PromoteType<V1, V2> epsilon = 2.0*std::numeric_limits<PromoteType<V1, V2> >::epsilon())
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "TinyArrayBase::isclose(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if(!mathfunctions::isclose(l[k], r[k], epsilon, epsilon))
            return false;
    return true;
}

// template <class V, class D, int ...M>
// inline bool
// operator==(TinyArrayBase<V, D, M...> const & l,
           // lemon::Invalid const &)
// {
    // for(int k=0; k < l.size(); ++k)
        // if(l[k] != -1)
            // return false;
    // return true;
// }

// template <class V, class D, int ...M>
// inline bool
// operator==(lemon::Invalid const &,
           // TinyArrayBase<V, D, M...> const & r)
// {
    // return r == lemon::INVALID;
// }

// template <class V, class D, int ... M>
// inline bool
// operator!=(TinyArrayBase<V, D, M...> const & l,
           // lemon::Invalid const &)
// {
    // return !(l == lemon::INVALID);
// }

// template <class V, class D, int ... M>
// inline bool
// operator!=(lemon::Invalid const &,
           // TinyArrayBase<V, D, M...> const & r)
// {
    // return !(r == lemon::INVALID);
// }



/********************************************************/
/*                                                      */
/*                 TinyArray-Arithmetic                 */
/*                                                      */
/********************************************************/

/** \addtogroup TinyArrayOperators
 */
//@{

#ifdef DOXYGEN
// Declare arithmetic functions for documentation,
// the implementations are provided by a macro below.

    /// scalar add-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator+=(TinyArrayBase<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise add-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator+=(TinyArrayBase<V1, DERIVED, N...> & l,
           TinyArrayBase<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise addition
template <class V1, class D1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator+(TinyArrayBase<V1, D1, N...> const & l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// element-wise scalar addition
template <class V1, class D1, class V2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator+(TinyArrayBase<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar addition
template <class V1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator+(V1 l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// scalar subtract-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator-=(TinyArrayBase<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise subtract-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator-=(TinyArrayBase<V1, DERIVED, N...> & l,
           TinyArrayBase<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise subtraction
template <class V1, class D1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator-(TinyArrayBase<V1, D1, N...> const & l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// element-wise scalar subtraction
template <class V1, class D1, class V2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator-(TinyArrayBase<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar subtraction
template <class V1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator-(V1 l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// scalar multiply-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator*=(TinyArrayBase<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise multiply-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator*=(TinyArrayBase<V1, DERIVED, N...> & l,
           TinyArrayBase<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise multiplication
template <class V1, class D1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator*(TinyArrayBase<V1, D1, N...> const & l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// element-wise scalar multiplication
template <class V1, class D1, class V2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator*(TinyArrayBase<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar multiplication
template <class V1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator*(V1 l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// scalar divide-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator/=(TinyArrayBase<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise divide-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator/=(TinyArrayBase<V1, DERIVED, N...> & l,
           TinyArrayBase<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise division
template <class V1, class D1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator/(TinyArrayBase<V1, D1, N...> const & l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// element-wise scalar division
template <class V1, class D1, class V2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator/(TinyArrayBase<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar division
template <class V1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator/(V1 l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// scalar modulo-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator%=(TinyArrayBase<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise modulo-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator%=(TinyArrayBase<V1, DERIVED, N...> & l,
           TinyArrayBase<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise modulo
template <class V1, class D1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator%(TinyArrayBase<V1, D1, N...> const & l,
          TinyArrayBase<V2, D2, N...> const & r);

    /// element-wise scalar modulo
template <class V1, class D1, class V2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator%(TinyArrayBase<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar modulo
template <class V1, class V2, class D2, int ... N>
TinyArray<PromoteType<V1, V2>, N...>
operator%(V1 l,
          TinyArrayBase<V2, D2, N...> const & r);

#else

#define XTENSOR_TINYARRAY_OPERATORS(OP) \
template <class V1, class DERIVED, int ... N, class V2, \
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value && \
                        std::is_convertible<V2, V1>::value> > \
DERIVED & \
operator OP##=(TinyArrayBase<V1, DERIVED, N...> & l, \
               V2 r) \
{ \
    for(int i=0; i<l.size(); ++i) \
        l[i] OP##= r; \
    return static_cast<DERIVED &>(l); \
} \
 \
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... M, int ... N> \
inline DERIVED &  \
operator OP##=(TinyArrayBase<V1, DERIVED, M...> & l, \
               TinyArrayBase<V2, OTHER_DERIVED, N...> const & r) \
{ \
    XTENSOR_ASSERT_MSG(l.size() == r.size(), \
        "TinyArrayBase::operator" #OP "=(): size mismatch."); \
    for(int i=0; i<l.size(); ++i) \
        l[i] OP##= r[i]; \
    return static_cast<DERIVED &>(l); \
} \
template <class V1, class D1, class V2, class D2, int ... N> \
inline \
TinyArray<decltype((*(V1*)0) OP (*(V2*)0)), N...> \
operator OP(TinyArrayBase<V1, D1, N...> const & l, \
            TinyArrayBase<V2, D2, N...> const & r) \
{ \
    TinyArray<decltype((*(V1*)0) OP (*(V2*)0)), N...> res(l); \
    return res OP##= r; \
} \
 \
template <class V1, class D1, class V2, int ... N, \
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value> >\
inline \
TinyArray<decltype((*(V1*)0) OP (*(V2*)0)), N...> \
operator OP(TinyArrayBase<V1, D1, N...> const & l, \
            V2 r) \
{ \
    TinyArray<decltype((*(V1*)0) OP (*(V2*)0)), N...> res(l); \
    return res OP##= r; \
} \
 \
template <class V1, class V2, class D2, int ... N, \
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value && \
                        !std::is_base_of<std::ios_base, V1>::value> >\
inline \
TinyArray<decltype((*(V1*)0) OP (*(V2*)0)), N...> \
operator OP(V1 l, \
             TinyArrayBase<V2, D2, N...> const & r) \
{ \
    TinyArray<decltype((*(V1*)0) OP (*(V2*)0)), N...> res{l}; \
    return res OP##= r; \
} \
 \
template <class V1, class V2, class D2, \
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value && \
                        !std::is_base_of<std::ios_base, V1>::value> >\
inline \
TinyArray<decltype((*(V1*)0) OP (*(V2*)0)), runtime_size> \
operator OP(V1 l, \
             TinyArrayBase<V2, D2, runtime_size> const & r) \
{ \
    TinyArray<decltype((*(V1*)0) OP (*(V2*)0)), runtime_size> res(tags::size=r.size(), l); \
    return res OP##= r; \
}

XTENSOR_TINYARRAY_OPERATORS(+)
XTENSOR_TINYARRAY_OPERATORS(-)
XTENSOR_TINYARRAY_OPERATORS(*)
XTENSOR_TINYARRAY_OPERATORS(/)
XTENSOR_TINYARRAY_OPERATORS(%)
XTENSOR_TINYARRAY_OPERATORS(&)
XTENSOR_TINYARRAY_OPERATORS(|)
XTENSOR_TINYARRAY_OPERATORS(^)
XTENSOR_TINYARRAY_OPERATORS(<<)
XTENSOR_TINYARRAY_OPERATORS(>>)

#undef XTENSOR_TINYARRAY_OPERATORS

#endif // DOXYGEN

#define XTENSOR_TINYARRAY_UNARY_FUNCTION(FCT) \
template <class V, class D, int ... N> \
inline \
TinyArray<BoolPromote<decltype(mathfunctions::FCT(*(V*)0))>, N...> \
FCT(TinyArrayBase<V, D, N...> const & v) \
{ \
    TinyArray<BoolPromote<decltype(mathfunctions::FCT(*(V*)0))>, N...> res(v.size(), DontInit); \
    for(int k=0; k < v.size(); ++k) \
        res[k] = mathfunctions::FCT(v[k]); \
    return res; \
}

XTENSOR_TINYARRAY_UNARY_FUNCTION(abs)
XTENSOR_TINYARRAY_UNARY_FUNCTION(fabs)

XTENSOR_TINYARRAY_UNARY_FUNCTION(cos)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sin)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tan)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sin_pi)
XTENSOR_TINYARRAY_UNARY_FUNCTION(cos_pi)
XTENSOR_TINYARRAY_UNARY_FUNCTION(acos)
XTENSOR_TINYARRAY_UNARY_FUNCTION(asin)
XTENSOR_TINYARRAY_UNARY_FUNCTION(atan)

XTENSOR_TINYARRAY_UNARY_FUNCTION(cosh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sinh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tanh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(acosh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(asinh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(atanh)

XTENSOR_TINYARRAY_UNARY_FUNCTION(sqrt)
XTENSOR_TINYARRAY_UNARY_FUNCTION(cbrt)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sqrti)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sq)
XTENSOR_TINYARRAY_UNARY_FUNCTION(elementwiseNorm)
XTENSOR_TINYARRAY_UNARY_FUNCTION(elementwiseSquaredNorm)

XTENSOR_TINYARRAY_UNARY_FUNCTION(exp)
XTENSOR_TINYARRAY_UNARY_FUNCTION(exp2)
XTENSOR_TINYARRAY_UNARY_FUNCTION(expm1)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log2)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log10)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log1p)
XTENSOR_TINYARRAY_UNARY_FUNCTION(logb)
XTENSOR_TINYARRAY_UNARY_FUNCTION(ilogb)

XTENSOR_TINYARRAY_UNARY_FUNCTION(ceil)
XTENSOR_TINYARRAY_UNARY_FUNCTION(floor)
XTENSOR_TINYARRAY_UNARY_FUNCTION(trunc)
XTENSOR_TINYARRAY_UNARY_FUNCTION(round)
XTENSOR_TINYARRAY_UNARY_FUNCTION(lround)
XTENSOR_TINYARRAY_UNARY_FUNCTION(llround)
XTENSOR_TINYARRAY_UNARY_FUNCTION(roundi)
XTENSOR_TINYARRAY_UNARY_FUNCTION(even)
XTENSOR_TINYARRAY_UNARY_FUNCTION(odd)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sign)
XTENSOR_TINYARRAY_UNARY_FUNCTION(signi)

XTENSOR_TINYARRAY_UNARY_FUNCTION(erf)
XTENSOR_TINYARRAY_UNARY_FUNCTION(erfc)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tgamma)
XTENSOR_TINYARRAY_UNARY_FUNCTION(lgamma)
XTENSOR_TINYARRAY_UNARY_FUNCTION(loggamma)

XTENSOR_TINYARRAY_UNARY_FUNCTION(conj)
XTENSOR_TINYARRAY_UNARY_FUNCTION(real)
XTENSOR_TINYARRAY_UNARY_FUNCTION(imag)
XTENSOR_TINYARRAY_UNARY_FUNCTION(arg)

#undef XTENSOR_TINYARRAY_UNARY_FUNCTION

    /// Arithmetic negation
template <class V, class D, int ... N>
inline
TinyArray<V, N...>
operator-(TinyArrayBase<V, D, N...> const & v)
{
    TinyArray<V, N...> res(v.size(), DontInit);
    for(int k=0; k < v.size(); ++k)
        res[k] = -v[k];
    return res;
}

    /// Boolean negation
template <class V, class D, int ... N>
inline
TinyArray<V, N...>
operator!(TinyArrayBase<V, D, N...> const & v)
{
    TinyArray<V, N...> res(v.size(), DontInit);
    for(int k=0; k < v.size(); ++k)
        res[k] = !v[k];
    return res;
}

    /// Bitwise negation
template <class V, class D, int ... N>
inline
TinyArray<V, N...>
operator~(TinyArrayBase<V, D, N...> const & v)
{
    TinyArray<V, N...> res(v.size(), DontInit);
    for(int k=0; k < v.size(); ++k)
        res[k] = ~v[k];
    return res;
}

#define XTENSOR_TINYARRAY_BINARY_FUNCTION(FCT) \
template <class V1, class D1, class V2, class D2, int ... N> \
inline \
TinyArray<decltype(mathfunctions::FCT(*(V1*)0, *(V2*)0)), N...> \
FCT(TinyArrayBase<V1, D1, N...> const & l, \
    TinyArrayBase<V2, D2, N...> const & r) \
{ \
    XTENSOR_ASSERT_MSG(l.size() == r.size(), #FCT "(TinyArray, TinyArray): size mismatch."); \
    TinyArray<decltype(mathfunctions::FCT(*(V1*)0, *(V2*)0)), N...> res(l.size(), DontInit); \
    for(int k=0; k < l.size(); ++k) \
        res[k] = mathfunctions::FCT(l[k], r[k]); \
    return res; \
}

XTENSOR_TINYARRAY_BINARY_FUNCTION(atan2)
XTENSOR_TINYARRAY_BINARY_FUNCTION(copysign)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fdim)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmax)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmin)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmod)
XTENSOR_TINYARRAY_BINARY_FUNCTION(hypot)

#undef XTENSOR_TINYARRAY_BINARY_FUNCTION

    /** Apply pow() function to each vector component.
    */
template <class V, class D, class E, int ... N>
inline auto
pow(TinyArrayBase<V, D, N...> const & v, E exponent) ->
         TinyArray<decltype(mathfunctions::pow(v[0], exponent)), N...>
{
    TinyArray<decltype(mathfunctions::pow(v[0], exponent)), N...> res(v.size(), DontInit);
    for(int k=0; k < v.size(); ++k)
        res[k] = mathfunctions::pow(v[k], exponent);
    return res;
}

    /// cross product
template <class V1, class D1, class V2, class D2, int N,
          XTENSOR_REQUIRE<N == 3 || N == runtime_size> >
inline
TinyArray<PromoteType<V1, V2>, N>
cross(TinyArrayBase<V1, D1, N> const & r1,
      TinyArrayBase<V2, D2, N> const & r2)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N, r1.size() == 3 && r2.size() == 3,
        "cross(): cross product requires size() == 3.");
    typedef TinyArray<PromoteType<V1, V2>, N> Res;
    return  Res{r1[1]*r2[2] - r1[2]*r2[1],
                r1[2]*r2[0] - r1[0]*r2[2],
                r1[0]*r2[1] - r1[1]*r2[0]};
}

    /// dot product of two vectors
template <class V1, class D1, class V2, class D2, int N, int M>
inline
PromoteType<V1, V2>
dot(TinyArrayBase<V1, D1, N> const & l,
    TinyArrayBase<V2, D2, M> const & r)
{
    XTENSOR_ASSERT_MSG(l.size() == r.size(), "dot(): size mismatch.");
    PromoteType<V1, V2> res = PromoteType<V1, V2>();
    for(int k=0; k < l.size(); ++k)
        res += l[k] * r[k];
    return res;
}

    /// sum of the vector's elements
template <class V, class D, int ... N>
inline
PromoteType<V>
sum(TinyArrayBase<V, D, N...> const & l)
{
    PromoteType<V> res = PromoteType<V>();
    for(int k=0; k < l.size(); ++k)
        res += l[k];
    return res;
}

    /// mean of the vector's elements
template <class V, class D, int ... N>
inline RealPromoteType<V>
mean(TinyArrayBase<V, D, N...> const & t)
{
    using Promote = RealPromoteType<V>;
    const Promote sumVal = static_cast<Promote>(sum(t));
    if(t.size() > 0)
        return sumVal / t.size();
    else
        return sumVal;
}

    /// cumulative sum of the vector's elements
template <class V, class D, int ... N>
inline
TinyArray<PromoteType<V>, N...>
cumsum(TinyArrayBase<V, D, N...> const & l)
{
    TinyArray<PromoteType<V>, N...> res(l);
    for(int k=1; k < l.size(); ++k)
        res[k] += res[k-1];
    return res;
}

    /// product of the vector's elements
template <class V, class D, int ... N>
inline
PromoteType<V>
prod(TinyArrayBase<V, D, N...> const & l)
{
    using Promote = PromoteType<V>;
    if(l.size() == 0)
        return Promote();
    Promote res = Promote(1);
    for(int k=0; k < l.size(); ++k)
        res *= l[k];
    return res;
}

    /// cumulative product of the vector's elements
template <class V, class D, int ... N>
inline
TinyArray<PromoteType<V>, N...>
cumprod(TinyArrayBase<V, D, N...> const & l)
{
    TinyArray<PromoteType<V>, N...> res(l);
    for(int k=1; k < l.size(); ++k)
        res[k] *= res[k-1];
    return res;
}

    // /// \brief compute the F-order or C-order (default) stride of a given shape.
    // /// Example: {200, 100, 50}  =>  {5000, 50, 1}
// template <class V, class D, int N>
// inline
// TinyArray<PromoteType<V>, N>
// shapeToStrides(TinyArrayBase<V, D, N> const & shape,
               // MemoryOrder order = C_ORDER)
// {
    // TinyArray<PromoteType<V>, N> res(shape.size(), DontInit);

    // if(order == C_ORDER)
    // {
        // res[shape.size()-1] = 1;
        // for(int k=shape.size()-2; k >= 0; --k)
            // res[k] = res[k+1] * shape[k+1];
    // }
    // else
    // {
        // res[0] = 1;
        // for(int k=1; k < shape.size(); ++k)
            // res[k] = res[k-1] * shape[k-1];
    // }
    // return res;
// }

    /// element-wise minimum
template <class V1, class D1, class V2, class D2, int ... N>
inline
TinyArray<PromoteType<V1, V2>, N...>
min(TinyArrayBase<V1, D1, N...> const & l,
    TinyArrayBase<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "min(): size mismatch.");
    TinyArray<PromoteType<V1, V2>, N...> res(l.size(), DontInit);
    for(int k=0; k < l.size(); ++k)
        res[k] =  mathfunctions::min(l[k], r[k]);
    return res;
}

    /// element-wise minimum with a constant
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value>>
inline
TinyArray<PromoteType<V1, V2>, N...>
min(TinyArrayBase<V1, D1, N...> const & l,
    V2 const & r)
{
    TinyArray<PromoteType<V1, V2>, N...> res(l.size(), DontInit);
    for(int k=0; k < l.size(); ++k)
        res[k] =  mathfunctions::min(l[k], r);
    return res;
}

    /// element-wise minimum with a constant
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value>>
inline
TinyArray<PromoteType<V1, V2>, N...>
min(V1 const & l,
    TinyArrayBase<V2, D2, N...> const & r)
{
    TinyArray<PromoteType<V1, V2>, N...> res(r.size(), DontInit);
    for(int k=0; k < r.size(); ++k)
        res[k] =  mathfunctions::min(l, r[k]);
    return res;
}

    /** Index of minimal element.

        Returns -1 for an empty array.
    */
template <class V, class D, int ... N>
inline int
min_element(TinyArrayBase<V, D, N...> const & l)
{
    if(l.size() == 0)
        return -1;
    int m = 0;
    for(int i=1; i<l.size(); ++i)
        if(l[i] < l[m])
            m = i;
    return m;
}

    /// minimal element
template <class V, class D, int ... N>
inline
V const &
min(TinyArrayBase<V, D, N...> const & l)
{
    int m = min_element(l);
    xtensor_precondition(m >= 0, "min() on empty TinyArray.");
    return l[m];
}

    /// element-wise maximum
template <class V1, class D1, class V2, class D2, int ... N>
inline
TinyArray<PromoteType<V1, V2>, N...>
max(TinyArrayBase<V1, D1, N...> const & l,
    TinyArrayBase<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "max(): size mismatch.");
    TinyArray<PromoteType<V1, V2>, N...> res(l.size(), DontInit);
    for(int k=0; k < l.size(); ++k)
        res[k] =  mathfunctions::max(l[k], r[k]);
    return res;
}

    /// element-wise maximum with a constant
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V2>::value>>
inline
TinyArray<PromoteType<V1, V2>, N...>
max(TinyArrayBase<V1, D1, N...> const & l,
    V2 const & r)
{
    TinyArray<PromoteType<V1, V2>, N...> res(l.size(), DontInit);
    for(int k=0; k < l.size(); ++k)
        res[k] =  mathfunctions::max(l[k], r);
    return res;
}

    /// element-wise maximum with a constant
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!TinyArrayConcept<V1>::value>>
inline
TinyArray<PromoteType<V1, V2>, N...>
max(V1 const & l,
    TinyArrayBase<V2, D2, N...> const & r)
{
    TinyArray<PromoteType<V1, V2>, N...> res(r.size(), DontInit);
    for(int k=0; k < r.size(); ++k)
        res[k] =  mathfunctions::max(l, r[k]);
    return res;
}

    /** Index of maximal element.

        Returns -1 for an empty array.
    */
template <class V, class D, int ... N>
inline int
max_element(TinyArrayBase<V, D, N...> const & l)
{
    if(l.size() == 0)
        return -1;
    int m = 0;
    for(int i=1; i<l.size(); ++i)
        if(l[m] < l[i])
            m = i;
    return m;
}

    /// maximal element
template <class V, class D, int ... N>
inline V const &
max(TinyArrayBase<V, D, N...> const & l)
{
    int m = max_element(l);
    xtensor_precondition(m >= 0, "max() on empty TinyArray.");
    return l[m];
}

/// squared norm
template <class V, class D, int ... N>
inline SquaredNormType<TinyArrayBase<V, D, N...> >
squaredNorm(TinyArrayBase<V, D, N...> const & t)
{
    using Type = SquaredNormType<TinyArrayBase<V, D, N...> >;
    Type result = Type();
    for(int i=0; i<t.size(); ++i)
        result += squaredNorm(t[i]);
    return result;
}

template <class V, int N>
inline SquaredNormType<TinySymmetricView<V, N> >
squaredNorm(TinySymmetricView<V, N> const & t)
{
    using Type = SquaredNormType<TinySymmetricView<V, N> >;
    Type result = Type();
    for (int i = 0; i < N; ++i)
    {
        result += squaredNorm(t(i, i));
        for (int j = i + 1; j < N; ++j)
        {
            auto c = squaredNorm(t(i, j));
            result += c + c;
        }
    }

    return result;
}

template <class V, class D, int ... N>
inline
NormType<V>
sizeDividedSquaredNorm(TinyArrayBase<V, D, N...> const & t)
{
    return NormType<V>(squaredNorm(t)) / t.size();
}

template <class V, class D, int ... N>
inline
NormType<V>
sizeDividedNorm(TinyArrayBase<V, D, N...> const & t)
{
    return NormType<V>(norm(t)) / t.size();
}

    /// reversed copy
template <class V, class D, int ... N>
inline
TinyArray<V, N...>
reversed(TinyArrayBase<V, D, N...> const & t)
{
    return TinyArray<V, N...>(t.begin(), t.end(), ReverseCopy);
}

    /** \brief transposed copy

        Elements are arranged such that <tt>res[k] = t[permutation[k]]</tt>.
    */
template <class V1, class D1, class V2, class D2, int N, int M>
inline
TinyArray<V1, N>
transpose(TinyArrayBase<V1, D1, N> const & v,
          TinyArrayBase<V2, D2, M> const & permutation)
{
    return v.transpose(permutation);
}

template <class V1, class D1, int N>
inline
TinyArray<V1, N>
transpose(TinyArrayBase<V1, D1, N> const & v)
{
    return reversed(v);
}

template <class V1, class D1, int N1, int N2>
inline
TinyArray<V1, N2, N1>
transpose(TinyArrayBase<V1, D1, N1, N2> const & v)
{
    TinyArray<V1, N2, N1> res(DontInit);
    for(int i=0; i < N1; ++i)
    {
        for(int j=0; j < N2; ++j)
        {
            res(j,i) = v(i,j);
        }
    }
    return res;
}

template <class V, int N>
inline
TinySymmetricView<V, N>
transpose(TinySymmetricView<V, N> const & v)
{
    return v;
}

    /** \brief Clip negative values.

        All elements smaller than 0 are set to zero.
    */
template <class V, class D, int ... N>
inline
TinyArray<V, N...>
clipLower(TinyArrayBase<V, D, N...> const & t)
{
    return clipLower(t, V());
}

    /** \brief Clip values below a threshold.

        All elements smaller than \a val are set to \a val.
    */
template <class V, class D, int ... N>
inline
TinyArray<V, N...>
clipLower(TinyArrayBase<V, D, N...> const & t, const V val)
{
    TinyArray<V, N...> res(t.size(), DontInit);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] = t[k] < val ? val :  t[k];
    }
    return res;
}

    /** \brief Clip values above a threshold.

        All elements bigger than \a val are set to \a val.
    */
template <class V, class D, int ... N>
inline
TinyArray<V, N...>
clipUpper(TinyArrayBase<V, D, N...> const & t, const V val)
{
    TinyArray<V, N...> res(t.size(), DontInit);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] = t[k] > val ? val :  t[k];
    }
    return res;
}

    /** \brief Clip values to an interval.

        All elements less than \a valLower are set to \a valLower, all elements
        bigger than \a valUpper are set to \a valUpper.
    */
template <class V, class D, int ... N>
inline
TinyArray<V, N...>
clip(TinyArrayBase<V, D, N...> const & t,
     const V valLower, const V valUpper)
{
    TinyArray<V, N...> res(t.size(), DontInit);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] =  (t[k] < valLower)
                       ? valLower
                       : (t[k] > valUpper)
                             ? valUpper
                             : t[k];
    }
    return res;
}

    /** \brief Clip values to a vector of intervals.

        All elements less than \a valLower are set to \a valLower, all elements
        bigger than \a valUpper are set to \a valUpper.
    */
template <class V, class D1, class D2, class D3, int ... N>
inline
TinyArray<V, N...>
clip(TinyArrayBase<V, D1, N...> const & t,
     TinyArrayBase<V, D2, N...> const & valLower,
     TinyArrayBase<V, D3, N...> const & valUpper)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., t.size() == valLower.size() && t.size() == valUpper.size(),
        "clip(): size mismatch.");
    TinyArray<V, N...> res(t.size(), DontInit);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] =  (t[k] < valLower[k])
                       ? valLower[k]
                       : (t[k] > valUpper[k])
                             ? valUpper[k]
                             : t[k];
    }
    return res;
}

template <class T1, class D1, class T2, class D2, int ... N>
inline void
swap(TinyArrayBase<T1, D1, N...> & l,
     TinyArrayBase<T2, D2, N...> & r)
{
    l.swap(r);
}

//@}

#if 0 //#####################################
////////////////////////////////////////////////////////////
// PromoteTraits specializations

template <class T1, class D1, class T2, class D2, int ...N>
struct PromoteTraits<TinyArrayBase<T1, D1, N...>, TinyArrayBase<T2, D2, N...> >
{
    static const bool value = PromoteTraits<T1, T2>::value;
    typedef TinyArray<PromoteType<T1, T2>, N...>     type;
};

template <class T1, class T2, int ...N>
struct PromoteTraits<TinyArray<T1, N...>, TinyArray<T2, N...> >
{
    static const bool value = PromoteTraits<T1, T2>::value;
    typedef TinyArray<PromoteType<T1, T2>, N...>      type;
};

template <class T1, class T2, int ...N>
struct PromoteTraits<TinyArrayView<T1, N...>, TinyArrayView<T2, N...> >
{
    static const bool value = PromoteTraits<T1, T2>::value;
    typedef TinyArray<PromoteType<T1, T2>, N...>      type;
};

template <class T1, class T2, int N>
struct PromoteTraits<TinySymmetricView<T1, N>, TinySymmetricView<T2, N> >
{
    static const bool value = PromoteTraits<T1, T2>::value;
    typedef TinyArray<PromoteType<T1, T2>, N*(N+1)/2>  type;
};

////////////////////////////////////////////////////////////
// NumericTraits specializations

template <class T, class D, int ...N>
struct NumericTraits<TinyArrayBase<T, D, N...>>
{
    typedef TinyArrayBase<T, D, N...>  Type;
    typedef T                          value_type;
    typedef PromoteType<Type>          Promote;
    typedef RealPromoteType<Type>      RealPromote;
    typedef TinyArray<typename NumericTraits<T>::UnsignedPromote, N...>               UnsignedPromote;
    typedef TinyArray<std::complex<typename NumericTraits<T>::ComplexPromote>, N...>  ComplexPromote;

    static Type zero() { return {}; }
    static Type one() { return {NumericTraits<T>::one()}; }
    static Type nonZero() { return {NumericTraits<T>::one()}; }
    static Type epsilon() { return {NumericTraits<T>::epsilon()}; }
    static Type smallestPositive() { return {NumericTraits<T>::smallestPositive()}; }

    static const std::ptrdiff_t static_size = Type::static_size;

    static Promote toPromote(Type const & v) { return v; }
    static Type    fromPromote(Promote const & v) { return v; }
    static Type    fromRealPromote(RealPromote v) { return Type(v); }
};

template <class T, int ...N>
struct NumericTraits<TinyArray<T, N...>>
: public NumericTraits<typename TinyArray<T, N...>::base_type>
{
    typedef TinyArray<T, N...>  Type;
};

template <class T, int ...N>
struct NumericTraits<TinyArrayView<T, N...>>
: public NumericTraits<typename TinyArrayView<T, N...>::base_type>
{
    typedef TinyArrayView<T, N...>  Type;
};

template <class T, int N>
struct NumericTraits<TinySymmetricView<T, N>>
: public NumericTraits<typename TinySymmetricView<T, N>::base_type>
{
    typedef TinySymmetricView<T, N>  Type;
};

// mask cl.exe shortcomings [end]
// #if defined(_MSC_VER)
// #pragma warning( pop )
// #endif

#endif // if 0 //#####################################

} // namespace xt

#undef XTENSOR_ASSERT_INSIDE

#endif // XTENSOR_XTINY_HPP
