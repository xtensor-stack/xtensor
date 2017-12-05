/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief standard mathematical functions for xexpressions
 */

#ifndef XMATHUTIL_HPP
#define XMATHUTIL_HPP

#include <cmath>
#include <cstdlib>   // std::abs(int) prior to C++ 17
#include <complex>
#include <algorithm> // std::min, std::max
#include <type_traits>

#include "xconcepts.hpp"
#include "xutils.hpp"

namespace xt
{
    template <class T = double>
    struct numeric_constants
    {
        static constexpr T PI = 3.141592653589793238463;
        static constexpr T PI_2 = 1.57079632679489661923;
        static constexpr T PI_4 = 0.785398163397448309616;
        static constexpr T D_1_PI = 0.318309886183790671538;
        static constexpr T D_2_PI = 0.636619772367581343076;
        static constexpr T D_2_SQRTPI = 1.12837916709551257390;
        static constexpr T SQRT2 = 1.41421356237309504880;
        static constexpr T SQRT1_2 = 0.707106781186547524401;
        static constexpr T E = 2.71828182845904523536;
        static constexpr T LOG2E = 1.44269504088896340736;
        static constexpr T LOG10E = 0.434294481903251827651;
        static constexpr T LN2 = 0.693147180559945309417;
    };

    /***********
     * Helpers *
     ***********/

#define UNSIGNED_ABS_FUNC(T)                                                     \
constexpr inline T abs(const T& x)                                               \
{                                                                                \
    return x;                                                                    \
}                                                                                \

#define INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, T)                        \
constexpr inline bool FUNC_NAME(const T& /*x*/) noexcept                         \
{                                                                                \
    return RETURN_VAL;                                                           \
}                                                                                \

#define INT_SPECIALIZATION(FUNC_NAME, RETURN_VAL)                                \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, char);                            \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, short);                           \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, int);                             \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, long);                            \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, long long);                       \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned char);                   \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned short);                  \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned int);                    \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned long);                   \
INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned long long);              \

    namespace math
    {
        using std::abs;
        using std::fabs;

        using std::cos;
        using std::sin;
        using std::tan;
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

        using std::exp;
        using std::exp2;
        using std::expm1;
        using std::log;
        using std::log2;
        using std::log10;
        using std::log1p;
        using std::logb;
        using std::ilogb;

        using std::floor;
        using std::ceil;
        using std::trunc;
        using std::round;
        using std::lround;
        using std::llround;
        using std::rint;
        using std::nearbyint;
        using std::remainder;

        using std::erf;
        using std::erfc;
        using std::erfc;
        using std::tgamma;
        using std::lgamma;

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

        using std::fma;

        using std::isnan;
        using std::isinf;
        using std::isfinite;
        using std::fpclassify;

        // Overload isinf, isnan and isfinite for complex datatypes,
        // following the Python specification:
        template <class T>
        inline bool isinf(const std::complex<T>& c)
        {
            return std::isinf(std::real(c)) || std::isinf(std::imag(c));
        }

        template <class T>
        inline bool isnan(const std::complex<T>& c)
        {
            return std::isnan(std::real(c)) || std::isnan(std::imag(c));
        }

        template <class T>
        inline bool isfinite(const std::complex<T>& c)
        {
            return !isinf(c) && !isnan(c);
        }

        // The following specializations are needed to avoid 'ambiguous overload' errors,
        // whereas 'unsigned char' and 'unsigned short' are automatically converted to 'int'.
        UNSIGNED_ABS_FUNC(unsigned int);
        UNSIGNED_ABS_FUNC(unsigned long);
        UNSIGNED_ABS_FUNC(unsigned long long);

#ifdef _WIN32
        INT_SPECIALIZATION(isinf, false);
        INT_SPECIALIZATION(isnan, false);
        INT_SPECIALIZATION(isfinite, true);
#endif

    } // namespace math

#undef UNSIGNED_ABS_FUNC
#undef INT_SPECIALIZATION
#undef INT_SPECIALIZATION_IMPL

    namespace detail
    {
        template <class T>
        struct isclose
        {
            using result_type = bool;
            isclose(double rtol, double atol, bool equal_nan)
            : m_rtol(rtol), m_atol(atol), m_equal_nan(equal_nan)
            {
            }

            bool operator()(const T& a, const T& b) const
            {
                using internal_type = promote_type_t<T, double>;
                if(math::isnan(a) && math::isnan(b))
                {
                    return m_equal_nan;
                }
                if(math::isinf(a) && math::isinf(b))
                {
                    // check for both infinity signs equal
                    return a == b;
                }
                auto d = math::abs((internal_type) a - (internal_type) b);
                return d <= m_atol || d <= m_rtol * (double) std::max(math::abs(a), math::abs(b));
            }

            template <class U>
            struct rebind
            {
                using type = isclose<U>;
            };

        private:
            double m_rtol;
            double m_atol;
            bool m_equal_nan;
        };
    }

    // /*************
     // * isclose() *
     // *************/

    // template <class U, class V,
              // XTENSOR_REQUIRE<std::is_arithmetic<U>::value && std::is_arithmetic<V>::value> >
    // inline bool
    // isclose(U u, V v, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
    // {
        // using namespace cmath;
        // using P = real_promote_t<promote_t<U, V>>;
        // P a = static_cast<P>(u),
          // b = static_cast<P>(v);

        // if(std::isnan(a) && std::isnan(b))
        // {
            // return equal_nan;
        // }
        // if(std::isinf(a) && std::isinf(b))
        // {
            // return std::signbit(a) == std::signbit(b);
        // }
        // P d = abs(a - b);
        // return d <= atol || d <= rtol * std::max(abs(a), abs(b));
    // }

    // /********
     // * sq() *
     // ********/

        // /** \brief The square function.

            // <tt>sq(x) = x*x</tt> is needed so often that it makes sense to define it as a function.
        // */
    // template <class T,
              // XTENSOR_REQUIRE<std::is_arithmetic<T>::value> >
    // inline auto
    // sq(T t)
    // {
        // return t*t;
    // }

    // /*********
     // * min() *
     // *********/

        // /** \brief A proper minimum function.

            // The <tt>std::min</tt> template matches everything -- this is way too
            // greedy to be useful. xtensor implements the basic <tt>min</tt> function
            // only for arithmetic types and provides explicit overloads for everything
            // else. Moreover, xtensor's <tt>min</tt> function also computes the minimum
            // between two different types, as long as they have a <tt>std::common_type</tt>.
        // */
    // template <class T1, class T2,
              // XTENSOR_REQUIRE<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value> >
    // inline std::common_type_t<T1, T2>
    // min(T1 const & t1, T2 const & t2)
    // {
        // using T = std::common_type_t<T1, T2>;
        // return std::min(static_cast<T>(t1), static_cast<T>(t2));
    // }

    // template <class T,
              // XTENSOR_REQUIRE<std::is_arithmetic<T>::value> >
    // inline T const &
    // min(T const & t1, T const & t2)
    // {
        // return std::min(t1, t2);
    // }

    // /*********
     // * max() *
     // *********/

        // /** \brief A proper maximum function.

            // The <tt>std::max</tt> template matches everything -- this is way too
            // greedy to be useful. xtensor implements the basic <tt>max</tt> function
            // only for arithmetic types and provides explicit overloads for everything
            // else. Moreover, xtensor's <tt>max</tt> function also computes the maximum
            // between two different types, as long as they have a <tt>std::common_type</tt>.
        // */
    // template <class T1, class T2,
              // XTENSOR_REQUIRE<std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value> >
    // inline std::common_type_t<T1, T2>
    // max(T1 const & t1, T2 const & t2)
    // {
        // using T = std::common_type_t<T1, T2>;
        // return std::max(static_cast<T>(t1), static_cast<T>(t2));
    // }

    // template <class T,
              // XTENSOR_REQUIRE<std::is_arithmetic<T>::value> >
    // inline T const &
    // max(T const & t1, T const & t2)
    // {
        // return std::max(t1, t2);
    // }

    // /********************
     // * even() and odd() *
     // ********************/

        // /** \brief Check if an integer is even.
        // */
    // template <class T,
              // XTENSOR_REQUIRE<std::is_integral<T>::value> >
    // inline bool
    // even(T const t)
    // {
        // using UT = typename std::make_unsigned<T>::type;
        // return (static_cast<UT>(t)&1) == 0;
    // }

        // /** \brief Check if an integer is odd.
        // */
    // template <class T,
              // XTENSOR_REQUIRE<std::is_integral<T>::value> >
    // inline bool
    // odd(T const t)
    // {
        // using UT = typename std::make_unsigned<T>::type;
        // return (static_cast<UT>(t)&1) != 0;
    // }

    // /**********************
     // * sin_pi(), cos_pi() *
     // **********************/

        // /** \brief sin(pi*x).

            // Essentially calls <tt>std::sin(PI*x)</tt> but uses a more accurate implementation
            // to make sure that <tt>sin_pi(1.0) == 0.0</tt> (which does not hold for
            // <tt>std::sin(PI)</tt> due to round-off error), and <tt>sin_pi(0.5) == 1.0</tt>.
        // */
    // template <class REAL,
              // XTENSOR_REQUIRE<std::is_floating_point<REAL>::value> >
    // REAL sin_pi(REAL x)
    // {
        // if(x < 0.0)
        // {
            // return -sin_pi(-x);
        // }
        // if(x < 0.5)
        // {
            // return std::sin(numeric_constants<REAL>::PI * x);
        // }

        // bool invert = false;
        // if(x < 1.0)
        // {
            // invert = true;
            // x = -x;
        // }

        // REAL rem = std::floor(x);
        // if(odd((int64_t)rem))
        // {
            // invert = !invert;
        // }
        // rem = x - rem;
        // if(rem > 0.5)
        // {
            // rem = 1.0 - rem;
        // }
        // if(rem == 0.5)
        // {
            // rem = REAL(1);
        // }
        // else
        // {
            // rem = std::sin(numeric_constants<REAL>::PI * rem);
        // }
        // return invert
                  // ? -rem
                  // : rem;
    // }

        // /** \brief cos(pi*x).

            // Essentially calls <tt>std::cos(PI*x)</tt> but uses a more accurate implementation
            // to make sure that <tt>cos_pi(1.0) == -1.0</tt> and <tt>cos_pi(0.5) == 0.0</tt>.
        // */
    // template <class REAL,
              // XTENSOR_REQUIRE<std::is_floating_point<REAL>::value> >
    // REAL cos_pi(REAL x)
    // {
        // return sin_pi(x+0.5);
    // }

    // /******************
     // * norm functions *
     // ******************/

        // /** \brief The L2-norm of a numerical object.

            // For scalar types: implemented as <tt>abs(t)</tt><br>
            // otherwise: implemented as <tt>sqrt(norm_sq(t))</tt>.
        // */
    // template <class T>
    // inline auto norm_l2(T const & t)
    // {
        // using cmath::sqrt;
        // return sqrt(norm_sq(t));
    // }

        // /** \brief The infinity norm of a numerical object.

            // Defined as an alias for <tt>norm_max()</tt>.
        // */
    // template <class T>
    // inline auto norm_linf(T const & t)
    // {
        // return norm_max(l);
    // }

        // /** \brief Squared norm of a complex number.

            // Equivalent to <tt>std::norm(t)</tt> (yes, the C++ standard really defines
            // <tt>norm()</tt> to compute the squared norm).
        // */
    // template <class T>
    // inline auto
    // norm_sq(std::complex<T> const & t)
    // {
        // return std::norm(t);
    // }

    // #define XTENSOR_DEFINE_NORM(T)                                         \
        // inline auto     norm_lp(T t, int p) -> decltype(cmath::abs(t))     \
                        // {                                                  \
                            // return p == 0                                  \
                                      // ? t != 0                             \
                                          // ? 1                              \
                                          // : 0                              \
                                      // : cmath::abs(t);                     \
                        // }                                                  \
        // inline size_t   norm_l0(T t)         { return t != 0 ? 1 : 0; }    \
        // inline auto     norm_l1(T t)         { return cmath::abs(t); }     \
        // inline auto     norm_l2(T t)         { return cmath::abs(t); }     \
        // inline auto     norm_max(T t)        { return cmath::abs(t); }     \
        // inline auto     norm_sq(T t)         { return sq(t); }             \
        // inline auto     mean_square(T t)     { return sq(t); }

    // XTENSOR_DEFINE_NORM(signed char)
    // XTENSOR_DEFINE_NORM(unsigned char)
    // XTENSOR_DEFINE_NORM(short)
    // XTENSOR_DEFINE_NORM(unsigned short)
    // XTENSOR_DEFINE_NORM(int)
    // XTENSOR_DEFINE_NORM(unsigned int)
    // XTENSOR_DEFINE_NORM(long)
    // XTENSOR_DEFINE_NORM(unsigned long)
    // XTENSOR_DEFINE_NORM(long long)
    // XTENSOR_DEFINE_NORM(unsigned long long)
    // XTENSOR_DEFINE_NORM(float)
    // XTENSOR_DEFINE_NORM(double)
    // XTENSOR_DEFINE_NORM(long double)

    // #undef XTENSOR_DEFINE_NORM
} // namespace xt

#endif
