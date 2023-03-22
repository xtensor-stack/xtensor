#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cmath>
#include <limits>
#include <type_traits>

#include "xtensor/xexpression.hpp"

namespace xt
{
    namespace detail
    {
        template <class T>
        bool check_is_small(const T& value, const T& tolerance)
        {
            using std::abs;
            return abs(value) < abs(tolerance);
        }

        template <class T>
        T safe_division(const T& lhs, const T& rhs)
        {
            if (rhs < static_cast<T>(1) && lhs > rhs * (std::numeric_limits<T>::max)())
            {
                return (std::numeric_limits<T>::max)();
            }
            if ((lhs == static_cast<T>(0))
                || (rhs > static_cast<T>(1) && lhs < rhs * (std::numeric_limits<T>::min)()))
            {
                return static_cast<T>(0);
            }
            return lhs / rhs;
        }

        template <class T>
        bool check_is_close(const T& lhs, const T& rhs, const T& relative_precision)
        {
            using std::abs;
            T diff = abs(lhs - rhs);
            T d1 = safe_division(diff, T(abs(rhs)));
            T d2 = safe_division(diff, T(abs(lhs)));

            return d1 <= relative_precision && d2 <= relative_precision;
        }
    }

    template <class T>
    bool scalar_near(const T& lhs, const T& rhs)
    {
        using std::abs;
        using std::max;

        if (std::isnan(lhs))
        {
            return std::isnan(rhs);
        }

        if (std::isinf(lhs))
        {
            return std::isinf(rhs) && (lhs * rhs > 0) /* same sign */;
        }

        T relative_precision = 2048 * std::numeric_limits<T>::epsilon();
        T absolute_zero_prox = 2048 * std::numeric_limits<T>::epsilon();

        if (max(abs(lhs), abs(rhs)) < T(1e-3))
        {
            using res_type = decltype(lhs - rhs);
            return detail::check_is_small(lhs - rhs, res_type(absolute_zero_prox));
        }
        else
        {
            return detail::check_is_close(lhs, rhs, relative_precision);
        }
    }

    template <class T>
    bool scalar_near(const std::complex<T>& lhs, const std::complex<T>& rhs)
    {
        return scalar_near(lhs.real(), rhs.real()) && scalar_near(lhs.imag(), rhs.imag());
    }

    template <class E1, class E2>
    bool tensor_near(const E1& e1, const E2& e2)
    {
        bool res = e1.dimension() == e2.dimension()
                   && std::equal(e1.shape().begin(), e1.shape().end(), e2.shape().begin());
        auto iter1 = e1.begin();
        auto iter2 = e2.begin();
        auto iter_end = e1.end();
        while (res && iter1 != iter_end)
        {
            res = scalar_near(*iter1++, *iter2++);
        }
        return res;
    }
}

#endif
