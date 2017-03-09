/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief standard mathematical functions for xexpressions
 */

#ifndef XBUILDER_HPP
#define XBUILDER_HPP

#include <utility>
#include <functional>
#include <cmath>

#include "xfunction.hpp"
#include "xbroadcast.hpp"
#include "xgenerator.hpp"

#ifdef X_OLD_CLANG
    #include <initializer_list>
    #include <vector>
#else
    #include <array>
#endif

namespace xt
{

    /********
     * ones *
     ********/

    /**
     * Returns an \ref xexpression containing ones of the specified shape.
     * @tparam shape the shape of the returned expression.
     */
    template <class T, class S>
    inline auto ones(S shape) noexcept
    {
        return broadcast(T(1), std::forward<S>(shape));
    }

#ifdef X_OLD_CLANG
    template <class T, class I>
    inline auto ones(std::initializer_list<I> shape) noexcept
    {
        return broadcast(T(1), shape);
    }
#else
    template <class T, class I, std::size_t L>
    inline auto ones(const I(&shape)[L]) noexcept
    {
        return broadcast(T(1), shape);
    }
#endif

    /*********
     * zeros *
     *********/

    /**
     * Returns an \ref xexpression containing zeros of the specified shape.
     * @tparam shape the shape of the returned expression.
     */
    template <class T, class S>
    inline auto zeros(S shape) noexcept
    {
        return broadcast(T(0), std::forward<S>(shape));
    }
    
#ifdef X_OLD_CLANG
    template <class T, class I>
    inline auto zeros(std::initializer_list<I> shape) noexcept
    {
        return broadcast(T(0), shape);
    }
#else
    template <class T, class I, std::size_t L>
    inline auto zeros(const I(&shape)[L]) noexcept
    {
        return broadcast(T(0), shape);
    }
#endif

    namespace detail
    {
        template <class T>
        struct arange_impl
        {
            using value_type = T;

            arange_impl(T start, T stop, T step)
                : m_start(start), m_stop(stop), m_step(step)
            {
            }

            template <class... Args>
            inline T operator()(Args... args) const
            {
                return access_impl(args...);
            }

            inline T operator[](const xindex& idx) const
            {
                return m_start + m_step * T(idx[0]);
            }

            template <class It>
            inline T element(It first, It /*last*/) const
            {
                return m_start + m_step * T(*first);
            }

        private:
            value_type m_start;
            value_type m_stop;
            value_type m_step;

            template <class T1, class... Args>
            inline T access_impl(T1 t, Args... /*args*/) const
            {
                return m_start + m_step * T(t);
            }

            inline T access_impl() const
            {
                return m_start;
            }
        };

        template <class F>
        struct fn_impl
        {
            using value_type = typename F::value_type;
            using size_type = std::size_t;

            fn_impl(F&& f) : m_ft(f)
            {
            }

            inline value_type operator()() const
            {
                // special case when called without args (happens when printing)
                return value_type();
            }

            template <class... Args>
            inline value_type operator()(Args... args) const
            {
                size_type idx [sizeof...(Args)] = {static_cast<size_type>(args)...};
                return access_impl(std::begin(idx), std::end(idx));
            }

            inline value_type operator[](const xindex& idx) const
            {
                return access_impl(idx.begin(), idx.end());
            }

            template <class It>
            inline value_type element(It first, It last) const
            {
                return access_impl(first, last);
            }

        private:
            F m_ft;
            template <class It>
            inline value_type access_impl(const It& begin, const It& end) const
            {
                return m_ft(begin, end);
            }
        };

        template <class T>
        struct eye_fn
        {
            using value_type = T;

            eye_fn(int k) : m_k(k)
            {
            }

            template <class It>
            inline T operator()(const It& /*begin*/, const It& end) const
            {
                return *(end - 1) == *(end - 2) + m_k ? T(1) : T(0);
            }

        private:
            int m_k;
        };
    }

    /**
     * Generates an array with ones on the diagonal.
     * @param shape shape of the resulting expression
     * @param k index of the diagonal. 0 (default) refers to the main diagonal,
     *          a positive value refers to an upper diagonal, and a negative
     *          value to a lower diagonal.
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T = bool>
    inline auto eye(const std::vector<std::size_t>& shape, int k = 0)
    {
        return detail::make_xgenerator(detail::fn_impl<detail::eye_fn<T>>(detail::eye_fn<T>(k)), shape);
    }

    /**
     * Generates a (n x n) array with ones on the diagonal.
     * @param n length of the diagonal.
     * @param k index of the diagonal. 0 (default) refers to the main diagonal,
     *          a positive value refers to an upper diagonal, and a negative
     *          value to a lower diagonal.
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T = bool>
    inline auto eye(std::size_t n, int k = 0)
    {
        return eye<T>({n, n}, k);
    }

    /**
     * Generates numbers evenly spaced within given half-open interval [start, stop).
     * @param start start of the interval
     * @param stop stop of the interval
     * @param step stepsize
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto arange(T start, T stop, T step = 1) noexcept
    {
        std::size_t shape = static_cast<std::size_t>(std::ceil((stop - start) / step));
        return detail::make_xgenerator(detail::arange_impl<T>(start, stop, step), {shape});
    }

    /**
     * Generate numbers evenly spaced within given half-open interval [0, stop)
     * with a step size of 1.
     * @param stop stop of the interval
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto arange(T stop) noexcept
    {
        return arange<T>(T(0), stop, T(1));
    }

    /**
     * Generates @a num_samples evenly spaced numbers over given interval
     * @param start start of interval
     * @param stop stop of interval
     * @param num_samples number of samples (defaults to 50)
     * @param endpoint if true, include endpoint (defaults to true)
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto linspace(T start, T stop, std::size_t num_samples = 50, bool endpoint = true) noexcept
    {
        T step = (stop - start) / T(num_samples - (endpoint ? 1 : 0));
        return detail::make_xgenerator(detail::arange_impl<T>(start, stop, step), {num_samples});
    }

    /**
     * Generates @a num_samples numbers evenly spaced on a log scale over given interval
     * @param start start of interval (pow(base, start) is the first value).
     * @param stop stop of interval (pow(base, stop) is the final value, except if endpoint = false)
     * @param num_samples number of samples (defaults to 50)
     * @param base the base of the log space.
     * @param endpoint if true, include endpoint (defaults to true)
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto logspace(T start, T stop, std::size_t num_samples, T base = 10, bool endpoint = true) noexcept
    {
        return pow(std::forward<T>(base), linspace(start, stop, num_samples, endpoint));
    }

    namespace detail
    {
        template <class... CT>
        struct concatenate_impl
        {
            using size_type = std::size_t;
            using value_type = std::common_type_t<typename std::decay_t<CT>::value_type...>;

            inline concatenate_impl(std::tuple<CT...>&& t, std::size_t axis)
                : m_t(t), m_axis(axis)
            {
            }

            template <class... Args>
            inline value_type operator()(Args... args) const
            {
                return access_impl(xindex({{static_cast<size_type>(args)...}}));
            }

            inline value_type operator[](const xindex& idx) const
            {
                return access_impl(idx);
            }

            template <class It>
            inline value_type element(It first, It last) const
            {
                return access_impl(xindex(first, last));
            }

        private:

            inline value_type access_impl(xindex idx) const
            {
                auto match = [this, &idx](auto& arr) {
                    if (idx[this->m_axis] >= arr.shape()[this->m_axis])
                    {
                        idx[this->m_axis] -= arr.shape()[this->m_axis];
                        return false;
                    }
                    return true;
                };

                auto get = [&idx](auto& arr) {
                    return arr[idx];
                };

                std::size_t i = 0;
                for (; i < sizeof...(CT); ++i)
                {
                    if (apply<bool>(i, match, m_t)) 
                    {
                        break;
                    }
                }
                return apply<value_type>(i, get, m_t);
            }

            std::tuple<CT...> m_t;
            size_type m_axis;
        };

        template <class... CT>
        struct stack_impl
        {
            using size_type = std::size_t;
            using value_type = std::common_type_t<typename std::decay_t<CT>::value_type...>;

            inline stack_impl(std::tuple<CT...>&& t, std::size_t axis)
                : m_t(t), m_axis(axis)
            {
            }

            template <class... Args>
            inline value_type operator()(Args... args) const
            {
                return access_impl(xindex({{static_cast<size_type>(args)...}}));
            }

            inline value_type operator[](const xindex& idx) const
            {
                return access_impl(idx);
            }

            template <class It>
            inline value_type element(It first, It last) const
            {
                return access_impl(xindex(first, last));
            }

        private:

            inline value_type access_impl(xindex idx) const
            {
                auto get_item = [&idx](auto& arr) {
                    return arr[idx];
                };
                std::size_t i = idx[m_axis];
                idx.erase(idx.begin() + m_axis);
                return apply<value_type>(i, get_item, m_t);
            }

            const std::tuple<CT...> m_t;
            const size_type m_axis;
        };

        template <class CT>
        struct repeat_impl
        {
            using xexpression_type = std::decay_t<CT>;
            using size_type = typename xexpression_type::size_type;
            using value_type = typename xexpression_type::value_type;

            repeat_impl(CT source, size_type axis) :
                m_source(source), m_axis(axis)
            {
            }

            template <class... Args>
            value_type operator()(Args... args) const
            {
                // to catch a -Wuninitialized
                if (sizeof...(Args))
                {
                    std::array<size_type, sizeof...(Args)> args_arr({static_cast<size_type>(args)...});
                    return m_source(args_arr[m_axis]);
                }
                return m_source();
            }

            value_type operator[](const xindex& idx) const
            {
                return m_source[{ idx[m_axis] }];
            }

            template <class It>
            inline value_type element(It first, It) const
            {
                return m_source[{*(first + m_axis)}];
            }

        private:

            CT m_source;
            size_type m_axis;
        };
    }

    /**
     * @brief Creates tuples from arguments for \ref concatenate and \ref stack.
     *        Very similar to std::make_tuple.
     */
    template <class... Types>
    inline auto xtuple(Types&&... args)
    {
        return std::tuple<detail::const_closure_t<Types>...>(std::forward<Types>(args)...);
    }

    /**
     * @brief Concatenates xexpressions along \em axis.
     *
     * @param t \ref xtuple of xexpressions to concatenate
     * @param axis axis along which elements are concatenated
     * @returns xgenerator evaluating to concatenated elements
     *
     * \code{.cpp}
     * xt::xarray<double> a = {{1, 2, 3}};
     * xt::xarray<double> b = {{2, 3, 4}};
     * xt::xarray<double> c = xt::concatenate(xt::xtuple(a, b)); // => {{1, 2, 3},
     *                                                                  {2, 3, 4}}
     * xt::xarray<double> d = xt::concatenate(xt::xtuple(a, b), 1); // => {{1, 2, 3, 2, 3, 4}}
     * \endcode
     */
    template <class... CT>
    inline auto concatenate(std::tuple<CT...>&& t, std::size_t axis = 0)
    {
        using shape_type = promote_shape_t<typename std::decay_t<CT>::shape_type...>;
        shape_type new_shape = forward_sequence<shape_type>(std::get<0>(t).shape());
        auto shape_at_axis = [&axis](std::size_t prev, auto& arr) -> std::size_t {
            return prev + arr.shape()[axis];
        };
        new_shape[axis] += accumulate(shape_at_axis, std::size_t(0), t) - new_shape[axis];
        return detail::make_xgenerator(detail::concatenate_impl<CT...>(std::forward<std::tuple<CT...>>(t), axis), new_shape);
    }

    namespace detail
    {
        template <class T, std::size_t N>
        inline std::array<T, N + 1> add_axis(std::array<T, N> arr, std::size_t axis, std::size_t value)
        {
            std::array<T, N + 1> temp;
            std::copy(arr.begin(), arr.begin() + axis, temp.begin());
            temp[axis] = value;
            std::copy(arr.begin() + axis, arr.end(), temp.begin() + axis + 1);
            return temp;
        }

        template <class T>
        inline T add_axis(T arr, std::size_t axis, std::size_t value)
        {
            T temp(arr);
            temp.insert(temp.begin() + axis, value);
            return temp;
        }
    }

    /**
     * @brief Stack xexpressions along \em axis.
     *        Stacking always creates a new dimension along which elements are stacked.
     *
     * @param t \ref xtuple of xexpressions to concatenate
     * @param axis axis along which elements are stacked
     * @returns xgenerator evaluating to stacked elements
     *
     * \code{.cpp}
     * xt::xarray<double> a = {1, 2, 3};
     * xt::xarray<double> b = {5, 6, 7};
     * xt::xarray<double> s = xt::stack(xt::xtuple(a, b)); // => {{1, 2, 3},
     *                                                            {5, 6, 7}}
     * xt::xarray<double> t = xt::stack(xt::xtuple(a, b), 1); // => {{1, 5},
     *                                                               {2, 6},
     *                                                               {3, 7}}
     * \endcode
     */
    template <class... CT>
    inline auto stack(std::tuple<CT...>&& t, std::size_t axis = 0)
    {
        using shape_type = promote_shape_t<typename std::decay_t<CT>::shape_type...>;
        auto new_shape = detail::add_axis(forward_sequence<shape_type>(std::get<0>(t).shape()), axis, sizeof...(CT));
        return detail::make_xgenerator(detail::stack_impl<CT...>(std::forward<std::tuple<CT...>>(t), axis), new_shape);
    }

    namespace detail
    {

        template <std::size_t... I, class... E>
        inline auto meshgrid_impl(std::index_sequence<I...>, E&&... e) noexcept
        {
#if defined X_OLD_CLANG || defined _MSC_VER
            const std::array<std::size_t, sizeof...(E)> shape { e.shape()[0]... };
            return std::make_tuple(
                detail::make_xgenerator(
                    detail::repeat_impl<xclosure_t<E>>(std::forward<E>(e), I),
                    shape
                )...
            );
#else
            return std::make_tuple(
                detail::make_xgenerator(
                    detail::repeat_impl<xclosure_t<E>>(std::forward<E>(e), I),
                    { e.shape()[0]... }
                )...
            );
#endif
        }
    }

    /**
     * @brief Return coordinate tensors from coordinate vectors.
     *        Make N-D coordinate tensor expressions for vectorized evaluations of N-D scalar/vector
     *        fields over N-D grids, given one-dimensional coordinate arrays x1, x2,..., xn.
     *
     * @param e xexpressions to concatenate
     * @returns tuple of xgenerator expressions.
     */
    template <class... E>
    inline auto meshgrid(E&&... e) noexcept
    {
        return detail::meshgrid_impl(std::make_index_sequence<sizeof...(E)>(), std::forward<E>(e)...);
    }

    namespace detail
    {
        template <class CT>
        struct diagonal_fn
        {
            using xexpression_type = std::decay_t<CT>;
            using value_type = typename xexpression_type::value_type;

            diagonal_fn(CT source) : m_source(source)
            {
            }

            template <class It>
            inline value_type operator()(It begin, It /*end*/) const
            {
                return m_source(*begin, *begin);
            }

        private:
            CT m_source;
        };

        template <class CT>
        struct diag_fn
        {
            using xexpression_type = std::decay_t<CT>;
            using value_type = typename xexpression_type::value_type;

            diag_fn(CT source) : m_source(source)
            {
            }

            template <class It>
            inline value_type operator()(It begin, It /*end*/) const
            {
                return *begin == *(begin + 1) ? m_source(*begin) : value_type(0);
            }

        private:
            CT m_source;
        };

        template <class CT>
        struct flipud_impl
        {
            using xexpression_type = std::decay_t<CT>;
            using value_type = typename xexpression_type::value_type;
            using size_type = typename xexpression_type::size_type;

            flipud_impl(CT source)
                : m_source(source), m_shape_first(m_source.shape().front() - 1)
            {
            }

            inline value_type operator()() const
            {
                // special case when called without args (happens when printing)
                return value_type();
            }

            template <class... Args>
            inline value_type operator()(std::size_t first_idx, Args... args) const
            {
                return m_source(m_shape_first - first_idx, args...);
            }

            inline value_type operator[](xindex idx) const
            {
                idx.front() = m_shape_first - idx.front();
                return m_source[idx];
            }

            template <class It>
            inline value_type element(It first, It last) const
            {
                return operator[](xindex(first, last));
            }

        private:
            CT m_source;
            const size_type m_shape_first;
        };

        template <class CT>
        struct fliplr_impl
        {
            using xexpression_type = std::decay_t<CT>;
            using value_type = typename xexpression_type::value_type;
            using size_type = typename xexpression_type::size_type;

            fliplr_impl(CT source)
                : m_source(source), m_shape_last(m_source.shape().back() - 1)
            {
            }

            inline value_type operator()() const
            {
                // special case when called without args (happens when printing)
                return value_type();
            }

            template <class... Args>
            inline value_type operator()(Args... args, std::size_t last_idx) const
            {
                return m_source(args..., m_shape_last - last_idx);
            }

            inline value_type operator[](xindex idx) const
            {
                idx.back() = m_shape_last - idx.back();
                return m_source[idx];
            }

            template <class It>
            inline value_type element(It first, It last) const
            {
                return operator[](xindex(first, last));
            }

        private:
            CT m_source;
            const size_type m_shape_last;
        };

        template <class CT, class Comp>
        struct trilu_fn
        {
            using xexpression_type = std::decay_t<CT>;
            using value_type = typename xexpression_type::value_type;
            using signed_idx_type = long int;

            trilu_fn(CT source, int k, Comp comp)
                : m_source(source), m_k(k), m_comp(comp)
            {
            }

            template <class It>
            inline value_type operator()(It begin, It end) const
            {
                // have to cast to signed int otherwise -1 can lead to overflow
                return m_comp(signed_idx_type(*begin) + m_k, signed_idx_type(*(begin + 1))) ? m_source.element(begin, end) : value_type(0);
            }

        private:
            CT m_source;
            const signed_idx_type m_k;
            const Comp m_comp;
        };
    }

    /**
     * @brief Returns the elements on the diagonal of arr
     *
     * @param arr the input array
     * @returns xexpression with values of the diagonal
     *
     * \code{.cpp}
     * xt::xarray<double> a = {{1, 2, 3},
     *                         {4, 5, 6}
     *                         {7, 8, 9}};
     * auto b = xt::diagonal(a); // => {1, 5, 9}
     * \endcode
     */
    template <class E>
    inline auto diagonal(E&& arr)
    {
        using CT = xclosure_t<E>;
        std::size_t s = arr.shape()[0];
        return detail::make_xgenerator(detail::fn_impl<detail::diagonal_fn<CT>>(detail::diagonal_fn<CT>(std::forward<E>(arr))),
                                       { s });
    }

    /**
     * @brief xexpression with values of arr on the diagonal, zeroes otherwise
     *
     * @param arr the 1D input array of length n
     * @returns xexpression function with shape n x n and arr on the diagonal
     *
     * \code{.cpp}
     * xt::xarray<double> a = {1, 5, 9};
     * auto b = xt::diag(a); // => {{1, 0, 0},
     *                       //     {0, 5, 0},
     *                       //     {0, 0, 9}}
     * \endcode
     */
    template <class E>
    inline auto diag(E&& arr)
    {
        using CT = xclosure_t<E>;
        std::size_t s = arr.shape()[0];
        return detail::make_xgenerator(detail::fn_impl<detail::diag_fn<CT>>(detail::diag_fn<CT>(std::forward<E>(arr))),
                                       { s, s });
    }

    /**
     * @brief Flip xexpression in the left/right direction. Essentially flips the last axis.
     *
     * @param arr the input array
     * @returns xexpression with values flipped in left/right direction
     */
    template <class E>
    inline auto fliplr(E&& arr)
    {
        using CT = xclosure_t<E>;
        auto shape = arr.shape();
        return detail::make_xgenerator(detail::fliplr_impl<CT>(std::forward<E>(arr)),
                                       shape);
    }

    /**
     * @brief Flip xexpression in the up/down direction. Essentially flips the last axis.
     *
     * @param arr the input array
     * @returns xexpression with values flipped in up/down direction
     */
    template <class E>
    inline auto flipud(E&& arr)
    {
        using CT = xclosure_t<E>;
        auto shape = arr.shape();
        return detail::make_xgenerator(detail::flipud_impl<CT>(std::forward<E>(arr)),
                                       shape);
    }

    /**
     * @brief Extract lower triangular matrix from xexpression. The parameter k selects the
     *        offset of the diagonal.
     *
     * @param arr the input array
     * @param k the diagonal above which to zero elements. 0 (default) selects the main diagonal,
     *          k < 0 is below the main diagonal, k > 0 above.
     * @returns xexpression containing lower triangle from arr, 0 otherwise
     */
    template <class E>
    inline auto tril(E&& arr, int k = 0)
    {
        using CT = xclosure_t<E>;
        auto shape = arr.shape();
        return detail::make_xgenerator(detail::fn_impl<detail::trilu_fn<CT, std::greater_equal<long int>>>(
                                       detail::trilu_fn<CT, std::greater_equal<long int>>(std::forward<E>(arr), k, std::greater_equal<long int>())),
                                       shape);
    }

    /**
     * @brief Extract upper triangular matrix from xexpression. The parameter k selects the
     *        offset of the diagonal.
     *
     * @param arr the input array
     * @param k the diagonal below which to zero elements. 0 (default) selects the main diagonal,
     *          k < 0 is below the main diagonal, k > 0 above.
     * @returns xexpression containing lower triangle from arr, 0 otherwise
     */
    template <class E>
    inline auto triu(E&& arr, int k = 0)
    {
        using CT = xclosure_t<E>;
        auto shape = arr.shape();
        return detail::make_xgenerator(detail::fn_impl<detail::trilu_fn<CT, std::less_equal<long int>>>(
                                       detail::trilu_fn<CT, std::less_equal<long int>>(std::forward<E>(arr), k, std::less_equal<long int>())),
                                       shape);
    }
}
#endif
