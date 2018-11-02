/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_SORT_HPP
#define XTENSOR_SORT_HPP

#include <algorithm>
#include <utility>

#include "xarray.hpp"
#include "xeval.hpp"
#include "xslice.hpp"  // for xnone
#include "xmanipulation.hpp"
#include "xtensor.hpp"

namespace xt
{
    namespace detail
    {
        constexpr std::size_t normalize_axis(std::ptrdiff_t axis, std::size_t dim)
        {
            return axis >= 0 ? static_cast<std::size_t>(axis) : static_cast<std::size_t>(static_cast<std::ptrdiff_t>(dim) + axis);
        }

        template <class E, class F>
        inline void call_over_leading_axis(E& ev, F&& fct)
        {
            std::size_t n_iters = 1;
            std::ptrdiff_t secondary_stride;
            if (ev.layout() == layout_type::row_major)
            {
                n_iters = std::accumulate(ev.shape().begin(), ev.shape().end() - 1,
                                          std::size_t(1), std::multiplies<>());
                secondary_stride = static_cast<std::ptrdiff_t>(ev.strides()[ev.dimension() - 2]);
            }
            else
            {
                n_iters = std::accumulate(ev.shape().begin() + 1, ev.shape().end(),
                                          std::size_t(1), std::multiplies<>());
                secondary_stride = static_cast<std::ptrdiff_t>(ev.strides()[1]);
            }

            std::ptrdiff_t offset = 0;

            for (std::size_t i = 0; i < n_iters; ++i, offset += secondary_stride)
            {
                size_t adj_secondary_stride = (std::max)(secondary_stride, std::ptrdiff_t(1));
                fct(ev.data() + offset, ev.data() + offset + adj_secondary_stride);
            }
        }

        template <class E>
        inline std::size_t leading_axis(const E& e)
        {
            if (e.layout() == layout_type::row_major)
            {
                return e.dimension() - 1;
            }
            else if (e.layout() == layout_type::column_major)
            {
                return 0;
            }
            throw std::runtime_error("Layout not supported.");
        }

        // get permutations to transpose and reverse-transpose array
        inline std::pair<dynamic_shape<std::size_t>, dynamic_shape<std::size_t>>
        get_permutations(std::size_t dim, std::size_t ax, layout_type layout)
        {
            dynamic_shape<std::size_t> permutation(dim);
            std::iota(permutation.begin(), permutation.end(), std::size_t(0));
            permutation.erase(permutation.begin() + std::ptrdiff_t(ax));

            if (layout == layout_type::row_major)
            {
                permutation.push_back(ax);
            }
            else
            {
                permutation.insert(permutation.begin(), ax);
            }

            // TODO find a more clever way to get reverse permutation?
            dynamic_shape<std::size_t> reverse_permutation;
            for (std::size_t i = 0; i < dim; ++i)
            {
                auto it = std::find(permutation.begin(), permutation.end(), i);
                reverse_permutation.push_back(std::size_t(std::distance(permutation.begin(), it)));
            }

            return std::make_pair(std::move(permutation), std::move(reverse_permutation));
        }

        template <class E, class R, class F>
        inline auto run_lambda_over_axis(const E& e, R& res, std::size_t axis, F&& lambda)
        {
            if (axis != detail::leading_axis(res))
            {
                dynamic_shape<std::size_t> permutation, reverse_permutation;
                std::tie(permutation, reverse_permutation) = get_permutations(e.dimension(), axis, e.layout());

                res = transpose(e, permutation);
                detail::call_over_leading_axis(res, std::forward<F>(lambda));
                res = transpose(res, reverse_permutation);
            }
            else
            {
                res = e;
                detail::call_over_leading_axis(res, std::forward<F>(lambda));
            }
        }

        template <class VT>
        struct flatten_sort_result_type
        {
            using type = VT;
        };

        template <class VT, std::size_t N, layout_type L>
        struct flatten_sort_result_type<xtensor<VT, N, L>>
        {
            using type = xtensor<VT, 1, L>;
        };

        template <class VT, class S, layout_type L>
        struct flatten_sort_result_type<xtensor_fixed<VT, S, L>>
        {
            using type = xtensor_fixed<VT, xshape<fixed_compute_size<S>::value>, L>;
        };


        template <class E, class R = typename flatten_sort_result_type<E>::type>
        inline auto flat_sort_impl(const xexpression<E>& e)
        {
            const auto& de = e.derived_cast();
            R ev;
            ev.resize({de.size()});

            std::copy(de.cbegin(), de.cend(), ev.begin());
            std::sort(ev.begin(), ev.end());

            return ev;
        }
    }

    template <class E>
    inline auto sort(const xexpression<E>& e, placeholders::xtuph /*t*/)
    {
        return detail::flat_sort_impl(e);
    }

    namespace detail
    {
        template <class T>
        struct sort_eval_type
        {
            using type = typename T::temporary_type;
        };

        template <class T, std::size_t... I, layout_type L>
        struct sort_eval_type<xtensor_fixed<T, fixed_shape<I...>, L>>
        {
            using type = xtensor<T, sizeof...(I), L>;
        };
    }

    /**
     * Sort xexpression (optionally along axis)
     * The sort is performed using the ``std::sort`` functions.
     * A copy of the xexpression is created and returned.
     *
     * @param e xexpression to sort
     * @param axis axis along which sort is performed
     *
     * @return sorted array (copy)
     */
    template <class E>
    inline auto sort(const xexpression<E>& e, std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;

        const auto& de = e.derived_cast();

        if (de.dimension() == 1)
        {
            return detail::flat_sort_impl<std::decay_t<decltype(de)>, eval_type>(de);
        }

        std::size_t ax = detail::normalize_axis(axis, de.dimension());

        eval_type res;
        detail::run_lambda_over_axis(de, res, ax, [](auto begin, auto end) { std::sort(begin, end); });
        return res;
    }

    namespace detail
    {
        template <class VT, class T>
        struct rebind_value_type
        {
            using type = xarray<VT, xt::layout_type::dynamic>;
        };

        template <class VT, class EC, layout_type L>
        struct rebind_value_type<VT, xarray<EC, L>>
        {
            using type = xarray<VT, L>;
        };

        template <class VT, class EC, std::size_t N, layout_type L>
        struct rebind_value_type<VT, xtensor<EC, N, L>>
        {
            using type = xtensor<VT, N, L>;
        };

        template <class VT, class ET, class S, layout_type L>
        struct rebind_value_type<VT, xtensor_fixed<ET, S, L>>
        {
            using type = xtensor_fixed<VT, S, L>;
        };

        template <class VT, class T>
        struct flatten_rebind_value_type
        {
            using type = typename rebind_value_type<VT, T>::type;
        };

        template <class VT, class EC, std::size_t N, layout_type L>
        struct flatten_rebind_value_type<VT, xtensor<EC, N, L>>
        {
            using type = xtensor<VT, 1, L>;
        };

        template <class VT, class ET, class S, layout_type L>
        struct flatten_rebind_value_type<VT, xtensor_fixed<ET, S, L>>
        {
            using type = xtensor_fixed<VT, xshape<fixed_compute_size<S>::value>, L>;
        };

        template <class T>
        struct argsort_result_type
        {
            using type = typename rebind_value_type<typename T::temporary_type::size_type,
                                                    typename T::temporary_type>::type;
        };

        template <class T>
        struct linear_argsort_result_type
        {
            using type = typename flatten_rebind_value_type<typename T::temporary_type::size_type,
                                                            typename T::temporary_type>::type;
        };

        template <class Ed, class Ei>
        inline void argsort_over_leading_axis(const Ed& data, Ei& inds)
        {
            std::size_t n_iters = 1;
            std::ptrdiff_t data_secondary_stride;
            std::ptrdiff_t inds_secondary_stride;
            if (data.layout() == layout_type::row_major)
            {
                n_iters = std::accumulate(data.shape().begin(), data.shape().end() - 1,
                                          std::size_t(1), std::multiplies<>());
                data_secondary_stride = static_cast<std::ptrdiff_t>(data.strides()[data.dimension() - 2]);
                inds_secondary_stride = static_cast<std::ptrdiff_t>(inds.strides()[inds.dimension() - 2]);
            }
            else
            {
                n_iters = std::accumulate(data.shape().begin() + 1, data.shape().end(),
                                          std::size_t(1), std::multiplies<>());
                data_secondary_stride = static_cast<std::ptrdiff_t>(data.strides()[1]);
                inds_secondary_stride = static_cast<std::ptrdiff_t>(inds.strides()[1]);
            }

            std::ptrdiff_t data_offset = 0;
            std::ptrdiff_t inds_offset = 0;

            for (std::size_t i = 0; i < n_iters; ++i, data_offset += data_secondary_stride,
                    inds_offset += inds_secondary_stride)
            {
                auto comp = [&data, &data_offset](std::size_t x, std::size_t y) {
                    return (*(data.data() + data_offset + x) <
                            *(data.data() + data_offset + y));
                };
                std::iota(inds.data() + inds_offset, inds.data() + inds_offset + inds_secondary_stride, 0);
                std::sort(inds.data() + inds_offset, inds.data() + inds_offset + inds_secondary_stride, comp);
            }
        }

        template <class E, class R = typename detail::linear_argsort_result_type<E>::type>
        inline auto flatten_argsort_impl(const xexpression<E>& e)
        {
            using result_type = R;

            const auto& de = e.derived_cast();

            result_type result;
            result.resize({de.size()});
            auto comp = [&de](std::size_t x, std::size_t y) {
                return de[x] < de[y];
            };
            std::iota(result.begin(), result.end(), 0);
            std::sort(result.begin(), result.end(), comp);

            return result;
        }
    }

    template <class E>
    inline auto argsort(const xexpression<E>& e, placeholders::xtuph /*t*/)
    {
        return detail::flatten_argsort_impl(e);
    }

    /**
     * Argsort xexpression (optionally along axis)
     * Performs an indirect sort along the given axis. Returns an xarray
     * of indices of the same shape as e that index data along the given axis in
     * sorted order.
     *
     * @param e xexpression to argsort
     * @param axis axis along which argsort is performed
     *
     * @return argsorted index array
     */
    template <class E>
    inline auto argsort(const xexpression<E>& e, std::ptrdiff_t axis = -1)
    {
        using eval_type = typename detail::sort_eval_type<E>::type;
        using result_type = typename detail::argsort_result_type<eval_type>::type;

        const auto& de = e.derived_cast();

        std::size_t ax = detail::normalize_axis(axis, de.dimension());

        if (de.dimension() == 1)
        {
            return detail::flatten_argsort_impl<E, result_type>(e);
        }

        if (ax != detail::leading_axis(de))
        {
            dynamic_shape<std::size_t> permutation, reverse_permutation;
            std::tie(permutation, reverse_permutation) = detail::get_permutations(de.dimension(), ax, de.layout());

            eval_type ev = transpose(de, permutation);
            result_type res = result_type::from_shape(ev.shape());
            detail::argsort_over_leading_axis(ev, res);
            res = transpose(res, reverse_permutation);
            return res;
        }
        else
        {
            result_type res = result_type::from_shape(de.shape());
            detail::argsort_over_leading_axis(de, res);
            return res;
        }
    }

    namespace detail
    {
        template <class T>
        struct argfunc_result_type
        {
            using type = xarray<std::size_t>;
        };

        template <class T, std::size_t N>
        struct argfunc_result_type<xtensor<T, N>>
        {
            using type = xtensor<std::size_t, N - 1>;
        };

        template <class IT, class F>
        inline std::size_t cmp_idx(IT iter, IT end, std::ptrdiff_t inc, F&& cmp)
        {
            std::size_t idx = 0;
            auto min = *iter;
            iter += inc;
            for (std::size_t i = 1; iter < end; iter += inc, ++i)
            {
                if (cmp(*iter, min))
                {
                    min = *iter;
                    idx = i;
                }
            }
            return idx;
        }

        template <class E, class F>
        inline xtensor<std::size_t, 0> arg_func_impl(const E& e, F&& f)
        {
            return cmp_idx(e.template begin<XTENSOR_DEFAULT_LAYOUT>(),
                           e.template end<XTENSOR_DEFAULT_LAYOUT>(), 1,
                           std::forward<F>(f));
        }

        template <class E, class F>
        inline typename argfunc_result_type<E>::type
        arg_func_impl(const E& e, std::size_t axis, F&& cmp)
        {
            using eval_type = typename detail::sort_eval_type<E>::type;
            using value_type = typename E::value_type;
            using result_type = typename argfunc_result_type<E>::type;
            using result_shape_type = typename result_type::shape_type;

            if (e.dimension() == 1)
            {
                return arg_func_impl(e, std::forward<F>(cmp));
            }

            result_shape_type alt_shape;
            xt::resize_container(alt_shape, e.dimension() - 1);

            // Excluding copy, copy all of shape except for axis
            std::copy(e.shape().cbegin(), e.shape().cbegin() + std::ptrdiff_t(axis), alt_shape.begin());
            std::copy(e.shape().cbegin() + std::ptrdiff_t(axis) + 1, e.shape().cend(), alt_shape.begin() + std::ptrdiff_t(axis));

            result_type result = result_type::from_shape(std::move(alt_shape));
            auto result_iter = result.begin();

            auto arg_func_lambda = [&result_iter, &cmp](auto begin, auto end) {
                std::size_t idx = 0;
                value_type val = *begin;
                ++begin;
                for (std::size_t i = 1; begin != end; ++begin, ++i)
                {
                    if (cmp(*begin, val))
                    {
                        val = *begin;
                        idx = i;
                    }
                }
                *result_iter = idx;
                ++result_iter;
            };

            if (axis != detail::leading_axis(e))
            {
                dynamic_shape<std::size_t> permutation, reverse_permutation;
                std::tie(permutation, reverse_permutation) = detail::get_permutations(e.dimension(), axis, e.layout());

                // note: creating copy
                eval_type input = transpose(e, permutation);
                detail::call_over_leading_axis(input, arg_func_lambda);
                return result;
            }
            else
            {
                auto&& input = eval(e);
                detail::call_over_leading_axis(input, arg_func_lambda);
                return result;
            }
        }
    }

    template <class E>
    inline auto argmin(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl(ed, std::less<value_type>());
    }

    /**
     * Find position of minimal value in xexpression
     *
     * @param e input xexpression
     * @param axis select axis (or none)
     *
     * @return returns xarray with positions of minimal value
     */
    template <class E>
    inline auto argmin(const xexpression<E>& e, std::size_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl(ed, axis, std::less<value_type>());
    }

    template <class E>
    inline auto argmax(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl(ed, std::greater<value_type>());
    }

    /**
     * Find position of maximal value in xexpression
     *
     * @param e input xexpression
     * @param axis select axis (or none)
     *
     * @return returns xarray with positions of maximal value
     */
    template <class E>
    inline auto argmax(const xexpression<E>& e, std::size_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(e.derived_cast());
        return detail::arg_func_impl(ed, axis, std::greater<value_type>());
    }

    /**
     * Find unique elements of a xexpression. This returns a flattened xtensor with
     * sorted, unique elements from the original expression.
     *
     * @param e input xexpression (will be flattened)
     */
    template <class E>
    inline auto unique(const xexpression<E>& e)
    {
        auto sorted = sort(e, xnone());
        auto end = std::unique(sorted.begin(), sorted.end());
        std::size_t sz = static_cast<std::size_t>(std::distance(sorted.begin(), end));
        // TODO check if we can shrink the vector without reallocation
        using value_type = typename E::value_type;
        auto result = xtensor<value_type, 1>::from_shape({sz});
        std::copy(sorted.begin(), end, result.begin());
        return result;
    }

    /**
     * Find the set difference of two xexpressions. This returns a flattened xtensor with
     * the sorted, unique values in ar1 that are not in ar2.
     *
     * @param ar1 input xexpression (will be flattened)
     * @param ar2 input xexpression
     */
    template <class E1, class E2>
    inline auto setdiff1d(const xexpression<E1>& ar1, const xexpression<E2>& ar2)
    {
        using value_type = typename E1::value_type;

        auto unique1 = unique(ar1);
        auto unique2 = unique(ar2);

        auto tmp = xtensor<value_type, 1>::from_shape({unique1.size()});

        auto end = std::set_difference(
            unique1.begin(), unique1.end(),
            unique2.begin(), unique2.end(),
            tmp.begin()
        );

        std::size_t sz = static_cast<std::size_t>(std::distance(tmp.begin(), end));

        auto result = xtensor<value_type, 1>::from_shape({sz});

        std::copy(tmp.begin(), end, result.begin());

        return result;
    }
}

#endif
