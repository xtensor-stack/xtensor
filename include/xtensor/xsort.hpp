/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_SORT_HPP
#define XTENSOR_SORT_HPP

#include <iterator>
#include <algorithm>

#include "xarray.hpp"
#include "xslice.hpp"  // for xnone

namespace xt
{
    namespace detail
    {

        template <class T>
        class strided_iterator :
            public std::iterator<std::random_access_iterator_tag, T, ptrdiff_t, const T*, T&>
        {
        public:
            using value_type = T;
            using self_type = strided_iterator<value_type>;
            using difference_type = ptrdiff_t;
            using pointer = value_type*;
            using const_pointer = const value_type*;
            using reference = value_type&;
            using const_reference = const value_type&;

            strided_iterator<value_type>(value_type* ptr, std::size_t stride)
                : m_ptr(ptr), m_stride(stride)
            {
            }

            self_type& operator++() {
                m_ptr += m_stride;
                return *this;
            }

            self_type operator++(int) {
                self_type retval = *this;
                ++(*this);
                return retval;
            }

            self_type& operator--() {
                m_ptr -= m_stride;
                return *this;
            }

            self_type operator--(int) {
                self_type retval = *this;
                --(*this);
                return retval;
            }

            self_type& operator+=(const ptrdiff_t& inc)
            {
                m_ptr += inc * m_stride;
                return *this;
            }
            self_type& operator-=(const ptrdiff_t& inc)
            {
                m_ptr -= inc * m_stride;
                return *this;
            }

            self_type operator+(const ptrdiff_t& inc)
            {
                self_type retval = *this;
                retval += inc;
                return retval;
            }
            self_type operator-(const ptrdiff_t& inc)
            {
                self_type retval = *this;
                retval -= inc;
                return retval;
            }

            friend int operator-(const self_type& lhs, const self_type& rhs)
            {
                // TODO benchmark against having std::size_t m_pos member that keeps track of position
                return (lhs.m_ptr - rhs.m_ptr) / lhs.m_stride;
            }

            bool operator==(self_type other) const {
                return m_ptr == other.m_ptr;
            }
            bool operator!=(self_type other) const {
                return !(*this == other);
            }

            reference operator*() const {
                return *m_ptr;
            }
            pointer operator->() const {
                return m_ptr;
            }

            friend bool operator<(const self_type& lhs, const self_type& rhs)
            {
                return lhs.m_ptr < rhs.m_ptr;
            }
            friend bool operator>(const self_type& lhs, const self_type& rhs)
            {
                return lhs.m_ptr > rhs.m_ptr;
            }
            friend bool operator<=(const self_type& lhs, const self_type& rhs)
            {
                return lhs.m_ptr <= rhs.m_ptr;
            }
            friend bool operator>=(const self_type& lhs, const self_type& rhs)
            {
                return lhs.m_ptr >= rhs.m_ptr;
            }

        private:
            std::size_t m_stride;
            T* m_ptr;
        };

        template <class T>
        class stride_iterator_maker
        {
        public:
            stride_iterator_maker(T* ptr, std::size_t stride, std::size_t size)
                : m_ptr(ptr), m_orig_ptr(ptr), m_stride(stride), m_size(size)
            {
            }

            template <class A>
            stride_iterator_maker(A& a, std::size_t axis)
                : m_ptr(a.raw_data()),
                  m_orig_ptr(a.raw_data()),
                  m_stride(a.strides()[axis]),
                  m_size(a.strides()[axis] * a.shape()[axis])
            {
            }

            template <class A>
            stride_iterator_maker(A& a)
                : m_ptr(a.raw_data()),
                  m_orig_ptr(a.raw_data()),
                  m_stride(1),
                  m_size(a.size())
            {
            }

            strided_iterator<T> begin()
            {
                return strided_iterator<T>(m_ptr, m_stride);
            }

            strided_iterator<T> end()
            {
                return strided_iterator<T>(m_ptr + m_size, m_stride);
            }

            void set_ptr_offset(ptrdiff_t inc)
            {
                m_ptr = m_orig_ptr + inc;
            }

        private:
            std::size_t m_size;
            std::size_t m_stride;
            T* m_ptr;
            T* m_orig_ptr;
        };
    }

    template <class E>
    auto sort(const xexpression<E>& e)
    {
        const auto& de = e.derived_cast();
        return sort(de, de.dimension() - 1);
    }

    template <class E>
    auto sort(const xexpression<E>& e, placeholders::xtuph /*t*/)
    {
        using value_type = typename E::value_type;
        const auto de = e.derived_cast();
        xtensor<value_type, 1> ev;
        ev.reshape({de.size()});

        std::copy(de.begin(), de.end(), ev.begin());
        std::sort(ev.begin(), ev.end());

        return ev;
    }

    template <class E, class F>
    void call_over_axis(E& ev, F&& fct, std::size_t axis)
    {
        using value_type = typename E::value_type;


        // fast track for constant secondary stride
        if (E::static_layout == layout_type::row_major && (axis == 0 || ev.dimension() - 1 == axis))
        {
            std::size_t n_iters = 1;
            for (std::size_t i = 0; i < ev.dimension(); ++i)
            {
                if (i != axis)
                {
                    n_iters *= ev.shape()[i];
                }
            }

            ptrdiff_t secondary_stride = 0;

            if (axis == 0)
            {
                secondary_stride = 1;
            }
            else
            {
                secondary_stride = ev.strides()[axis - 1];
            }

            detail::stride_iterator_maker<value_type> smk = detail::stride_iterator_maker<value_type>(ev, axis);

            for (std::size_t i = 0, offset = secondary_stride; i < n_iters; ++i, offset += secondary_stride)
            {
                fct(smk.begin(), smk.end());
                smk.set_ptr_offset(offset);
            }
            return;
        }

        auto iter_shape = ev.shape();
        iter_shape.erase(iter_shape.begin() + axis);

        std::vector<std::size_t> iter_strides = ev.strides();
        iter_strides.erase(iter_strides.begin() + axis);

        xindex temp_idx(iter_shape.size());
        auto next_idx = [&iter_shape, &iter_strides, &temp_idx]()
        {
            ptrdiff_t offset = 0;
            for (int i = int(iter_shape.size() - 1); i >= 0; --i)
            {
                if (temp_idx[(std::size_t) i] >= iter_shape[(std::size_t) i] - 1)
                {
                    temp_idx[(std::size_t) i] = 0;
                }
                else
                {
                    temp_idx[(std::size_t) i]++;
                    break;
                }
            }
            return std::inner_product(temp_idx.begin(), temp_idx.end(),
                                      iter_strides.begin(), ptrdiff_t(0));
        };

        detail::stride_iterator_maker<value_type> smk = detail::stride_iterator_maker<value_type>(ev, axis);

        fct(smk.begin(), smk.end());
        ptrdiff_t offset = next_idx();
        while (offset != 0)
        {
            smk.set_ptr_offset(offset);
            fct(smk.begin(), smk.end());
            offset = next_idx();
        }
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
    auto sort(const xexpression<E>& e, std::size_t axis)
    {
        using eval_type = typename E::temporary_type;
        using value_type = typename E::value_type;
        eval_type ev = e.derived_cast();
        call_over_axis(ev, [](auto begin, auto end){ std::sort(begin, end); }, axis);
        return ev;
    }

    namespace detail
    {
        template <class IT, class F>
        inline std::size_t cmp_idx(IT iter, IT end, ptrdiff_t inc, F&& cmp)
        {
            std::size_t idx = 0;
            double min = *iter;
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

        // TODO allow returning of xtensor<size_t, dim - 1> ...
        template <class A, class F>
        xarray<std::size_t> arg_func_impl(A& a, F&& f)
        {
            return cmp_idx(a.raw_data(), a.raw_data() + a.data().size(), 1, std::forward<F>(f));
        }

        template <class A, class F>
        xarray<std::size_t> arg_func_impl(A& a, std::size_t axis, F&& cmp)
        {
            // TOOD numpy transposes the axis so, that the axis over which the argmin is taken is at the end
            //      this might improve performance drastically for large matrices because the relevant memory is
            //      continous together and not scattered.
            using value_type = typename A::value_type;

            std::vector<std::size_t> new_shape = a.shape();
            new_shape.erase(new_shape.begin() + axis);

            xt::xarray<std::size_t> result = xt::xarray<std::size_t>::from_shape(new_shape);

            auto result_iter = result.begin();
            call_over_axis(a, [&result_iter, &cmp](auto begin, auto end) {
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
            }, axis);
            return result;
        }
    }

    template <class E>
    auto argmin(xexpression<E>& a)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(a.derived_cast());
        return detail::arg_func_impl(ed, std::less<value_type>());
    }

    /**
     * Find position of minimal value in xexpression
     *
     * @param a xexpression to compute argmin on
     * @param axis select axis (or none)
     *
     * @return returns xarray with positions of minimal value
     */
    template <class E>
    auto argmin(xexpression<E>& a, std::size_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(a.derived_cast());
        return detail::arg_func_impl(ed, axis, std::less<value_type>());
    }

    template <class E>
    auto argmax(xexpression<E>& a)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(a.derived_cast());
        return detail::arg_func_impl(ed, std::greater<value_type>());
    }

    /**
     * Find position of maximal value in xexpression
     *
     * @param a xexpression to compute argmin on
     * @param axis select axis (or none)
     *
     * @return returns xarray with positions of minimal value
     */
    template <class E>
    auto argmax(xexpression<E>& a, std::size_t axis)
    {
        using value_type = typename E::value_type;
        auto&& ed = eval(a.derived_cast());
        return detail::arg_func_impl(ed, axis, std::greater<value_type>());
    }

}

#endif