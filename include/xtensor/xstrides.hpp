/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_STRIDES_HPP
#define XTENSOR_STRIDES_HPP

#include <cstddef>
#include <functional>
#include <numeric>

#include "xexception.hpp"
#include "xtensor_forward.hpp"

namespace xt
{

    template <class shape_type>
    std::size_t compute_size(const shape_type& shape) noexcept;

    /***************
     * data offset *
     ***************/

    template <class size_type, class S>
    size_type data_offset(const S& strides) noexcept;

    template <class size_type, class S, class Arg, class... Args>
    size_type data_offset(const S& strides, Arg arg, Args... args) noexcept;

    template <class size_type, class S, class... Args>
    size_type unchecked_data_offset(const S& strides, Args... args) noexcept;

    template <class size_type, class S, class It>
    size_type element_offset(const S& strides, It first, It last) noexcept;

    /*******************
     * strides builder *
     *******************/

    template <class shape_type, class strides_type>
    std::size_t compute_strides(const shape_type& shape, layout_type l, strides_type& strides);

    template <class shape_type, class strides_type, class backstrides_type>
    std::size_t compute_strides(const shape_type& shape, layout_type l,
                                strides_type& strides, backstrides_type& backstrides);

    template <class shape_type, class strides_type>
    void adapt_strides(const shape_type& shape, strides_type& strides) noexcept;

    template <class shape_type, class strides_type, class backstrides_type>
    void adapt_strides(const shape_type& shape, strides_type& strides,
                       backstrides_type& backstrides) noexcept;

    /*****************
     * unravel_index *
     *****************/

    template <class S>
    S unravel_from_strides(typename S::value_type index, const S& strides, layout_type l);

    template <class S>
    S unravel_index(typename S::value_type index, const S& shape, layout_type l);

    /***********************
     * broadcast functions *
     ***********************/

    template <class S1, class S2>
    bool broadcast_shape(const S1& input, S2& output);

    template <class S1, class S2>
    bool broadcastable(const S1& s1, S2& s2);

    /*************************
     * check strides overlap *
     *************************/

    template <layout_type L>
    struct check_strides_overlap;

    /********************************************
     * utility functions for strided containers *
     ********************************************/

    template <class C, class It>
    It strided_data_end(const C& c, It end, layout_type l)
    {
        using difference_type = typename std::iterator_traits<It>::difference_type;
        if (c.dimension() == 0)
        {
            return end;
        }
        else
        {
            auto leading_stride = (l == layout_type::row_major ? c.strides().back() : c.strides().front());
            return end + difference_type(leading_stride - 1);
        }
    }

    /******************
     * Implementation *
     ******************/

    template <class shape_type>
    inline std::size_t compute_size(const shape_type& shape) noexcept
    {
        using size_type = std::decay_t<typename shape_type::value_type>;
        return static_cast<std::size_t>(std::accumulate(shape.cbegin(), shape.cend(), size_type(1), std::multiplies<size_type>()));
    }

    namespace detail
    {
        template <class size_type, std::size_t dim, class S>
        inline size_type raw_data_offset(const S&) noexcept
        {
            return 0;
        }

        template <class size_type, std::size_t dim, class S, class Arg, class... Args>
        inline size_type raw_data_offset(const S& strides, Arg arg, Args... args) noexcept
        {
            using strides_type = decltype(strides[0]);
            return strides_type(arg) * strides[dim] + raw_data_offset<size_type, dim + 1>(strides, args...);
        }
    }

    template <class size_type, class S>
    inline size_type data_offset(const S&) noexcept
    {
        return 0;
    }

    template <class size_type, class S, class Arg, class... Args>
    inline size_type data_offset(const S& strides, Arg arg, Args... args) noexcept
    {
        constexpr std::size_t nargs = sizeof...(Args) + 1;
        if (nargs == strides.size())
        {
            // Correct number of arguments: iterate
            return detail::raw_data_offset<size_type, 0>(strides, arg, args...);
        }
        else if (nargs > strides.size())
        {
            // Too many arguments: drop the first
            return data_offset<size_type, S>(strides, args...);
        }
        else
        {
            // Too few arguments: right to left scalar product
            auto view = strides.cend() - nargs;
            return detail::raw_data_offset<size_type, 0>(view, arg, args...);
        }
    }

    template <class size_type, class S, class... Args>
    inline size_type unchecked_data_offset(const S& strides, Args... args) noexcept
    {
        return detail::raw_data_offset<size_type, 0>(strides.cbegin(), args...);
    }

    template <class size_type, class S, class It>
    inline size_type element_offset(const S& strides, It first, It last) noexcept
    {
        using difference_type = typename std::iterator_traits<It>::difference_type;
        auto size = static_cast<difference_type>((std::min)(static_cast<typename S::size_type>(std::distance(first, last)), strides.size()));
        return std::inner_product(last - size, last, strides.cend() - size, size_type(0));
    }

    namespace detail
    {
        template <class shape_type, class strides_type, class bs_ptr>
        inline void adapt_strides(const shape_type& shape, strides_type& strides,
                                  bs_ptr backstrides, typename strides_type::size_type i) noexcept
        {
            if (shape[i] == 1)
            {
                strides[i] = 0;
            }
            (*backstrides)[i] = strides[i] * (shape[i] - 1);
        }

        template <class shape_type, class strides_type>
        inline void adapt_strides(const shape_type& shape, strides_type& strides,
                                  std::nullptr_t, typename strides_type::size_type i) noexcept
        {
            if (shape[i] == 1)
            {
                strides[i] = 0;
            }
        }

        template <class shape_type, class strides_type, class bs_ptr>
        inline std::size_t compute_strides(const shape_type& shape, layout_type l,
                                           strides_type& strides, bs_ptr bs)
        {
            std::size_t data_size = 1;
            if (l == layout_type::row_major)
            {
                for (std::size_t i = strides.size(); i != 0; --i)
                {
                    strides[i - 1] = data_size;
                    data_size = strides[i - 1] * shape[i - 1];
                    adapt_strides(shape, strides, bs, i - 1);
                }
            }
            else
            {
                for (std::size_t i = 0; i < strides.size(); ++i)
                {
                    strides[i] = data_size;
                    data_size = strides[i] * shape[i];
                    adapt_strides(shape, strides, bs, i);
                }
            }
            return data_size;
        }
    }

    template <class shape_type, class strides_type>
    inline std::size_t compute_strides(const shape_type& shape, layout_type l, strides_type& strides)
    {
        return detail::compute_strides(shape, l, strides, nullptr);
    }

    template <class shape_type, class strides_type, class backstrides_type>
    inline std::size_t compute_strides(const shape_type& shape, layout_type l,
                                       strides_type& strides,
                                       backstrides_type& backstrides)
    {
        return detail::compute_strides(shape, l, strides, &backstrides);
    }

    template <class shape_type, class strides_type>
    inline bool do_strides_match(const shape_type& shape, const strides_type& strides, layout_type l)
    {
        using shape_value_type = typename shape_type::value_type;
        shape_value_type data_size = 1;
        if (l == layout_type::row_major)
        {
            for (std::size_t i = strides.size(); i != 0; --i)
            {
                if ((shape[i - 1] == 1 && strides[i - 1] != 0) || (shape[i - 1] != 1 && strides[i - 1] != data_size))
                {
                    return false;
                }
                data_size *= shape[i - 1];
            }
            return true;
        }
        else if (l == layout_type::column_major)
        {
            for (std::size_t i = 0; i < strides.size(); ++i)
            {
                if ((shape[i] != 1 && strides[i] != data_size) || (shape[i] == 1 && strides[i] != 0))
                {
                    return false;
                }
                data_size *= shape[i];
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    template <class shape_type, class strides_type>
    inline void adapt_strides(const shape_type& shape, strides_type& strides) noexcept
    {
        for (typename shape_type::size_type i = 0; i < shape.size(); ++i)
        {
            detail::adapt_strides(shape, strides, nullptr, i);
        }
    }

    template <class shape_type, class strides_type, class backstrides_type>
    inline void adapt_strides(const shape_type& shape, strides_type& strides,
                              backstrides_type& backstrides) noexcept
    {
        for (typename shape_type::size_type i = 0; i < shape.size(); ++i)
        {
            detail::adapt_strides(shape, strides, &backstrides, i);
        }
    }

    namespace detail
    {
        template <class S>
        inline S unravel_noexcept(typename S::value_type idx, const S& strides, layout_type l) noexcept
        {
            using value_type = typename S::value_type;
            using size_type = typename S::size_type;
            S result = xtl::make_sequence<S>(strides.size(), 0);
            if (l == layout_type::row_major)
            {
                for (size_type i = 0; i < strides.size(); ++i)
                {
                    value_type str = strides[i];
                    value_type quot = str != 0 ? idx / str : 0;
                    idx = str != 0 ? idx % str : idx;
                    result[i] = quot;
                }
            }
            else
            {
                for (size_type i = strides.size(); i != 0; --i)
                {
                    value_type str = strides[i - 1];
                    value_type quot = str != 0 ? idx / str : 0;
                    idx = str != 0 ? idx % str : idx;
                    result[i - 1] = quot;
                }
            }
            return result;
        }
    }

    template <class S>
    inline S unravel_from_strides(typename S::value_type index, const S& strides, layout_type l)
    {
        if (l != layout_type::row_major && l != layout_type::column_major)
        {
            throw std::runtime_error("unravel_index: dynamic layout not supported");
        }
        return detail::unravel_noexcept(index, strides, l);
    }

    template <class S>
    inline S unravel_index(typename S::value_type index, const S& shape, layout_type l)
    {
        S strides = xtl::make_sequence<S>(shape.size(), 0);
        compute_strides(shape, l, strides);
        return unravel_from_strides(index, strides, l);
    }

    template <class S1, class S2>
    inline bool broadcast_shape(const S1& input, S2& output)
    {
        bool trivial_broadcast = (input.size() == output.size());
        // Indices are faster than reverse iterators
        using value_type = typename S2::value_type;
        auto output_index = output.size();
        auto input_index = input.size();
        if (output_index < input_index)
        {
            throw_broadcast_error(output, input);
        }
        for (; input_index != 0; --input_index, --output_index)
        {
            // First case: output = (0, 0, ...., 0)
            // output is a new shape that has not been through
            // the broadcast process yet; broadcast is trivial
            if (output[output_index - 1] == 0)
            {
                output[output_index - 1] = static_cast<value_type>(input[input_index - 1]);
            }
            // Second case: output has been initialized to 1. Broacast is trivial
            // only if input is 1 to.
            else if (output[output_index - 1] == 1)
            {
                output[output_index - 1] = static_cast<value_type>(input[input_index - 1]);
                trivial_broadcast = trivial_broadcast && (input[input_index - 1] == 1);
            }
            // Third case: output has been initialized to something different from 1.
            // if input is 1, then the broadcast is not trivial
            else if (input[input_index - 1] == 1)
            {
                trivial_broadcast = false;
            }
            // Last case: input and output must have the same value, else
            // shape are not compatible and an exception is thrown
            else if (static_cast<value_type>(input[input_index - 1]) != output[output_index - 1])
            {
                throw_broadcast_error(output, input);
            }
        }
        return trivial_broadcast;
    }

    template <class S1, class S2>
    inline bool broadcastable(const S1& src_shape, const S2& dst_shape)
    {
        auto src_iter = src_shape.crbegin();
        auto dst_iter = dst_shape.crbegin();
        bool res = dst_shape.size() >= src_shape.size();
        for (; src_iter != src_shape.crend() && res; ++src_iter, ++dst_iter)
        {
            res = (static_cast<std::size_t>(*src_iter) == static_cast<std::size_t>(*dst_iter)) || (*src_iter == 1);
        }
        return res;
    }

    template <>
    struct check_strides_overlap<layout_type::row_major>
    {
        template <class S1, class S2>
        static std::size_t get(const S1& s1, const S2& s2)
        {
            using value_type = typename S1::value_type;
            // Indices are faster than reverse iterators
            auto s1_index = s1.size();
            auto s2_index = s2.size();

            for (; s2_index != 0; --s1_index, --s2_index)
            {
                if (static_cast<value_type>(s1[s1_index - 1]) != static_cast<value_type>(s2[s2_index - 1]))
                {
                    break;
                }
            }
            return s1_index;
        }
    };

    template <>
    struct check_strides_overlap<layout_type::column_major>
    {
        template <class S1, class S2>
        static std::size_t get(const S1& s1, const S2& s2)
        {
            // Indices are faster than reverse iterators
            using size_type = typename S1::size_type;
            using value_type = typename S1::value_type;
            size_type index = 0;

            // This check is necessary as column major "broadcasting" is still
            // peformed in a row major fashion
            if (s1.size() != s2.size())
                return 0;

            auto size = s2.size();

            for (; index < size; ++index)
            {
                if (static_cast<value_type>(s1[index]) != static_cast<value_type>(s2[index]))
                {
                    break;
                }
            }
            return index;
        }
    };
}

#endif
