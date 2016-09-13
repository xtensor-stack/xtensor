#ifndef NDINDEX_HPP
#define NDINDEX_HPP

#include <vector>
#include <algorithm>

namespace qs
{

    template <class C>
    struct array_inner_types;

    template <class S>
    using array_shape = std::vector<S>;

    template <class S>
    using array_strides = std::vector<S>;

    template <class S, class... Args>
    S data_offset(const array_strides<S>& strides, Args... args);

    template <class S>
    S data_size(const array_shape<S>& s);


    /********************************
     * data_offset implementation
     ********************************/

    namespace detail
    {
        template <class S>
        inline S data_offset_impl(const array_strides<S>& strides, S i)
        {
            return i * strides.back();
        }

        template <class S, class... Args>
        inline S data_offset_impl(const array_strides<S>& strides, S i, Args... args)
        {
            return i * strides[strides.size() - sizeof...(args) - 1] + data_offset_impl(strides, args...);
        }
    }

    template <class S, class... Args>
    inline S data_offset(const array_strides<S>& strides, Args... args)
    {
        return detail::data_offset_impl(strides, static_cast<S>(args)...);
    }

    template <class S>
    inline S data_size(const array_shape<S>& s)
    {
        return std::accumulate(s.begin(), s.end(), S(1), std::multiplies<S>());
    }

}

#endif

