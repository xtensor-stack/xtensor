#ifndef XINDEX_HPP
#define XINDEX_HPP

#include <vector>
#include <numeric>
#include <functional>

namespace qs
{

    template <class C>
    struct array_inner_types;

    template <class S>
    using xshape = std::vector<S>;

    template <class S>
    using xstrides = std::vector<S>;

    template <class S, class... Args>
    S data_offset(const xstrides<S>& strides, Args... args);

    template <class S>
    S data_size(const xshape<S>& s);


    /********************************
     * data_offset implementation
     ********************************/

    namespace detail
    {
        template <class S>
        inline S data_offset_impl(const xstrides<S>& strides, S i)
        {
            return i * strides.back();
        }

        template <class S, class... Args>
        inline S data_offset_impl(const xstrides<S>& strides, S i, Args... args)
        {
            return i * strides[strides.size() - sizeof...(args) - 1] + data_offset_impl(strides, args...);
        }
    }

    template <class S, class... Args>
    inline S data_offset(const xstrides<S>& strides, Args... args)
    {
        return detail::data_offset_impl(strides, static_cast<S>(args)...);
    }

    template <class S>
    inline S data_size(const xshape<S>& s)
    {
        return std::accumulate(s.begin(), s.end(), S(1), std::multiplies<S>());
    }

}

#endif

