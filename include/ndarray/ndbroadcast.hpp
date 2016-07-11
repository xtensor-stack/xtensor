#ifndef NDBROADCAST_HPP
#define NDBROADCAST_HPP

#include <array>
#include <stdexcept>
#include "ndindex.hpp"

namespace ndarray
{

    template <class S, size_t N>
    S broadcast_dim(std::array<const array_shape<S>*, N>& shape);

    template <class S>
    bool broadcast_shape(const array_shape<S>& input, array_shape<S>& output);

    template <class S, size_t N>
    bool broadcast_shape(const std::array<const array_shape<S>*, N>& input, array_shape<S>& output);


    /****************************************
     * broadcast functions implementation
     ****************************************/

    template <class S, size_t N>
    inline S broadcast_dim(const std::array<const array_shape<S>*, N>& shape)
    {
        using shape_type = array_shape<S>;
        S ndim = std::accumulate(shape.begin(), shape.end(), S(0),
                [](S res, const shape_type* s) { return std::max(s->size(), res); });
        return ndim;
    }

    template <class S>
    inline bool broadcast_shape(const array_shape<S>& input, array_shape<S>& output)
    {
        size_t size = output.size();
        bool trivial_broadcast = (input.size() == output.size());
        auto output_iter = output.rbegin();
        for(auto input_iter = input.rbegin(); input_iter != input.rend();
            ++input_iter, ++output_iter)
        {
            if(*output_iter == 1)
            {
                *output_iter = *input_iter;
            }
            else if((*input_iter != 1) && (*output_iter != *input_iter))
            {
                throw std::runtime_error("broadcast error : incompatible dimension of inputs");
            }
            trivial_broadcast = trivial_broadcast && (*output_iter == *input_iter);
        }
        return trivial_broadcast;
    }

    template <class S, size_t N>
    inline bool broadcast_shape(const std::array<const array_shape<S>*, N>& input, array_shape<S>& output)
    {
        bool trivial_broadcast = true;
        for(size_t i = 0; i < N; ++i)
        {
            bool itrivial_broadcast = broadcast_shape(*input[i], output);
            trivial_broadcast = trivial_broadcast && itrivial_broadcast;
        }
        return trivial_broadcast;
    }

}

#endif

