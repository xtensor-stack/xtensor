/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_PAD_HPP
#define XTENSOR_PAD_HPP

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../views/xstrided_view.hpp"
#include "../views/xview.hpp"

using namespace xt::placeholders;  // to enable _ syntax

namespace xt
{
    /**
     * @brief Defines different algorithms to be used in ``xt::pad``:
     * - ``constant``: Pads with a constant value.
     * - ``symmetric``: Pads with the reflection of the vector mirrored along the edge of the array.
     * - ``reflect``: Pads with the reflection of the vector mirrored on the first and last values
     *   of the vector along each axis.
     * - ``wrap``: Pads with the wrap of the vector along the axis. The first values are used to pad
     *   the end and the end values are used to pad the beginning.
     * - ``periodic`` : ``== wrap`` (pads with periodic repetitions of the vector).
     *
     * OpenCV to xtensor:
     * - ``BORDER_CONSTANT == constant``
     * - ``BORDER_REFLECT == symmetric``
     * - ``BORDER_REFLECT_101 == reflect``
     * - ``BORDER_WRAP == wrap``
     */
    enum class pad_mode
    {
        constant,
        symmetric,
        reflect,
        wrap,
        periodic,
        edge
    };

    namespace detail
    {
        template <class S>
        inline std::size_t checked_count(S value, const char* error)
        {
            if constexpr (std::is_signed_v<S>)
            {
                if (value < S(0))
                {
                    XTENSOR_THROW(std::runtime_error, error);
                }
            }
            return static_cast<std::size_t>(value);
        }

        // pad_width is indexed by axis, and the modes other than constant read the
        // padding back from the values of the array: without these checks, invalid
        // widths read past the end of pad_width or return uninitialized values.
        template <class S, class T>
        inline void check_pad_width(const std::vector<std::vector<S>>& pad_width, const T& shape, pad_mode mode)
        {
            if (pad_width.size() != shape.size())
            {
                XTENSOR_THROW(std::runtime_error, "pad: pad_width must have one {before, after} pair per axis.");
            }
            for (std::size_t axis = 0; axis < shape.size(); ++axis)
            {
                if (pad_width[axis].size() != 2)
                {
                    XTENSOR_THROW(
                        std::runtime_error,
                        "pad: pad_width must have one {before, after} pair per axis."
                    );
                }
                const char* negative = "pad: pad widths must not be negative.";
                const std::size_t before = checked_count(pad_width[axis][0], negative);
                const std::size_t after = checked_count(pad_width[axis][1], negative);
                const std::size_t size = static_cast<std::size_t>(shape[axis]);
                std::size_t max_width = (std::numeric_limits<std::size_t>::max)();
                switch (mode)
                {
                    case pad_mode::constant:
                        break;
                    case pad_mode::symmetric:
                    case pad_mode::wrap:
                    case pad_mode::periodic:
                        max_width = size;
                        break;
                    case pad_mode::reflect:
                        max_width = size == 0 ? 0 : size - 1;
                        break;
                    case pad_mode::edge:
                        max_width = size == 0 ? 0 : max_width;
                        break;
                }
                if (before > max_width || after > max_width)
                {
                    XTENSOR_THROW(
                        std::runtime_error,
                        "pad: pad width too large for the size of the axis in this mode."
                    );
                }
            }
        }
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis:
     * `{{before_1, after_1}, ..., {before_N, after_N}}`.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E, class S = typename std::decay_t<E>::size_type, class V = typename std::decay_t<E>::value_type>
    inline auto
    pad(E&& e,
        const std::vector<std::vector<S>>& pad_width,
        pad_mode mode = pad_mode::constant,
        V constant_value = 0)
    {
        detail::check_pad_width(pad_width, e.shape(), mode);

        using size_type = typename std::decay_t<E>::size_type;
        using return_type = temporary_type_t<E>;

        // place the original array in the center

        auto new_shape = e.shape();
        xt::xstrided_slice_vector sv;
        sv.reserve(e.shape().size());
        for (size_type axis = 0; axis < e.shape().size(); ++axis)
        {
            size_type nb = static_cast<size_type>(pad_width[axis][0]);
            size_type ne = static_cast<size_type>(pad_width[axis][1]);
            size_type ns = nb + e.shape(axis) + ne;
            new_shape[axis] = ns;
            sv.push_back(xt::range(nb, nb + e.shape(axis)));
        }

        if (mode == pad_mode::constant)
        {
            return_type out(new_shape, constant_value);
            xt::strided_view(out, sv) = e;
            return out;
        }

        return_type out(new_shape);
        xt::strided_view(out, sv) = e;

        // construct padded regions based on original image

        xt::xstrided_slice_vector svs(e.shape().size(), xt::all());
        xt::xstrided_slice_vector svt(e.shape().size(), xt::all());

        for (size_type axis = 0; axis < e.shape().size(); ++axis)
        {
            size_type nb = static_cast<size_type>(pad_width[axis][0]);
            size_type ne = static_cast<size_type>(pad_width[axis][1]);

            if (nb > static_cast<size_type>(0))
            {
                svt[axis] = xt::range(0, nb);

                if (mode == pad_mode::wrap || mode == pad_mode::periodic)
                {
                    svs[axis] = xt::range(e.shape(axis), nb + e.shape(axis));
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::symmetric)
                {
                    svs[axis] = xt::range(2 * nb - 1, nb - 1, -1);
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::reflect)
                {
                    svs[axis] = xt::range(2 * nb, nb, -1);
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::edge)
                {
                    svs[axis] = xt::range(nb, nb + 1);
                    xt::strided_view(out, svt) = xt::broadcast(
                        xt::strided_view(out, svs),
                        xt::strided_view(out, svt).shape()
                    );
                }
            }

            if (ne > static_cast<size_type>(0))
            {
                svt[axis] = xt::range(out.shape(axis) - ne, out.shape(axis));

                if (mode == pad_mode::wrap || mode == pad_mode::periodic)
                {
                    svs[axis] = xt::range(nb, nb + ne);
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::symmetric)
                {
                    if (ne == nb + e.shape(axis))
                    {
                        svs[axis] = xt::range(nb + e.shape(axis) - 1, _, -1);
                    }
                    else
                    {
                        svs[axis] = xt::range(nb + e.shape(axis) - 1, nb + e.shape(axis) - ne - 1, -1);
                    }
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::reflect)
                {
                    if (ne == nb + e.shape(axis) - 1)
                    {
                        svs[axis] = xt::range(nb + e.shape(axis) - 2, _, -1);
                    }
                    else
                    {
                        svs[axis] = xt::range(nb + e.shape(axis) - 2, nb + e.shape(axis) - ne - 2, -1);
                    }
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                }
                else if (mode == pad_mode::edge)
                {
                    svs[axis] = xt::range(out.shape(axis) - ne - 1, out.shape(axis) - ne);
                    xt::strided_view(out, svt) = xt::broadcast(
                        xt::strided_view(out, svs),
                        xt::strided_view(out, svt).shape()
                    );
                }
            }

            svs[axis] = xt::all();
            svt[axis] = xt::all();
        }

        return out;
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis:
     * `{before, after}`.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E, class S = typename std::decay_t<E>::size_type, class V = typename std::decay_t<E>::value_type>
    inline auto
    pad(E&& e, const std::vector<S>& pad_width, pad_mode mode = pad_mode::constant, V constant_value = 0)
    {
        std::vector<std::vector<S>> pw(e.shape().size(), pad_width);

        return pad(e, pw, mode, constant_value);
    }

    /**
     * @brief Pad an array.
     *
     * @param e The array.
     * @param pad_width Number of values padded to the edges of each axis.
     * @param mode The type of algorithm to use. [default: `xt::pad_mode::constant`].
     * @param constant_value The value to set the padded values for each axis
     * (used in `xt::pad_mode::constant`).
     * @return The padded array.
     */
    template <class E, class S = typename std::decay_t<E>::size_type, class V = typename std::decay_t<E>::value_type>
    inline auto pad(E&& e, S pad_width, pad_mode mode = pad_mode::constant, V constant_value = 0)
    {
        std::vector<std::vector<S>> pw(e.shape().size(), {pad_width, pad_width});

        return pad(e, pw, mode, constant_value);
    }

    namespace detail
    {

        template <class E, class S>
        inline auto tile(E&& e, const S& reps)
        {
            using size_type = typename std::decay_t<E>::size_type;

            using return_type = temporary_type_t<E>;

            const size_type dim = e.dimension();
            if (reps.size() != dim)
            {
                XTENSOR_THROW(std::runtime_error, "tile: reps must have one value per axis.");
            }

            using new_shape_type = typename return_type::shape_type;
            auto new_shape = xtl::make_sequence<new_shape_type>(dim);

            xt::xstrided_slice_vector sv(dim);

            for (size_type axis = 0; axis < dim; ++axis)
            {
                const size_type n = checked_count(reps[axis], "tile: reps must not be negative.");
                new_shape[axis] = e.shape(axis) * n;
                sv[axis] = xt::range(0, e.shape(axis));
            }
            return_type out(new_shape);

            xt::strided_view(out, sv) = e;

            xt::xstrided_slice_vector svs(dim, xt::all());
            xt::xstrided_slice_vector svt(dim, xt::all());

            for (size_type axis = 0; axis < dim; ++axis)
            {
                for (size_type i = 1; i < static_cast<size_type>(reps[axis]); ++i)
                {
                    svs[axis] = xt::range(0, e.shape(axis));
                    svt[axis] = xt::range(i * e.shape(axis), (i + 1) * e.shape(axis));
                    xt::strided_view(out, svt) = xt::strided_view(out, svs);
                    svs[axis] = xt::all();
                    svt[axis] = xt::all();
                }
            }

            return out;
        }
    }

    /**
     * @brief Tile an array.
     *
     * @param e The array.
     * @param reps The number of repetitions of A along each axis.
     * @return The tiled array.
     */

    template <class E, class S = typename std::decay_t<E>::size_type>
    inline auto tile(E&& e, std::initializer_list<S> reps)
    {
        return detail::tile(std::forward<E>(e), std::vector<S>{reps});
    }

    template <class E, class C, XTL_REQUIRES(std::negation<xtl::is_integral<C>>)>
    inline auto tile(E&& e, const C& reps)
    {
        return detail::tile(std::forward<E>(e), reps);
    }

    /**
     * @brief Tile an array.
     *
     * @param e The array.
     * @param reps The number of repetitions of A along the first axis.
     * @return The tiled array.
     */
    template <class E, class S = typename std::decay_t<E>::size_type, XTL_REQUIRES(xtl::is_integral<S>)>
    inline auto tile(E&& e, S reps)
    {
        if (e.dimension() == 0)
        {
            XTENSOR_THROW(std::runtime_error, "tile: a 0-D expression has no first axis to repeat along.");
        }
        std::vector<S> tw(e.dimension(), static_cast<S>(1));
        tw[0] = reps;
        return detail::tile(std::forward<E>(e), tw);
    }
}

#endif
