/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_INDEX_MAPPER_HPP
#define XTENSOR_INDEX_MAPPER_HPP

#include "xview.hpp"

namespace xt
{

    template <class UndefinedView>
    struct index_mapper;

    /**
     * @class index_mapper
     * @brief A helper class for mapping indices between views and their underlying containers.
     *
     * The `index_mapper` class provides functionality to convert indices from a view's coordinate system
     * to the corresponding indices in the underlying container. This is particularly useful for views
     * that contain integral slices (fixed indices), as these slices reduce the dimensionality of the view.
     *
     * @tparam UndefinedView The primary template parameter, specialized for `xt::xview` types.
     *
     * @note This class is specialized for `xt::xview<UnderlyingContainer, Slices...>` types only.
     *       Other view types will trigger a compilation error.
     *
     * @example
     * @code
     * xt::xarray<double> a = xt::arange(24).reshape({2, 3, 4});
     * auto view1 = xt::view(a, 1, xt::all(), xt::all());  // Fixed first dimension
     * index_mapper<decltype(view1)> mapper;
     *
     * // Map view indices (i,j) to container indices (1,i,j)
     * double val = mapper.map(a, view1, 0, 0);  // Returns a(1, 0, 0)
     * double val2 = mapper.map(a, view1, 1, 2); // Returns a(1, 1, 2)
     * @endcode
     */
    template <class UnderlyingContainer, class... Slices>
    class index_mapper<xt::xview<UnderlyingContainer, Slices...>>
    {
        static constexpr size_t n_slices = sizeof...(Slices);  ///<  @brief Total number of slices in the view
        static constexpr size_t n_free = ((!std::is_integral_v<Slices>) +...);  ///<  @brief Number of free
                                                                                ///<  (non-integral) slices

    public:

        using view_type = xt::xview<UnderlyingContainer, Slices...>;  ///<   @brief The view type this mapper
                                                                      ///<   works with

        using value_type = typename xt::xview<UnderlyingContainer, Slices...>::value_type;  ///<  @brief Value
                                                                                            ///<  type of the
                                                                                            ///<  underlying
                                                                                            ///<  container

    private:

        /// @brief Helper type alias for the I-th slice type
        template <size_t I>
        using ith_slice_type = std::tuple_element_t<I, std::tuple<Slices...>>;

        /**
         * @brief Helper metafunction to generate an index sequence excluding newaxis slices.
         *
         * This recursive template builds an `std::index_sequence` containing indices of slices
         * that are not `xt::newaxis`. Newaxis slices increase dimensionality but don't correspond
         * to actual dimensions in the underlying container.
         *
         * @tparam first Current slice index being processed.
         * @tparam indices... Accumulated indices of non-newaxis slices.
         */
        template <size_t first, size_t... indices>
        struct indices_sequence_helper
        {
            using not_new_axis_type = typename indices_sequence_helper<first + 1, indices..., first>::Type;  // we add the current axis
            using new_axis_type = typename indices_sequence_helper<first + 1, indices...>::Type;  // we skip
                                                                                                  // the
                                                                                                  // current
                                                                                                  // axis

            using Type = std::conditional_t<xt::detail::is_newaxis<ith_slice_type<first>>::value, new_axis_type, not_new_axis_type>;
        };

        /// @brief Base case: recursion termination
        template <size_t... indices>
        struct indices_sequence_helper<n_slices, indices...>
        {
            using Type = std::index_sequence<indices...>;
        };

        /// @brief Index sequence of non-newaxis
        using indices_sequence = indices_sequence_helper<0>::Type;

        /**
         * @brief Maps an index for a specific slice to the corresponding index in the underlying container.
         *
         * For integral slices (fixed indices), returns the fixed index value.
         * For non-integral slices (like `xt::all()`), applies the slice transformation to the index.
         *
         * @tparam I The slice index to map.
         * @tparam Index Type of the index (must be integral).
         * @param view The view object containing slice information.
         * @param i The index within the slice to map.
         * @return size_t The mapped index in the underlying container.
         *
         * @throws Assertion failure if `i != 0` for integral slices.
         * @throws Assertion failure if `i >= slice.size()` for non-integral slices.
         */
        template <size_t I, std::integral Index>
        size_t map_ith_index(const view_type& view, const Index i) const;

        /**
         * @brief Maps all indices and accesses the container.
         *
         * @tparam Is Index sequence for parameter pack expansion.
         * @param container The underlying container to access.
         * @param view The view providing slice information.
         * @param indices Array of indices for all slices.
         * @return value_type The value at the mapped location in the container.
         */
        template <size_t... Is>
        value_type map_all_indices(
            const UnderlyingContainer& container,
            const view_type& view,
            std::index_sequence<Is...>,
            const std::array<size_t, n_slices>& indices
        ) const;

        /**
         * @brief Maps all indices and accesses the container with bounds checking.
         *
         * Same as `map_all_indices` but uses `container.at()` which performs bounds checking.
         *
         * @tparam Is Index sequence for parameter pack expansion.
         * @param container The underlying container to access.
         * @param view The view providing slice information.
         * @param indices Array of indices for all slices.
         * @return value_type The value at the mapped location in the container.
         *
         * @throws std::out_of_range if any index is out of bounds.
         */
        template <size_t... Is>
        value_type map_at_all_indices(
            const UnderlyingContainer& container,
            const view_type& view,
            std::index_sequence<Is...>,
            const std::array<size_t, n_slices>& indices
        ) const;

    public:

        /**
         * @brief Maps view indices to container indices and returns the value.
         *
         * Converts the provided indices (for the free dimensions of the view) to
         * the corresponding indices in the underlying container and returns the value.
         *
         * @tparam Indices Types of the indices (must be integral).
         * @param container The underlying container to access.
         * @param view The view providing slice information.
         * @param indices The indices for the free dimensions of the view.
         * @return value_type The value at the mapped location in the container.
         *
         * @note The number of provided indices must equal `n_free` (number of non-integral slices).
         *
         * @example
         * @code
         * // For view(a, 1, all(), all()):
         * // n_free = 2 (two all() slices)
         * mapper.map(a, view, i, j);  // Maps to a(1, i, j)
         * @endcode
         */
        template <std::integral... Indices>
        value_type
        map(const UnderlyingContainer& container, const view_type& view, const Indices... indices) const
            requires(sizeof...(Indices) == n_free);

        /**
         * @brief Maps view indices to container indices with bounds checking.
         *
         * Same as `map()` but uses bounds-checked access via `container.at()`.
         *
         * @tparam Indices Types of the indices (must be integral).
         * @param container The underlying container to access.
         * @param view The view providing slice information.
         * @param indices The indices for the free dimensions of the view.
         * @return value_type The value at the mapped location in the container.
         *
         * @throws std::out_of_range if any mapped index is out of bounds.
         */
        template <std::integral... Indices>
        value_type
        map_at(const UnderlyingContainer& container, const view_type& view, const Indices... indices) const
            requires(sizeof...(Indices) == n_free);

        constexpr size_t dimension() const
        {
            return n_free;
        }
    };

    /*******************************
     * index_mapper implementation *
     *******************************/

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map(
        const UnderlyingContainer& container,
        const view_type& view,
        const Indices... indices
    ) const -> value_type
        requires(sizeof...(Indices) == n_free)
    {
        std::array<size_t, sizeof...(indices)> args{size_t(indices)...};

        auto it = std::cbegin(args);
        std::array<size_t, n_slices> args_full{(std::is_integral_v<Slices> ? size_t(0) : *it++)...};

        return map_all_indices(container, view, indices_sequence{}, args_full);
    }

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_at(
        const UnderlyingContainer& container,
        const view_type& view,
        const Indices... indices
    ) const -> value_type
        requires(sizeof...(Indices) == n_free)
    {
        std::array<size_t, sizeof...(indices)> args{size_t(indices)...};

        auto it = std::cbegin(args);
        std::array<size_t, n_slices> args_full{(std::is_integral_v<Slices> ? size_t(0) : *it++)...};

        return map_at_all_indices(container, view, indices_sequence{}, args_full);
    }

    template <class UnderlyingContainer, class... Slices>
    template <size_t... Is>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_all_indices(
        const UnderlyingContainer& container,
        const view_type& view,
        std::index_sequence<Is...>,
        const std::array<size_t, n_slices>& indices
    ) const -> value_type
    {
        return container(map_ith_index<Is>(view, indices[Is])...);
    }

    template <class UnderlyingContainer, class... Slices>
    template <size_t... Is>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_at_all_indices(
        const UnderlyingContainer& container,
        const view_type& view,
        std::index_sequence<Is...>,
        const std::array<size_t, n_slices>& indices
    ) const -> value_type
    {
        return container.at(map_ith_index<Is>(view, indices[Is])...);
    }

    template <class UnderlyingContainer, class... Slices>
    template <size_t I, std::integral Index>
    auto
    index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_ith_index(const view_type& view, const Index i) const
        -> size_t
    {
        using current_slice = std::tuple_element_t<I, std::tuple<Slices...>>;

        static_assert(not xt::detail::is_newaxis<current_slice>::value);

        const auto& slice = std::get<I>(view.slices());

        if constexpr (std::is_integral_v<current_slice>)
        {
            assert(i == 0);
            return size_t(slice);
        }
        else
        {
            assert(i < slice.size());
            return size_t(slice(i));
        }
    }

}  // namespace xt

#endif  // XTENSOR_INDEX_MAPPER_HPP
