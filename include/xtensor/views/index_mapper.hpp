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
        /// @brief Total number of explicitly passed slices in the view
        static constexpr size_t n_slices = sizeof...(Slices);

        /// @brief Number of slices that are integral constants (fixed indices)
        static constexpr size_t nb_integral_slices = (std::is_integral_v<Slices> + ...);

        /// @brief Number of slices that are xt::newaxis (insert a  dimension)
        static constexpr size_t nb_new_axis_slices = (xt::detail::is_newaxis<Slices>::value + ...);

        /**
         * Compute how many indices are needed to address the underlying container
         * when given N indices in the view.
         */
        template <std::integral... Indices>
        static constexpr size_t n_indices_full_v = size_t(
            sizeof...(Indices) + nb_integral_slices - nb_new_axis_slices
        );

    public:

        /// @brief The view type this mapper works with
        using view_type = xt::xview<UnderlyingContainer, Slices...>;

        ///<  @brief Value type of the underlying container
        using value_type = typename xt::xview<UnderlyingContainer, Slices...>::value_type;

    private:

        /// @brief Helper type alias for the I-th slice type
        template <size_t I>
        using ith_slice_type = std::tuple_element_t<I, std::tuple<Slices...>>;

        /// @brief True if the I-th slice is an integral slice (fixed index)
        template <size_t I>
        static consteval bool is_ith_slice_integral();

        /// @brief True if the I-th slice is a newaxis slice
        template <size_t I>
        static consteval bool is_ith_slice_new_axis();

        /**
         * Helper metafunction to build an index_sequence that skips
         * newaxis slices.
         *
         * The resulting sequence contains only the indices that
         * correspond to real container dimensions.
         */
        template <size_t first, size_t bound, size_t... indices>
        struct indices_sequence_helper
        {
            // we add the current axis
            using not_new_axis_type = typename indices_sequence_helper<first + 1, bound, indices..., first>::Type;

            // we skip the current axis
            using new_axis_type = typename indices_sequence_helper<first + 1, bound, indices...>::Type;

            // NOTE: is_ith_slice_new_axis works even if first >= sizeof...(Slices)
            using Type = std::conditional_t<is_ith_slice_new_axis<first>(), new_axis_type, not_new_axis_type>;
        };

        /// @brief Base case: recursion termination
        template <size_t bound, size_t... indices>
        struct indices_sequence_helper<bound, bound, indices...>
        {
            using Type = std::index_sequence<indices...>;
        };

        ///<  @brief Index sequence of non-newaxis slices
        template <size_t bound>
        using indices_sequence = indices_sequence_helper<0, bound>::Type;

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
        template <size_t n_indices, size_t... Is>
        value_type map_all_indices(
            const UnderlyingContainer& container,
            const view_type& view,
            std::index_sequence<Is...>,
            const std::array<size_t, n_indices>& indices
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
        template <size_t n_indices, size_t... Is>
        value_type map_at_all_indices(
            const UnderlyingContainer& container,
            const view_type& view,
            std::index_sequence<Is...>,
            const std::array<size_t, n_indices>& indices
        ) const;

        /// @brief Expand view indices into a full index array, inserting dummy indices for integral slices
        template <std::integral... Indices>
        std::array<size_t, n_indices_full_v<Indices...>> get_indices_full(const Indices... indices) const;

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
         * @example
         * @code
         * // For view(a, 1, all(), all()):
         * mapper.map(a, view, i, j);  // Maps to a(1, i, j)
         * @endcode
         */
        template <std::integral... Indices>
        value_type
        map(const UnderlyingContainer& container, const view_type& view, const Indices... indices) const;

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
        map_at(const UnderlyingContainer& container, const view_type& view, const Indices... indices) const;

        /// @brief Return the dimensionality of the view
        size_t dimension(const UnderlyingContainer& container) const;
    };

    /*******************************
     * index_mapper implementation *
     *******************************/

    template <class UnderlyingContainer, class... Slices>
    template <size_t I>
    consteval bool index_mapper<xt::xview<UnderlyingContainer, Slices...>>::is_ith_slice_integral()
    {
        if constexpr (I < sizeof...(Slices))
        {
            return std::is_integral_v<ith_slice_type<I>>;
        }
        else
        {
            return false;
        }
    }

    template <class UnderlyingContainer, class... Slices>
    template <size_t I>
    consteval bool index_mapper<xt::xview<UnderlyingContainer, Slices...>>::is_ith_slice_new_axis()
    {
        if constexpr (I < sizeof...(Slices))
        {
            return xt::detail::is_newaxis<ith_slice_type<I>>::value;
        }
        else
        {
            return false;
        }
    }

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto
    index_mapper<xt::xview<UnderlyingContainer, Slices...>>::get_indices_full(const Indices... indices) const
        -> std::array<size_t, n_indices_full_v<Indices...>>
    {
        constexpr size_t n_indices_full = n_indices_full_v<Indices...>;

        std::array<size_t, sizeof...(indices)> args{size_t(indices)...};
        std::array<size_t, n_indices_full> args_full;

        const auto fill_args_full = [&args_full, &args]<size_t... Is>(std::index_sequence<Is...>)
        {
            auto it = std::cbegin(args);

            ((args_full[Is] = (is_ith_slice_integral<Is>()) ? size_t(0) : *it++), ...);
        };

        fill_args_full(std::make_index_sequence<n_indices_full>{});

        return args_full;
    }

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map(
        const UnderlyingContainer& container,
        const view_type& view,
        const Indices... indices
    ) const -> value_type
    {
        constexpr size_t n_indices_full = n_indices_full_v<Indices...>;

        return map_all_indices(container, view, indices_sequence<n_indices_full>{}, get_indices_full(indices...));
    }

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_at(
        const UnderlyingContainer& container,
        const view_type& view,
        const Indices... indices
    ) const -> value_type
    {
        constexpr size_t n_indices_full = n_indices_full_v<Indices...>;

        return map_at_all_indices(container, view, indices_sequence<n_indices_full>{}, get_indices_full(indices...));
    }

    template <class UnderlyingContainer, class... Slices>
    template <size_t n_indices, size_t... Is>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_all_indices(
        const UnderlyingContainer& container,
        const view_type& view,
        std::index_sequence<Is...>,
        const std::array<size_t, n_indices>& indices
    ) const -> value_type
    {
        return container(map_ith_index<Is>(view, indices[Is])...);
    }

    template <class UnderlyingContainer, class... Slices>
    template <size_t n_indices, size_t... Is>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_at_all_indices(
        const UnderlyingContainer& container,
        const view_type& view,
        std::index_sequence<Is...>,
        const std::array<size_t, n_indices>& indices
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
        if constexpr (I < sizeof...(Slices))
        {
            // if the slice is explicitly specified, use it
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
        else
        {
            // else assume xt::all
            return i;
        }
    }

    template <class UnderlyingContainer, class... Slices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::dimension(const UnderlyingContainer& container
    ) const -> size_t
    {
        return container.dimension() - nb_integral_slices + nb_new_axis_slices;
    }

}  // namespace xt

#endif  // XTENSOR_INDEX_MAPPER_HPP
