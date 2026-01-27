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
     * @enum access_t
     * @brief Defines the access policy for the underlying container.
     */
    enum class access_t
    {
        SAFE,   ///< Use .at() accessor (bounds checked).
        UNSAFE  ///< Use operator() accessor (no bounds checking).
    };

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
    public:

        /// @brief The view type this mapper works with
        using view_type = xt::xview<UnderlyingContainer, Slices...>;

        /// @brief Reference type of the underlying view.
        using reference = typename xt::xview<UnderlyingContainer, Slices...>::reference;

        /// @brief Const reference type of the underlying view.
        using const_reference = typename xt::xview<UnderlyingContainer, Slices...>::const_reference;

        /// @brief Total number of explicitly passed slices in the view
        static constexpr size_t n_slices = sizeof...(Slices);

        /// @brief Number of slices that are integral constants (fixed indices)
        static constexpr size_t nb_integral_slices = (std::is_integral_v<Slices> + ...);

        /// @brief Number of slices that are xt::newaxis (insert a  dimension)
        static constexpr size_t nb_new_axis_slices = (xt::detail::is_newaxis_v<Slices> + ...);

        /**
         * Compute how many indices are needed to address the underlying container
         * when given N indices in the view.
         */
        template <std::integral... Indices>
        static constexpr size_t n_indices_full_v = size_t(sizeof...(Indices) + nb_integral_slices);

        /**
         * @brief Map view indices to container reference using UNSAFE access.
         * @param container The source container.
         * @param view The view defining the mapping.
         * @param indices The indices in view-space.
         * @return Reference to the element in the container.
         */
        template <std::integral... Indices>
        reference map(UnderlyingContainer& container, const view_type& view, const Indices... indices) const;

        /**
         * @brief Map view indices to container const_reference using UNSAFE access.
         * @param container The source container.
         * @param view The view defining the mapping.
         * @param indices The indices in view-space.
         * @return Reference to the element in the container.
         */
        template <std::integral... Indices>
        const_reference
        cmap(const UnderlyingContainer& container, const view_type& view, const Indices... indices) const;

        /**
         * @brief Map view indices to container reference using SAFE access.
         * @param container The source container.
         * @param view The view defining the mapping.
         * @param indices The indices in view-space.
         * @return Reference to the element in the container.
         */
        template <std::integral... Indices>
        reference map_at(UnderlyingContainer& container, const view_type& view, const Indices... indices) const;

        /**
         * @brief Map view indices to container const_reference using SAFE access.
         * @param container The source container.
         * @param view The view defining the mapping.
         * @param indices The indices in view-space.
         * @return Reference to the element in the container.
         */
        template <std::integral... Indices>
        const_reference
        cmap_at(const UnderlyingContainer& container, const view_type& view, const Indices... indices) const;

        /// @brief Return the dimensionality of the view
        size_t dimension(const UnderlyingContainer& container) const;

    private:

        /// @brief Alias for selecting reference type based on const-correctness.
        template <bool IS_CONST>
        using conditional_reference = std::conditional_t<IS_CONST, const_reference, reference>;

        /// @brief Helper type alias for the I-th slice type
        template <size_t I>
        using slice_type = std::tuple_element_t<I, std::tuple<Slices...>>;

        /// @brief True if the I-th slice is an integral slice (fixed index)
        template <size_t I>
        static consteval bool is_slice_integral();

        /// @brief True if the I-th slice is a newaxis slice
        template <size_t I>
        static consteval bool is_slice_new_axis();

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
            using not_new_axis_type = typename indices_sequence_helper<first + 1, bound, indices..., first>::type;

            // we skip the current axis
            using new_axis_type = typename indices_sequence_helper<first + 1, bound, indices...>::type;

            // NOTE: is_slice_new_axis works even if first >= sizeof...(Slices)
            using type = std::conditional_t<is_slice_new_axis<first>(), new_axis_type, not_new_axis_type>;
        };

        /// @brief Base case: recursion termination
        template <size_t bound, size_t... indices>
        struct indices_sequence_helper<bound, bound, indices...>
        {
            using type = std::index_sequence<indices...>;
        };

        ///<  @brief Index sequence of non-newaxis slices
        template <size_t bound>
        using indices_sequence = indices_sequence_helper<0, bound>::type;

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
         * @brief Main recursion/logic handler for mapping operations.
         * Handles dimension dropping if the provided index count exceeds view dimensionality.
         *
         * @tparam IS_CONST Boolean flag; true if the operation is on a const container.
         * @tparam ACCESS The access policy (SAFE for .at(), UNSAFE for operator()).
         * @param is_const Tag used for compile-time dispatching of const-correctness.
         * @param container The underlying container (xarray, xtensor, etc.) being accessed.
         * @param access Tag used for compile-time dispatching of the access method.
         * @param view The xview instance that defines the coordinate transformation.
         * @param firstIndice The current leading index in the coordinate pack.
         * @param otherIndices The remaining indices in the coordinate pack.
         */
        template <bool IS_CONST, access_t ACCESS, std::integral FirstIndice, std::integral... OtherIndices>
        conditional_reference<IS_CONST> map_main(
            std::bool_constant<IS_CONST> /* is_const */,
            std::conditional_t<IS_CONST, const UnderlyingContainer&, UnderlyingContainer&> container,
            std::integral_constant<access_t, ACCESS> /* access */,
            const view_type& view,
            const FirstIndice firstIndice,
            const OtherIndices... otherIndices
        ) const;

        /**
         * @brief Base case for map_main recursion, where no indices is supplied and assumes (0, 0, ...).
         *
         * @tparam IS_CONST Boolean flag; true if the operation is on a const container.
         * @tparam ACCESS The access policy (SAFE for .at(), UNSAFE for operator()).
         * @param is_const Tag used for compile-time dispatching of const-correctness.
         * @param container The underlying container (xarray, xtensor, etc.) being accessed.
         * @param access Tag used for compile-time dispatching of the access method.
         * @param view The xview instance that defines the coordinate transformation.
         */
        template <bool IS_CONST, access_t ACCESS>
        conditional_reference<IS_CONST> map_main(
            std::bool_constant<IS_CONST> /* is_const */,
            std::conditional_t<IS_CONST, const UnderlyingContainer&, UnderlyingContainer&> container,
            std::integral_constant<access_t, ACCESS> /* access */,
            const view_type& view
        ) const;

        /**
         * @brief Maps all indices and accesses the container.
         *
         * @tparam IS_CONST Boolean flag for const-correctness.
         * @tparam ACCESS The access policy (SAFE or UNSAFE).
         * @tparam n_indices The size of the index array (calculated from view/container info).
         * @tparam Is A pack of indices `0, 1, ..., n-1` used to unroll the mapping loop.
         * @param is_const Tag for const-correctness dispatch.
         * @param container The underlying container being accessed.
         * @param access Tag for access method dispatch.
         * @param view The xview instance providing the slice transformations.
         * @param is_seq An index sequence used to drive the parameter pack expansion.
         * @param indices An array containing the view-space indices to be mapped.
         */
        template <bool IS_CONST, access_t ACCESS, size_t n_indices, size_t... Is>
        conditional_reference<IS_CONST> map_all_indices(
            std::bool_constant<IS_CONST> /* is_const */,
            std::conditional_t<IS_CONST, const UnderlyingContainer&, UnderlyingContainer&> container,
            std::integral_constant<access_t, ACCESS> /* access */,
            const view_type& view,
            std::index_sequence<Is...> /* is_seq */,
            const std::array<size_t, n_indices>& indices
        ) const;

        /// @brief Expand view indices into a full index array, inserting dummy indices for integral slices
        template <std::integral... Indices>
        std::array<size_t, n_indices_full_v<Indices...>> get_indices_full(const Indices... indices) const;
    };

    /*******************************
     * index_mapper implementation *
     *******************************/

    template <class UnderlyingContainer, class... Slices>
    template <size_t I>
    consteval bool index_mapper<xt::xview<UnderlyingContainer, Slices...>>::is_slice_integral()
    {
        if constexpr (I < sizeof...(Slices))
        {
            return std::is_integral_v<slice_type<I>>;
        }
        else
        {
            return false;
        }
    }

    template <class UnderlyingContainer, class... Slices>
    template <size_t I>
    consteval bool index_mapper<xt::xview<UnderlyingContainer, Slices...>>::is_slice_new_axis()
    {
        if constexpr (I < sizeof...(Slices))
        {
            return xt::detail::is_newaxis_v<slice_type<I>>;
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

            ((args_full[Is] = (is_slice_integral<Is>()) ? size_t(0) : *it++), ...);
        };

        fill_args_full(std::make_index_sequence<n_indices_full>{});

        return args_full;
    }

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map(
        UnderlyingContainer& container,
        const view_type& view,
        const Indices... indices
    ) const -> reference
    {
        return map_main(
            std::false_type{},
            container,
            std::integral_constant<access_t, access_t::UNSAFE>{},
            view,
            indices...
        );
    }

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::cmap(
        const UnderlyingContainer& container,
        const view_type& view,
        const Indices... indices
    ) const -> const_reference
    {
        return map_main(
            std::true_type{},
            container,
            std::integral_constant<access_t, access_t::UNSAFE>{},
            view,
            indices...
        );
    }

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_at(
        UnderlyingContainer& container,
        const view_type& view,
        const Indices... indices
    ) const -> reference
    {
        return map_main(
            std::false_type{},
            container,
            std::integral_constant<access_t, access_t::SAFE>{},
            view,
            indices...
        );
    }

    template <class UnderlyingContainer, class... Slices>
    template <std::integral... Indices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::cmap_at(
        const UnderlyingContainer& container,
        const view_type& view,
        const Indices... indices
    ) const -> const_reference
    {
        return map_main(
            std::true_type{},
            container,
            std::integral_constant<access_t, access_t::SAFE>{},
            view,
            indices...
        );
    }

    template <class UnderlyingContainer, class... Slices>
    template <bool IS_CONST, access_t ACCESS, std::integral FirstIndice, std::integral... OtherIndices>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_main(
        std::bool_constant<IS_CONST> is_const,
        std::conditional_t<IS_CONST, const UnderlyingContainer&, UnderlyingContainer&> container,
        std::integral_constant<access_t, ACCESS> access,
        const view_type& view,
        const FirstIndice firstIndice,
        const OtherIndices... otherIndices
    ) const -> conditional_reference<IS_CONST>
    {
        constexpr size_t n_indices_full = n_indices_full_v<FirstIndice, OtherIndices...>;

        constexpr size_t underlying_n_dimensions = xt::static_dimension<
            typename std::decay_t<UnderlyingContainer>::shape_type>::value;

        // If there is too many indices, we need to drop the first ones.
        // If the number of dimensions of the underlying container is known at compile time we can drop them
        // at compile time Else a runtime-test is requires, which, breaks vectorization.
        // I don't know if we can do it in another way.

        if constexpr (underlying_n_dimensions != size_t(-1))
        {
            // the number of dimensions of the underlying container is known at compile time.
            constexpr size_t n_dimensions = underlying_n_dimensions - nb_integral_slices + nb_new_axis_slices;

            // we can perform compile time checks
            if constexpr (1 + sizeof...(OtherIndices) > n_dimensions)
            {
                return map_main(is_const, container, access, view, otherIndices...);
            }
            else
            {
                return map_all_indices(
                    is_const,
                    container,
                    access,
                    view,
                    indices_sequence<n_indices_full>{},
                    get_indices_full(firstIndice, otherIndices...)
                );
            }
        }
        else
        {
            // we need execution time checks
            if (1 + sizeof...(OtherIndices) > dimension(container))
            {
                return map_main(is_const, container, access, view, otherIndices...);
            }
            else
            {
                return map_all_indices(
                    is_const,
                    container,
                    access,
                    view,
                    indices_sequence<n_indices_full>{},
                    get_indices_full(firstIndice, otherIndices...)
                );
            }
        }
    }

    template <class UnderlyingContainer, class... Slices>
    template <bool IS_CONST, access_t ACCESS>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_main(
        std::bool_constant<IS_CONST> is_const,
        std::conditional_t<IS_CONST, const UnderlyingContainer&, UnderlyingContainer&> container,
        std::integral_constant<access_t, ACCESS> access,
        const view_type& view
    ) const -> conditional_reference<IS_CONST>
    {
        constexpr size_t n_indices_full = n_indices_full_v<>;

        return map_all_indices(
            is_const,
            container,
            access,
            view,
            indices_sequence<n_indices_full>{},
            get_indices_full()
        );
    }

    template <class UnderlyingContainer, class... Slices>
    template <bool IS_CONST, access_t ACCESS, size_t n_indices, size_t... Is>
    auto index_mapper<xt::xview<UnderlyingContainer, Slices...>>::map_all_indices(
        std::bool_constant<IS_CONST> /* is_const */,
        std::conditional_t<IS_CONST, const UnderlyingContainer&, UnderlyingContainer&> container,
        std::integral_constant<access_t, ACCESS> /* access */,
        const view_type& view,
        std::index_sequence<Is...> /* is_seq */,
        const std::array<size_t, n_indices>& indices
    ) const -> conditional_reference<IS_CONST>
    {
        if constexpr (ACCESS == access_t::SAFE)
        {
            return container.at(map_ith_index<Is>(view, indices[Is])...);
        }
        else
        {
            return container(map_ith_index<Is>(view, indices[Is])...);
        }
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

            static_assert(not xt::detail::is_newaxis_v<current_slice>);

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
