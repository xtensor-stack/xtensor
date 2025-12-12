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

template<class UndefinedView> struct index_mapper; 

template<class UnderlyingContainer, class... Slices>
class index_mapper< xt::xview<UnderlyingContainer, Slices...> >
{
    static constexpr size_t n_slices = sizeof...(Slices);
    static constexpr size_t n_free   = ((!std::is_integral_v<Slices>) + ... );
public:
    using view_type = xt::xview<UnderlyingContainer, Slices...>;

    using value_type = typename xt::xview<UnderlyingContainer, Slices...>::value_type;
private:
    template<size_t I> using ith_slice_type = std::tuple_element_t<I, std::tuple<Slices...> >;

    template<size_t first, size_t... indices>
    struct indices_sequence_helper
    {
        using not_new_axis_type = typename indices_sequence_helper<first + 1, indices..., first>::Type; // we add the current axis
        using new_axis_type     = typename indices_sequence_helper<first + 1, indices...>::Type;        // we skip the current axis

        using Type = std::conditional_t< xt::detail::is_newaxis< ith_slice_type<first> >::value , new_axis_type, not_new_axis_type>;
    };

    // closing the recurence
    template<size_t... indices>
    struct indices_sequence_helper<n_slices, indices...>
    {
        using Type = std::index_sequence<indices...>;
    };

    using indices_sequence  = indices_sequence_helper<0>::Type;
    
    static constexpr size_t n_all_indices = indices_sequence::size();

    template<size_t I, std::integral Index>
    size_t map_ith_index(const view_type& view, const Index i) const;

    template<size_t... Is>
    value_type map_all_indices(const UnderlyingContainer& container, const view_type& view, std::index_sequence<Is...>, const std::array<size_t, n_all_indices>& indices) const 
        requires(sizeof...(Is) == n_all_indices);

    template<size_t... Is>
    value_type map_at_all_indices(const UnderlyingContainer& container, const view_type& view, std::index_sequence<Is...>, const std::array<size_t, n_all_indices>& indices) const 
        requires(sizeof...(Is) == n_all_indices);
public:
    template<std::integral... Indices> 
    value_type map(const UnderlyingContainer& container, const view_type& view, const Indices... indices) const 
        requires(sizeof...(Indices) == n_free);
        
    template<std::integral... Indices> 
    value_type map_at(const UnderlyingContainer& container, const view_type& view, const Indices... indices) const 
        requires(sizeof...(Indices) == n_free);

    constexpr size_t dimension() const { return n_free; }
};

/*******************************
 * index_mapper implementation *
 *******************************/
 
template<class UnderlyingContainer, class... Slices> template<std::integral... Indices> 
auto index_mapper< xt::xview<UnderlyingContainer, Slices...> >::map(
    const UnderlyingContainer& container, 
    const view_type&           view, 
    const Indices...           indices) const -> value_type 
    requires(sizeof...(Indices) == n_free)
{
    std::array<size_t, sizeof...(indices)> args{ size_t(indices)...};

	auto it = std::cbegin(args);
	std::array<size_t, n_all_indices> args_full{ (std::is_integral_v<Slices> ? size_t(0) : *it++)... };
	
    return map_all_indices(container, view, indices_sequence{}, args_full);
}
 
template<class UnderlyingContainer, class... Slices> template<std::integral... Indices> 
auto index_mapper< xt::xview<UnderlyingContainer, Slices...> >::map_at(
	const UnderlyingContainer& container, 
    const view_type&           view, 
    const Indices...           indices) const -> value_type 
    requires(sizeof...(Indices) == n_free)
{
    std::array<size_t, sizeof...(indices)> args{ size_t(indices)...};
	
    auto it = std::cbegin(args);
    std::array<size_t, n_all_indices> args_full{ (std::is_integral_v<Slices> ? size_t(0) : *it++)... };

    return map_at_all_indices(container, view, indices_sequence{}, args_full);
}

template<class UnderlyingContainer, class... Slices> template<size_t... Is>
auto index_mapper< xt::xview<UnderlyingContainer, Slices...> >::map_all_indices(const UnderlyingContainer& container, const view_type& view, std::index_sequence<Is...>, const std::array<size_t, n_all_indices>& indices) const -> value_type
    requires(sizeof...(Is) == n_all_indices)
{		
	return container(map_ith_index<Is>(view, indices[Is])...);
}

template<class UnderlyingContainer, class... Slices> template<size_t... Is>
auto index_mapper< xt::xview<UnderlyingContainer, Slices...> >::map_at_all_indices(const UnderlyingContainer& container, const view_type& view, std::index_sequence<Is...>, const std::array<size_t, n_all_indices>& indices) const -> value_type
        requires(sizeof...(Is) == n_all_indices)
{
	return container.at(map_ith_index<Is>(view, indices[Is])...);
}

template<class UnderlyingContainer, class... Slices> template<size_t I, std::integral Index>
auto index_mapper< xt::xview<UnderlyingContainer, Slices...> >::map_ith_index(const view_type& view, const Index i) const -> size_t
{
    using current_slice = std::tuple_element_t<I, std::tuple<Slices...>>;

    static_assert(not xt::detail::is_newaxis<current_slice>::value);

    const auto& slice = std::get<I>(view.slices());
    
    if constexpr (std::is_integral_v<current_slice>) { assert(i == 0);           return size_t(slice);    }
    else                                             { assert(i < slice.size()); return size_t(slice(i)); }
}

} // namespace xt

#endif // XTENSOR_INDEX_MAPPER_HPP
