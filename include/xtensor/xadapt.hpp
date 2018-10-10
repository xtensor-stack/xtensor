/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ADAPT_HPP
#define XTENSOR_ADAPT_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>

#include <xtl/xsequence.hpp>

#include "xarray.hpp"
#include "xtensor.hpp"
#include "xfixed.hpp"

namespace xt
{
    namespace detail
    {
        template <class>
        struct array_size_impl;

        template <class T, std::size_t N>
        struct array_size_impl<std::array<T, N>>
        {
            static constexpr std::size_t value = N;
        };

        template <class C>
        using array_size = array_size_impl<std::decay_t<C>>;

        template <class P>
        struct default_allocator_for_ptr
        {
            using type = std::allocator<std::remove_const_t<std::remove_pointer_t<std::remove_reference_t<P>>>>;
        };

        template <class P>
        using default_allocator_for_ptr_t = typename default_allocator_for_ptr<P>::type;
    }

    /**************************
     * xarray_adaptor builder *
     **************************/

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and layout.
     * @param container the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, class SC,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0,
              typename std::enable_if_t<!std::is_pointer<C>::value, int> = 0>
    xarray_adaptor<xtl::closure_type_t<C>, L, std::decay_t<SC>>
    adapt(C&& container, const SC& shape, layout_type l = L);

    /**
     * Constructs an non-owning xarray_adaptor from a pointer with the specified shape and layout.
     * @param pointer the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, class SC,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0,
              typename std::enable_if_t<std::is_pointer<C>::value, int> = 0>
    inline auto adapt(C&& pointer, const SC& shape, layout_type l = L);

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param container the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     */
    template <class C, class SC, class SS,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0,
              typename std::enable_if_t<!std::is_same<layout_type, std::decay_t<SS>>::value, int> = 0>
    xarray_adaptor<xtl::closure_type_t<C>, layout_type::dynamic, std::decay_t<SC>>
    adapt(C&& container, SC&& shape, SS&& strides);

    /**
     * Constructs an xarray_adaptor of the given dynamically allocated C array,
     * with the specified shape and layout.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param shape the shape of the xarray_adaptor
     * @param l the layout_type of the xarray_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class P, class O, class SC, class A = detail::default_allocator_for_ptr_t<P>,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, L, SC>
    adapt(P&& pointer, typename A::size_type size, O ownership, const SC& shape, layout_type l = L, const A& alloc = A());

    /**
     * Constructs an xarray_adaptor of the given dynamically allocated C array,
     * with the specified shape and layout.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <class P, class O, class SC, class SS, class A = detail::default_allocator_for_ptr_t<P>,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int> = 0,
              typename std::enable_if_t<!std::is_same<layout_type, std::decay_t<SS>>::value, int> = 0>
    xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, layout_type::dynamic, std::decay_t<SC>>
    adapt(P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A());

    /***************************
     * xtensor_adaptor builder *
     ***************************/

    /**
     * Constructs a 1-D xtensor_adaptor of the given stl-like container,
     * with the specified layout_type.
     * @param container the container to adapt
     * @param l the layout_type of the xtensor_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C>
    xtensor_adaptor<C, 1, L>
    adapt(C&& container, layout_type l = L);

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param container the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, class SC,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0,
              typename std::enable_if_t<!std::is_pointer<C>::value, int> = 0>
    xtensor_adaptor<C, detail::array_size<SC>::value, L>
    adapt(C&& container, const SC& shape, layout_type l = L);

    /**
     * Constructs an non-owning xtensor_adaptor from a pointer with the specified shape and layout.
     * @param pointer the pointer to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, class SC,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0,
              typename std::enable_if_t<std::is_pointer<C>::value, int> = 0>
    auto adapt(C&& pointer, const SC& shape, layout_type l = L);

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param container the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     */
    template <class C, class SC, class SS,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0,
              typename std::enable_if_t<!std::is_same<layout_type, std::decay_t<SS>>::value, int> = 0>
    xtensor_adaptor<C, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(C&& container, SC&& shape, SS&& strides);

    /**
     * Constructs a 1-D xtensor_adaptor of the given dynamically allocated C array,
     * with the specified layout.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param l the layout_type of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class P, class O, class A = detail::default_allocator_for_ptr_t<P>>
    xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, 1, L>
    adapt(P&& pointer, typename A::size_type size, O ownership, layout_type l = L, const A& alloc = A());

    /**
     * Constructs an xtensor_adaptor of the given dynamically allocated C array,
     * with the specified shape and layout.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout_type of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class P, class O, class SC, class A = detail::default_allocator_for_ptr_t<P>,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0>
    xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, L>
    adapt(P&& pointer, typename A::size_type size, O ownership, const SC& shape, layout_type l = L, const A& alloc = A());

    /**
     * Constructs an xtensor_adaptor of the given dynamically allocated C array,
     * with the specified shape and strides.
     * @param pointer the pointer to the beginning of the dynamic array
     * @param size the size of the dynamic array
     * @param ownership indicates whether the adaptor takes ownership of the array.
     *        Possible values are ``no_ownerhsip()`` or ``acquire_ownership()``
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     * @param alloc the allocator used for allocating / deallocating the dynamic array
     */
    template <class P, class O, class SC, class SS, class A = detail::default_allocator_for_ptr_t<P>,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int> = 0,
              typename std::enable_if_t<!std::is_same<layout_type, std::decay_t<SS>>::value, int> = 0>
    xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(P&& pointer, typename A::size_type size, O ownership, SC&& shape, SS&& strides, const A& alloc = A());

    /**
     * Constructs an non-owning xtensor_fixed_adaptor from a pointer with the
     * specified shape and layout.
     * @param pointer the pointer to adapt
     * @param shape the shape of the xtensor_fixed_adaptor
     * @param l the layout_type of the xtensor_fixed_adaptor
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, std::size_t... X,
              typename std::enable_if_t<std::is_pointer<C>::value, int> = 0>
    inline auto adapt(C&& ptr, const fixed_shape<X...>& /*shape*/);

    /*****************************************
     * xarray_adaptor builder implementation *
     *****************************************/

    // shape only - container version
    template <layout_type L, class C, class SC,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>,
              typename std::enable_if_t<!std::is_pointer<C>::value, int>>
    inline xarray_adaptor<xtl::closure_type_t<C>, L, std::decay_t<SC>>
    adapt(C&& container, const SC& shape, layout_type l)
    {
        using return_type = xarray_adaptor<xtl::closure_type_t<C>, L, std::decay_t<SC>>;
        return return_type(std::forward<C>(container), shape, l);
    }

    template <layout_type L, class C, class SC,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>,
              typename std::enable_if_t<std::is_pointer<C>::value, int>>
    inline auto adapt(C&& pointer, const SC& shape, layout_type l)
    {
        using buffer_type = xbuffer_adaptor<C, xt::no_ownership, detail::default_allocator_for_ptr_t<C>>;
        using return_type = xarray_adaptor<buffer_type, L, std::decay_t<SC>>;
        std::size_t size = compute_size(shape);
        return return_type(buffer_type(pointer, size), shape, l);
    }

    // shape and strides - container version
    template <class C, class SC, class SS,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>,
              typename std::enable_if_t<!std::is_same<layout_type, std::decay_t<SS>>::value, int>>
    inline xarray_adaptor<xtl::closure_type_t<C>, layout_type::dynamic, std::decay_t<SC>>
    adapt(C&& container, SC&& shape, SS&& strides)
    {
        using return_type = xarray_adaptor<xtl::closure_type_t<C>, layout_type::dynamic, std::decay_t<SC>>;
        return return_type(std::forward<C>(container),
                           xtl::forward_sequence<typename return_type::inner_shape_type>(shape),
                           xtl::forward_sequence<typename return_type::inner_strides_type>(strides));
    }

    // shape only - buffer version
    template <layout_type L, class P, class O, class SC, class A,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>>
    inline xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, L, SC>
    adapt(P&& pointer, typename A::size_type size, O, const SC& shape, layout_type l, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xarray_adaptor<buffer_type, L, SC>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf), shape, l);
    }

    // shape and strides - buffer version
    template <class P, class O, class SC, class SS, class A,
              typename std::enable_if_t<!detail::is_array<std::decay_t<SC>>::value, int>,
              typename std::enable_if_t<!std::is_same<layout_type, std::decay_t<SS>>::value, int>>
    inline xarray_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, layout_type::dynamic, std::decay_t<SC>>
    adapt(P&& pointer, typename A::size_type size, O, SC&& shape, SS&& strides, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xarray_adaptor<buffer_type, layout_type::dynamic, std::decay_t<SC>>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf),
                           xtl::forward_sequence<typename return_type::inner_shape_type>(shape),
                           xtl::forward_sequence<typename return_type::inner_strides_type>(strides));
    }

    /******************************************
     * xtensor_adaptor builder implementation *
     ******************************************/

    // 1-D case - container version
    template <layout_type L, class C>
    inline xtensor_adaptor<C, 1, L>
    adapt(C&& container, layout_type l)
    {
        const std::array<typename std::decay_t<C>::size_type, 1> shape{container.size()};
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, 1, L>;
        return return_type(std::forward<C>(container), shape, l);
    }

    // shape only - container version
    template <layout_type L, class C, class SC,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>,
              typename std::enable_if_t<!std::is_pointer<C>::value, int>>
    inline xtensor_adaptor<C, detail::array_size<SC>::value, L>
    adapt(C&& container, const SC& shape, layout_type l)
    {
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, N, L>;
        return return_type(std::forward<C>(container), shape, l);
    }

    template <layout_type L, class C, class SC,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>,
              typename std::enable_if_t<std::is_pointer<C>::value, int>>
    inline auto adapt(C&& ptr, const SC& shape, layout_type l)
    {
        using buffer_type = xbuffer_adaptor<C, xt::no_ownership, detail::default_allocator_for_ptr_t<C>>;
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<buffer_type, N, L>;
        return return_type(buffer_type(ptr, compute_size(shape)), shape, l);
    }

    template <layout_type L, class C, std::size_t... X,
              typename std::enable_if_t<std::is_pointer<C>::value, int>>
    inline auto adapt(C&& ptr, const fixed_shape<X...>& /*shape*/)
    {
        using buffer_type = xbuffer_adaptor<C, xt::no_ownership, detail::default_allocator_for_ptr_t<C>>;
        using return_type = xfixed_adaptor<buffer_type, fixed_shape<X...>, L>;
        return return_type(buffer_type(ptr, detail::fixed_compute_size<fixed_shape<X...>>::value));
    }

#ifndef X_OLD_CLANG
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C, class T,  std::size_t N>
    inline auto adapt(C&& ptr, const T(&shape)[N])
    {
        using shape_type = std::array<std::size_t, N>;
        return adapt(std::forward<C>(ptr), xtl::forward_sequence<shape_type>(shape));
    }
#else
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class C>
    inline auto adapt(C&& ptr, std::initializer_list<std::size_t> shape)
    {
        using shape_type = xt::dynamic_shape<std::size_t>;
        return adapt(std::forward<C>(ptr), xtl::forward_sequence<shape_type>(shape));
    }
#endif

    // shape and strides - container version
    template <class C, class SC, class SS,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>,
              typename std::enable_if_t<!std::is_same<layout_type, std::decay_t<SS>>::value, int>>
    inline xtensor_adaptor<C, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(C&& container, SC&& shape, SS&& strides)
    {
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<xtl::closure_type_t<C>, N, layout_type::dynamic>;
        return return_type(std::forward<C>(container),
                           xtl::forward_sequence<typename return_type::inner_shape_type>(shape),
                           xtl::forward_sequence<typename return_type::inner_strides_type>(strides));
    }

    // 1-D case - buffer version
    template <layout_type L, class P, class O, class A>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, 1, L>
    adapt(P&& pointer, typename A::size_type size, O, layout_type l, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        using return_type = xtensor_adaptor<buffer_type, 1, L>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        const std::array<typename A::size_type, 1> shape{size};
        return return_type(std::move(buf), shape, l);
    }

    // shape only - buffer version
    template <layout_type L, class P, class O, class SC, class A,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, L>
    adapt(P&& pointer, typename A::size_type size, O, const SC& shape, layout_type l, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<buffer_type, N, L>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf), shape, l);
    }

    // shape and strides - buffer version
    template <class P, class O, class SC, class SS, class A,
              typename std::enable_if_t<detail::is_array<std::decay_t<SC>>::value, int>,
              typename std::enable_if_t<!std::is_same<layout_type, std::decay_t<SS>>::value, int>>
    inline xtensor_adaptor<xbuffer_adaptor<xtl::closure_type_t<P>, O, A>, detail::array_size<SC>::value, layout_type::dynamic>
    adapt(P&& pointer, typename A::size_type size, O, SC&& shape, SS&& strides, const A& alloc)
    {
        using buffer_type = xbuffer_adaptor<xtl::closure_type_t<P>, O, A>;
        constexpr std::size_t N = detail::array_size<SC>::value;
        using return_type = xtensor_adaptor<buffer_type, N, layout_type::dynamic>;
        buffer_type buf(std::forward<P>(pointer), size, alloc);
        return return_type(std::move(buf),
                           xtl::forward_sequence<typename return_type::inner_shape_type>(shape),
                           xtl::forward_sequence<typename return_type::inner_strides_type>(strides));
    }

#ifndef X_OLD_CLANG
    /**
     * Adapt a smart pointer to a typed memory block (unique_ptr or shared_ptr)
     *
     * \code{.cpp}
     * #include <xtensor/xadapt.hpp>
     * #include <xtensor/xio.hpp>
     *
     * std::shared_ptr<double> sptr(new double[8], std::default_delete<double[]>());
     * sptr.get()[2] = 321.;
     * auto xptr = adapt_smart_ptr(sptr, {4, 2});
     * xptr(1, 3) = 123.;
     * std::cout << xptr;
     * \endcode
     *
     * @param smart_ptr<T[]> a smart pointer to a memory block of T[]
     * @param shape The desired shape
     *
     * @return xtensor_adaptor for memory
     */
    template <class P, class I, std::size_t N>
    auto adapt_smart_ptr(P&& smart_ptr, const I(&shape)[N])
    {
        using buffer_adaptor = xbuffer_adaptor<decltype(smart_ptr.get()), smart_ownership,
                                               std::decay_t<P>>;
        std::array<std::size_t, N> fshape = xtl::forward_sequence<std::array<std::size_t, N>>(shape);
        return xtensor_adaptor<buffer_adaptor, N>(
            buffer_adaptor(smart_ptr.get(), compute_size(fshape), std::forward<P>(smart_ptr)),
            std::move(fshape)
        );
    }

    /**
     * Adapt a smart pointer (shared_ptr or unique_ptr)
     *
     * This function allows to automatically adapt a shared or unique pointer to
     * a given shape and operate naturally on it. Memory will be automatically
     * handled by the smart pointer implementation.
     *
     * \code{.cpp}
     * #include <xtensor/xadapt.hpp>
     * #include <xtensor/xio.hpp>
     *
     * struct Buffer {
     *     Buffer(std::vector<double>& buf) : m_buf(buf) {}
     *     ~Buffer() { std::cout << "deleted" << std::endl; }
     *     std::vector<double> m_buf;
     * };
     *
     * auto data = std::vector<double>{1,2,3,4,5,6,7,8};
     * auto shared_buf = std::make_shared<Buffer>(data);
     * auto unique_buf = std::make_unique<Buffer>(data);
     *
     * std::cout << shared_buf.use_count() << std::endl;
     * {
     *     auto obj = adapt_smart_ptr(shared_buf.get()->buf.data(),
     *                                {2, 4}, shared_buf);
     *     // Use count increased to 2
     *     std::cout << shared_buf.use_count() << std::endl;
     *     std::cout << obj << std::endl;
     * }
     * // Use count reset to 1
     * std::cout << shared_buf.use_count() << std::endl;
     *
     * {
     *     auto obj = adapt_smart_ptr(unique_buf.get()->buf.data(),
     *                                {2, 4}, std::move(unique_buf));
     *     std::cout << obj << std::endl;
     * }
     * \endcode
     *
     * @param A pointer to a typed data block (e.g. double*)
     * @param shape The desired shape
     * @param A smart pointer to move or copy, in order to manage memory
     *
     * @return xtensor_adaptor on the memory
     */
    template <class P, class I, std::size_t N, class D>
    auto adapt_smart_ptr(P&& data_ptr, const I(&shape)[N], D&& smart_ptr)
    {
        using buffer_adaptor = xbuffer_adaptor<P, smart_ownership,
                                               std::decay_t<D>>;
        std::array<std::size_t, N> fshape = xtl::forward_sequence<std::array<std::size_t, N>>(shape);
        return xtensor_adaptor<buffer_adaptor, N>(
            buffer_adaptor(data_ptr, compute_size(fshape), std::forward<D>(smart_ptr)),
            fshape
        );
    }
#endif
}

#endif
