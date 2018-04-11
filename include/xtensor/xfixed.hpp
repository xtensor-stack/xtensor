/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_FIXED_HPP
#define XTENSOR_FIXED_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "xcontainer.hpp"
#include "xstrides.hpp"
#include "xstorage.hpp"
#include "xsemantic.hpp"

#ifdef _MSC_VER
    #define CONSTEXPR_ENHANCED const
    #define CONSTEXPR_ENHANCED_STATIC const
    #define CONSTEXPR_RETURN
#else
    #define CONSTEXPR_ENHANCED constexpr
    #define CONSTEXPR_RETURN constexpr
    #define CONSTEXPR_ENHANCED_STATIC constexpr static
    #define HAS_CONSTEXPR_ENHANCED
#endif

namespace xt
{
    /**
     * @class fixed_shape
     * Fixed shape implementation for compile time defined arrays.
     * @sa xshape
     */
    template <std::size_t... X>
    class fixed_shape
    {
    public:

        using cast_type = const_array<std::size_t, sizeof...(X)>;

        constexpr static std::size_t size()
        {
            return sizeof...(X);
        }

        constexpr fixed_shape()
        {
        }

        constexpr operator cast_type() const
        {
            return cast_type({X...});
        }
    };
}

namespace std
{
    template <class T, size_t N>
    class tuple_size<xt::const_array<T, N>> :
        public integral_constant<size_t, N>
    {
    };
}

namespace xtl
{
    namespace detail
    {
        template <class T, std::size_t N>
        struct sequence_builder<xt::const_array<T, N>>
        {
            using sequence_type = xt::const_array<T, N>;
            using value_type = typename sequence_type::value_type;
            using size_type = typename sequence_type::size_type;

            inline static sequence_type make(size_type /*size*/, value_type /*v*/)
            {
                return sequence_type();
            }
        };
    }
}

namespace xt
{

    /**********************
     * xfixed declaration *
     **********************/

    template <class ET, class S, layout_type L, class Tag>
    class xfixed_container;

    namespace detail
    {
        /**************************************************************************************
           The following is something we can currently only dream about -- for when we drop 
           support for a lot of the old compilers (e.g. GCC 4.9, MSVC 2017 ;)

        template <class T>
        constexpr std::size_t calculate_stride(T& shape, std::size_t idx, layout_type L)
        {
            if (shape[idx] == 1)
            {
                return std::size_t(0);
            }

            std::size_t data_size = 1;
            std::size_t stride = 1;
            if (L == layout_type::row_major)
            {
                // because we have a integer sequence that counts
                // from 0 to sz - 1, we need to "invert" idx here
                idx = shape.size() - idx;
                for (std::size_t i = idx; i != 0; --i)
                {
                    stride = data_size;
                    data_size = stride * shape[i - 1];
                }
            }
            else
            {
                for (std::size_t i = 0; i < idx + 1; ++i)
                {
                    stride = data_size;
                    data_size = stride * shape[i];
                }
            }
            return stride;
        }

        *****************************************************************************************/

        template <layout_type L, std::size_t I, std::size_t... X>
        struct calculate_stride;

        template <std::size_t I, std::size_t Y, std::size_t... X>
        struct calculate_stride<layout_type::column_major, I, Y, X...>
        {
            constexpr static std::size_t value = Y *
                (calculate_stride<layout_type::column_major, I - 1, X...>::value == 0 ? 1 : calculate_stride<layout_type::column_major, I - 1, X...>::value);
        };

        template <std::size_t I, std::size_t... X>
        struct calculate_stride<layout_type::column_major, I, 1, X...>
        {
            constexpr static std::size_t value = 0;
        };

        template <std::size_t Y, std::size_t... X>
        struct calculate_stride<layout_type::column_major, 0, Y, X...>
        {
            constexpr static std::size_t value = 1;
        };

        template <std::size_t IDX, std::size_t... X>
        struct at
        {
            constexpr static std::size_t arr[sizeof...(X)] = {X...};
            constexpr static std::size_t value = arr[IDX];
        };

        template <std::size_t I, std::size_t... X>
        struct calculate_stride_row_major
        {
            constexpr static std::size_t value = (at<sizeof...(X) - I, X...>::value == 1 ? 0 : at<sizeof...(X) - I, X...>::value) *
                (calculate_stride_row_major<I - 1, X...>::value == 0 ? 
                    1 : calculate_stride_row_major<I - 1, X...>::value);
        };

        template <std::size_t... X>
        struct calculate_stride_row_major<0, X...>
        {
            constexpr static std::size_t value = 1;
        };

        template <std::size_t I, std::size_t... X>
        struct calculate_stride<layout_type::row_major, I, X...>
        {
            constexpr static std::size_t value = calculate_stride_row_major<sizeof...(X) - I - 1, X...>::value;
        };

        template <layout_type L, std::size_t... X, std::size_t... I>
        constexpr const_array<std::size_t, sizeof...(X)>
        get_strides_impl(const xt::fixed_shape<X...>& /*shape*/, std::index_sequence<I...>)
        {
            static_assert((L == layout_type::row_major) || (L == layout_type::column_major),
                          "Layout not supported for fixed array");
            return {calculate_stride<L, I, X...>::value...};
        }

        template <class T, std::size_t... I>
        constexpr T get_backstrides_impl(const T& shape, const T& strides, std::index_sequence<I...>)
        {
            return T({(strides[I] * (shape[I] - 1))...});
        }

        template <std::size_t... X>
        struct compute_size_impl;

        template <std::size_t Y, std::size_t... X>
        struct compute_size_impl<Y, X...>
        {
            constexpr static std::size_t value = Y * compute_size_impl<X...>::value;
        };

        template <std::size_t X>
        struct compute_size_impl<X>
        {
            constexpr static std::size_t value = X;
        };

        // TODO unify with constexpr compute_size when dropping MSVC 2015
        template <class T>
        struct fixed_compute_size;

        template <std::size_t... X>
        struct fixed_compute_size<xt::fixed_shape<X...>>
        {
            constexpr static std::size_t value = compute_size_impl<X...>::value;
        };
    }

    template <layout_type L, std::size_t... X>
    constexpr const_array<std::size_t, sizeof...(X)> get_strides(const fixed_shape<X...>& shape) noexcept
    {
        return detail::get_strides_impl<L>(shape, std::make_index_sequence<sizeof...(X)>{});
    }

    template <class T>
    constexpr T get_backstrides(const T& shape, const T& strides) noexcept
    {
        return detail::get_backstrides_impl(shape, strides,
                                            std::make_index_sequence<std::tuple_size<T>::value>{});
    }

    template <class ET, class S, layout_type L, class Tag>
    struct xcontainer_inner_types<xfixed_container<ET, S, L, Tag>>
    {
        using inner_shape_type = typename S::cast_type;
        using inner_strides_type = inner_shape_type;
        using backstrides_type = inner_shape_type;
        using inner_backstrides_type = backstrides_type;
        using shape_type = std::array<typename inner_shape_type::value_type,
                                      std::tuple_size<inner_shape_type>::value>;
        using strides_type = shape_type;
        using container_type = aligned_array<ET, detail::fixed_compute_size<S>::value>;
        using temporary_type = xfixed_container<ET, S, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class ET, class S, layout_type L, class Tag>
    struct xiterable_inner_types<xfixed_container<ET, S, L, Tag>>
        : xcontainer_iterable_types<xfixed_container<ET, S, L, Tag>>
    {
    };

    /**
     * @class xfixed_container
     * @brief Dense multidimensional container with tensor semantic and fixed
     * dimension.
     *
     * The xfixed_container class implements a dense multidimensional container
     * with tensor semantic and fixed dimension
     *
     * @tparam ET The type of the elements.
     * @tparam S The xshape template paramter of the container. 
     * @tparam L The layout_type of the tensor.
     * @tparam Tag The expression tag.
     * @sa xtensorf
     */
    template <class ET, class S, layout_type L, class Tag>
    class xfixed_container : public xcontainer<xfixed_container<ET, S, L, Tag>>,
                             public xcontainer_semantic<xfixed_container<ET, S, L, Tag>>
    {
    public:

        using self_type = xfixed_container<ET, S, L, Tag>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;

        using container_type = typename base_type::container_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        constexpr static std::size_t N = std::tuple_size<shape_type>::value;

        xfixed_container();
        explicit xfixed_container(value_type v);
        explicit xfixed_container(const inner_shape_type& shape, layout_type l = L);
        explicit xfixed_container(const inner_shape_type& shape, value_type v, layout_type l = L);
        xfixed_container(nested_initializer_list_t<value_type, N> t);

        ~xfixed_container() = default;

        xfixed_container(const xfixed_container&) = default;
        xfixed_container& operator=(const xfixed_container&) = default;

        xfixed_container(xfixed_container&&) = default;
        xfixed_container& operator=(xfixed_container&&) = default;

        template <class E>
        xfixed_container(const xexpression<E>& e);

        template <class E>
        xfixed_container& operator=(const xexpression<E>& e);

        template <class ST = shape_type>
        void resize(ST&& shape, bool force = false) const;

        template <class ST = shape_type>
        void reshape(ST&& shape, layout_type layout = L) const;

        template <class ST>
        bool broadcast_shape(ST& s, bool reuse_cache = false) const;

        constexpr layout_type layout() const noexcept;

    private:

        container_type m_data;

        CONSTEXPR_ENHANCED_STATIC inner_shape_type m_shape = S();
        CONSTEXPR_ENHANCED_STATIC inner_strides_type m_strides = get_strides<L>(S());
        CONSTEXPR_ENHANCED_STATIC inner_backstrides_type m_backstrides = get_backstrides(m_shape, m_strides);

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        CONSTEXPR_RETURN const inner_shape_type& shape_impl() const noexcept;
        CONSTEXPR_RETURN const inner_strides_type& strides_impl() const noexcept;
        CONSTEXPR_RETURN const inner_backstrides_type& backstrides_impl() const noexcept;

        friend class xcontainer<xfixed_container<ET, S, L, Tag>>;
    };

#ifdef HAS_CONSTEXPR_ENHANCED
    // Out of line definitions to prevent linker errors prior to C++17
    template <class ET, class S, layout_type L, class Tag>
    constexpr typename xfixed_container<ET, S, L, Tag>::inner_shape_type xfixed_container<ET, S, L, Tag>::m_shape;

    template <class ET, class S, layout_type L, class Tag>
    constexpr typename xfixed_container<ET, S, L, Tag>::inner_strides_type xfixed_container<ET, S, L, Tag>::m_strides;

    template <class ET, class S, layout_type L, class Tag>
    constexpr typename xfixed_container<ET, S, L, Tag>::inner_backstrides_type xfixed_container<ET, S, L, Tag>::m_backstrides;
#endif

    /****************************************
     * xfixed_container_adaptor declaration *
     ****************************************/

    template <class EC, class S, layout_type L, class Tag>
    class xfixed_adaptor;

    template <class EC, class S, layout_type L, class Tag>
    struct xcontainer_inner_types<xfixed_adaptor<EC, S, L, Tag>>
    {
        using container_type = std::remove_reference_t<EC>;
        using inner_shape_type = typename S::cast_type;
        using inner_strides_type = inner_shape_type;
        using backstrides_type = inner_shape_type;
        using inner_backstrides_type = backstrides_type;
        using shape_type = std::array<typename inner_shape_type::value_type,
                                      std::tuple_size<inner_shape_type>::value>;
        using strides_type = shape_type;
        using temporary_type = xfixed_container<typename container_type::value_type, S, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, class S, layout_type L, class Tag>
    struct xiterable_inner_types<xfixed_adaptor<EC, S, L, Tag>>
        : xcontainer_iterable_types<xfixed_adaptor<EC, S, L, Tag>>
    {
    };

    /**
     * @class xfixed_adaptor
     * @brief Dense multidimensional container adaptor with tensor semantic
     * and fixed dimension.
     *
     * The xfixed_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantic and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * tensor semantic to stl-like containers.
     *
     * @tparam EC The closure for the container type to adapt.
     * @tparam S The xshape template parameter for the fixed shape of the adaptor
     * @tparam L The layout_type of the adaptor.
     * @tparam Tag The expression tag.
     */
    template <class EC, class S, layout_type L, class Tag>
    class xfixed_adaptor : public xcontainer<xfixed_adaptor<EC, S, L, Tag>>,
                           public xcontainer_semantic<xfixed_adaptor<EC, S, L, Tag>>
    {
    public:

        using container_closure_type = EC;

        using self_type = xfixed_adaptor<EC, S, L, Tag>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;
        using temporary_type = typename semantic_base::temporary_type;
        using expression_tag = Tag;

        xfixed_adaptor(container_type&& data);
        xfixed_adaptor(const container_type& data);

        template <class D>
        xfixed_adaptor(D&& data);

        ~xfixed_adaptor() = default;

        xfixed_adaptor(const xfixed_adaptor&) = default;
        xfixed_adaptor& operator=(const xfixed_adaptor&);

        xfixed_adaptor(xfixed_adaptor&&) = default;
        xfixed_adaptor& operator=(xfixed_adaptor&&);
        xfixed_adaptor& operator=(temporary_type&&);

        template <class E>
        xfixed_adaptor& operator=(const xexpression<E>& e);

        constexpr layout_type layout() const noexcept;

    private:

        container_closure_type m_data;

        CONSTEXPR_ENHANCED_STATIC inner_shape_type m_shape = S();
        CONSTEXPR_ENHANCED_STATIC inner_strides_type m_strides = get_strides<L>(S());
        CONSTEXPR_ENHANCED_STATIC inner_backstrides_type m_backstrides = get_backstrides(m_shape, m_strides);

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        CONSTEXPR_RETURN const inner_shape_type& shape_impl() const noexcept;
        CONSTEXPR_RETURN const inner_strides_type& strides_impl() const noexcept;
        CONSTEXPR_RETURN const inner_backstrides_type& backstrides_impl() const noexcept;

        friend class xcontainer<xfixed_adaptor<EC, S, L, Tag>>;
    };

#ifdef HAS_CONSTEXPR_ENHANCED
    // Out of line definitions to prevent linker errors prior to C++17
    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_adaptor<EC, S, L, Tag>::inner_shape_type xfixed_adaptor<EC, S, L, Tag>::m_shape;

    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_adaptor<EC, S, L, Tag>::inner_strides_type xfixed_adaptor<EC, S, L, Tag>::m_strides;

    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_adaptor<EC, S, L, Tag>::inner_backstrides_type xfixed_adaptor<EC, S, L, Tag>::m_backstrides;
#endif

    /************************************
     * xfixed_container implementation *
     ************************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Create an uninitialized xfixed_container according to the shape template parameter.
     */
    template <class ET, class S, layout_type L, class Tag>
    inline xfixed_container<ET, S, L, Tag>::xfixed_container()
    {
    }

    /**
     * Create an xfixed_container, and initialize with the value of v.
     *
     * @param v the fill value
     */
    template <class ET, class S, layout_type L, class Tag>
    inline xfixed_container<ET, S, L, Tag>::xfixed_container(value_type v)
    {
        std::fill(this->begin(), this->end(), v);
    }

    /**
     * Create an uninitialized xfixed_container.
     * Note this function is only provided for homogenity, and the shape & layout argument is
     * disregarded (the template shape is always used).
     *
     * @param shape the shape of the xfixed_container (unused!)
     * @param l the layout_type of the xfixed_container (unused!)
     */
    template <class ET, class S, layout_type L, class Tag>
    inline xfixed_container<ET, S, L, Tag>::xfixed_container(const inner_shape_type& /*shape*/, layout_type /*l*/)
    {
    }

    /**
     * Create an xfixed_container, and initialize with the value of v.
     * Note, the shape argument to this function is only provided for homogenity,
     * and the shape argument is disregarded (the template shape is always used).
     *
     * @param shape the shape of the xfixed_container (unused!)
     * @param v the fill value
     * @param l the layout_type of the xfixed_container (unused!)
     */
    template <class ET, class S, layout_type L, class Tag>
    inline xfixed_container<ET, S, L, Tag>::xfixed_container(const inner_shape_type& /*shape*/, value_type v, layout_type /*l*/)
        : xfixed_container(v)
    {
    }

    /**
     * Allocates an xfixed_container with shape S with values from nested initializer lists.
     */
    template <class ET, class S, layout_type L, class Tag>
    inline xfixed_container<ET, S, L, Tag>::xfixed_container(nested_initializer_list_t<value_type, N> t)
    {
        L == layout_type::row_major ? nested_copy(m_data.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class ET, class S, layout_type L, class Tag>
    template <class E>
    inline xfixed_container<ET, S, L, Tag>::xfixed_container(const xexpression<E>& e)
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class ET, class S, layout_type L, class Tag>
    template <class E>
    inline auto xfixed_container<ET, S, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    /**
     * Note that the xfixed_container **cannot** be resized. Attempting to resize with a different
     * size throws an assert in debug mode.
     */
    template <class ET, class S, layout_type L, class Tag>
    template <class ST>
    inline void xfixed_container<ET, S, L, Tag>::resize(ST&& shape, bool) const
    {
        (void)(shape);  // remove unused parameter warning if XTENSOR_ASSERT undefined
        XTENSOR_ASSERT(std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size());
    }

    /**
     * Note that the xfixed_container **cannot** be reshaped to a shape different from ``S``.
     */
    template <class ET, class S, layout_type L, class Tag>
    template <class ST>
    inline void xfixed_container<ET, S, L, Tag>::reshape(ST&& shape, layout_type layout) const
    {
        if (!(std::equal(shape.begin(), shape.end(), m_shape.begin()) && shape.size() == m_shape.size() && layout == L))
        {
            throw std::runtime_error("Trying to reshape xtensorf with different shape or layout.");
        }
    }

    template <class ET, class S, layout_type L, class Tag>
    template <class ST>
    inline bool xfixed_container<ET, S, L, Tag>::broadcast_shape(ST& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    template <class ET, class S, layout_type L, class Tag>
    constexpr layout_type xfixed_container<ET, S, L, Tag>::layout() const noexcept
    {
        return base_type::static_layout;
    }

    template <class ET, class S, layout_type L, class Tag>
    inline auto xfixed_container<ET, S, L, Tag>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class ET, class S, layout_type L, class Tag>
    inline auto xfixed_container<ET, S, L, Tag>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    template <class ET, class S, layout_type L, class Tag>
    CONSTEXPR_RETURN auto xfixed_container<ET, S, L, Tag>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class ET, class S, layout_type L, class Tag>
    CONSTEXPR_RETURN auto xfixed_container<ET, S, L, Tag>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class ET, class S, layout_type L, class Tag>
    CONSTEXPR_RETURN auto xfixed_container<ET, S, L, Tag>::backstrides_impl() const noexcept -> const inner_backstrides_type&
    {
        return m_backstrides;
    }

    /*******************
     * xfixed_adaptor *
     *******************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xfixed_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class EC, class S, layout_type L, class Tag>
    inline xfixed_adaptor<EC, S, L, Tag>::xfixed_adaptor(container_type&& data)
        : base_type(), m_data(std::move(data))
    {
    }

    /**
     * Constructs an xfixed_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class EC, class S, layout_type L, class Tag>
    inline xfixed_adaptor<EC, S, L, Tag>::xfixed_adaptor(const container_type& data)
        : base_type(), m_data(data)
    {
    }

    /**
     * Constructs an xfixed_adaptor of the given stl-like container,
     * with the specified shape and layout_type.
     * @param data the container to adapt
     * @param shape the shape of the xfixed_adaptor
     * @param l the layout_type of the xfixed_adaptor
     */
    template <class EC, class S, layout_type L, class Tag>
    template <class D>
    inline xfixed_adaptor<EC, S, L, Tag>::xfixed_adaptor(D&& data)
        : base_type(), m_data(std::forward<D>(data))
    {
    }
    //@}

    template <class EC, class S, layout_type L, class Tag>
    inline auto xfixed_adaptor<EC, S, L, Tag>::operator=(const xfixed_adaptor& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class EC, class S, layout_type L, class Tag>
    inline auto xfixed_adaptor<EC, S, L, Tag>::operator=(xfixed_adaptor&& rhs) -> self_type&
    {
        base_type::operator=(std::move(rhs));
        m_data = rhs.m_data;
        return *this;
    }

    template <class EC, class S, layout_type L, class Tag>
    inline auto xfixed_adaptor<EC, S, L, Tag>::operator=(temporary_type&& rhs) -> self_type&
    {
        m_data = xtl::forward_sequence<container_type>(std::move(rhs.data()));
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class EC, class S, layout_type L, class Tag>
    template <class E>
    inline auto xfixed_adaptor<EC, S, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, class S, layout_type L, class Tag>
    inline auto xfixed_adaptor<EC, S, L, Tag>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, class S, layout_type L, class Tag>
    inline auto xfixed_adaptor<EC, S, L, Tag>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    template <class EC, class S, layout_type L, class Tag>
    constexpr layout_type xfixed_adaptor<EC, S, L, Tag>::layout() const noexcept
    {
        return base_type::static_layout;
    }

    template <class EC, class S, layout_type L, class Tag>
    CONSTEXPR_RETURN auto xfixed_adaptor<EC, S, L, Tag>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class EC, class S, layout_type L, class Tag>
    CONSTEXPR_RETURN auto xfixed_adaptor<EC, S, L, Tag>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class EC, class S, layout_type L, class Tag>
    CONSTEXPR_RETURN auto xfixed_adaptor<EC, S, L, Tag>::backstrides_impl() const noexcept -> const inner_backstrides_type&
    {
        return m_backstrides;
    }
}

#undef CONSTEXPR_ENHANCED
#undef CONSTEXPR_RETURN
#undef CONSTEXPR_ENHANCED_STATIC
#undef HAS_CONSTEXPR_ENHANCED

#endif