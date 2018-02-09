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
#include "xsemantic.hpp"

namespace xt
{

    // Fixed shape
    template <class T, std::size_t N>
    struct const_array
    {
        using size_type = std::size_t;
        using value_type = T;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;

        using iterator = pointer;
        using const_iterator = const_pointer;

        using reverse_iterator = std::reverse_iterator<const_iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        constexpr const_reference operator[](std::size_t idx) const
        {
            return m_data[idx];
        }

        constexpr const_iterator begin() const
        {
            return cbegin();
        }

        constexpr const_iterator end() const
        {
            return cend();
        }

        constexpr const_iterator cbegin() const
        {
            return m_data;
        }

        constexpr const_iterator cend() const
        {
            return m_data + N;
        }

        // TODO make constexpr once C++17 arrives
        reverse_iterator rbegin() const
        {
            return crbegin();
        }

        reverse_iterator rend() const
        {
            return crend();
        }

        const_reverse_iterator crbegin() const
        {
            return const_reverse_iterator(end());
        }

        const_reverse_iterator crend() const
        {
            return const_reverse_iterator(begin());
        }

        constexpr const_reference front() const
        {
            return m_data[0];
        }

        constexpr const_reference back() const
        {
            return m_data[size() - 1];
        }

        constexpr size_type size() const
        {
            return N;
        }

        T m_data[N ? N : 1];
    };

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

    template <class EC, class S, layout_type L, class Tag>
    class xfixed_container;

    namespace detail
    {
        // template <class T>
        // constexpr std::size_t calculate_stride(T& shape, std::size_t idx, layout_type L)
        // {
        //     if (shape[idx] == 1)
        //     {
        //         return std::size_t(0);
        //     }

        //     std::size_t data_size = 1;
        //     std::size_t stride = 1;
        //     if (L == layout_type::row_major)
        //     {
        //         // because we have a integer sequence that counts
        //         // from 0 to sz - 1, we need to "invert" idx here
        //         idx = shape.size() - idx;
        //         for (std::size_t i = idx; i != 0; --i)
        //         {
        //             stride = data_size;
        //             data_size = stride * shape[i - 1];
        //         }
        //     }
        //     else
        //     {
        //         for (std::size_t i = 0; i < idx + 1; ++i)
        //         {
        //             stride = data_size;
        //             data_size = stride * shape[i];
        //         }
        //     }
        //     return stride;
        // }

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
            constexpr static std::size_t value = (at<sizeof...(X) - I - 1, X...>::value == 1 ? 0 : at<sizeof...(X) - I - 1, X...>::value) * 
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
        constexpr xt::const_array<std::size_t, sizeof...(X)> get_strides_impl(const xt::fixed_shape<X...>& shape, std::index_sequence<I...>)
        {
            static_assert(((L == layout_type::row_major) || (L == layout_type::column_major)), 
                          "Layout not supported for fixed objects");
            return xt::const_array<std::size_t, sizeof...(X)>{calculate_stride<L, I, X...>::value...};
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

        template <std::size_t... X>
        constexpr std::size_t compute_size(const fixed_shape<X...>& /*s*/)
        {
            return compute_size_impl<X...>::value;
        }
    }

    // returns strides & backstrides in a tuple
    template <layout_type L, std::size_t... X>
    constexpr const_array<std::size_t, sizeof...(X)> get_strides(const fixed_shape<X...>& shape)
    {
        // constexpr std::size_t sz = std::tuple_size<T>::value;
        // constexpr std::size_t sz = fixed_shape<X...>::size();
        // auto index_sequence = ;
        return detail::get_strides_impl<L>(shape, std::make_index_sequence<sizeof...(X)>{});
    }

    template <class T>
    constexpr T get_backstrides(const T& shape, const T& strides)
    {
        // constexpr std::size_t sz = std::tuple_size<T>::value;
        // auto index_sequence = ;
        return detail::get_backstrides_impl(shape, strides, std::make_index_sequence<shape.size()>{});
    }

    template <class EC, class S, layout_type L, class Tag>
    struct xcontainer_inner_types<xfixed_container<EC, S, L, Tag>>
    {
        using inner_shape_type = typename S::cast_type;
        using inner_strides_type = inner_shape_type;
        using backstrides_type = inner_shape_type;
        using inner_backstrides_type = backstrides_type;
        using shape_type = std::array<typename inner_shape_type::value_type,
                                      std::tuple_size<inner_shape_type>::value>;
        using strides_type = shape_type;
        using container_type = std::array<EC, detail::compute_size(S())>;
        using temporary_type = xfixed_container<EC, S, L, Tag>;
        static constexpr layout_type layout = L;
    };

    template <class EC, class S, layout_type L, class Tag>
    struct xiterable_inner_types<xfixed_container<EC, S, L, Tag>>
        : xcontainer_iterable_types<xfixed_container<EC, S, L, Tag>>
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
     * @tparam EC The type of the container holding the elements.
     * @tparam S The shape of the container.
     * @tparam L The layout_type of the tensor.
     * @tparam Tag The expression tag.
     * @sa xtensor
     */
    template <class EC, class S, layout_type L, class Tag>
    class xfixed_container : public xcontainer<xfixed_container<EC, S, L, Tag>>,
                             public xcontainer_semantic<xfixed_container<EC, S, L, Tag>>
    {
    public:

        using self_type = xfixed_container<EC, S, L, Tag>;
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

        template <class ST>
        void reshape(ST&& s)
        {
            (void)(s); // remove unused parameter warning if XTENSOR_ASSERT undefined
            XTENSOR_ASSERT(std::equal(s.begin(), s.end(), m_shape.begin()) && s.size() == m_shape.size());
        }

        template <class ST>
        bool broadcast_shape(ST& s) const
        {
            return xt::broadcast_shape(m_shape, s);
        }

    private:

        container_type m_data;

        constexpr static inner_shape_type m_shape = S();
        constexpr static inner_strides_type m_strides = get_strides<L>(S());
        constexpr static inner_backstrides_type m_backstrides = get_backstrides(m_shape, m_strides);

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        constexpr const inner_shape_type& shape_impl() const noexcept;
        constexpr const inner_strides_type& strides_impl() const noexcept;
        constexpr const inner_backstrides_type& backstrides_impl() const noexcept;

        friend class xcontainer<xfixed_container<EC, S, L, Tag>>;
    };

    // Out of line definitions to prevent linker errors prior to C++17
    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_container<EC, S, L, Tag>::inner_shape_type xfixed_container<EC, S, L, Tag>::m_shape;

    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_container<EC, S, L, Tag>::inner_strides_type xfixed_container<EC, S, L, Tag>::m_strides;

    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_container<EC, S, L, Tag>::inner_backstrides_type xfixed_container<EC, S, L, Tag>::m_backstrides;

    /****************************************
     * xfixed_container_adaptor declaration *
     ****************************************/

    template <class EC, class S, layout_type L, class Tag>
    class xfixed_adaptor;

    template <class EC, class S, layout_type L, class Tag>
    struct xcontainer_inner_types<xfixed_adaptor<EC, S, L, Tag>>
    {
        using container_type = std::remove_reference_t<EC>;
        using shape_type = S;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xfixed_container<container_type, S, L, Tag>;
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
     * @tparam N The dimension of the adaptor.
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

    private:

        container_closure_type m_data;

        constexpr static inner_shape_type m_shape = S();
        constexpr static inner_strides_type m_strides = get_strides<L>(m_shape);
        constexpr static inner_backstrides_type m_backstrides = get_backstrides(m_shape, m_strides);

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        constexpr const inner_shape_type& shape_impl() const noexcept;
        constexpr const inner_strides_type& strides_impl() const noexcept;
        constexpr const inner_backstrides_type&  backstrides_impl() const noexcept;

        friend class xcontainer<xfixed_adaptor<EC, S, L, Tag>>;
    };

    // Out of line definitions to prevent linker errors prior to C++17
    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_adaptor<EC, S, L, Tag>::inner_shape_type xfixed_adaptor<EC, S, L, Tag>::m_shape;

    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_adaptor<EC, S, L, Tag>::inner_strides_type xfixed_adaptor<EC, S, L, Tag>::m_strides;

    template <class EC, class S, layout_type L, class Tag>
    constexpr typename xfixed_adaptor<EC, S, L, Tag>::inner_backstrides_type xfixed_adaptor<EC, S, L, Tag>::m_backstrides;

    /************************************
     * xfixed_container implementation *
     ************************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xfixed_container that holds 0 element.
     */
    template <class EC, class S, layout_type L, class Tag>
    inline xfixed_container<EC, S, L, Tag>::xfixed_container()
    {
    }

    /**
     * Allocates an xfixed_container with nested initializer lists.
     */
    template <class EC, class S, layout_type L, class Tag>
    inline xfixed_container<EC, S, L, Tag>::xfixed_container(nested_initializer_list_t<value_type, N> t)
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
    template <class EC, class S, layout_type L, class Tag>
    template <class E>
    inline xfixed_container<EC, S, L, Tag>::xfixed_container(const xexpression<E>& e)
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class EC, class S, layout_type L, class Tag>
    template <class E>
    inline auto xfixed_container<EC, S, L, Tag>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, class S, layout_type L, class Tag>
    inline auto xfixed_container<EC, S, L, Tag>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, class S, layout_type L, class Tag>
    inline auto xfixed_container<EC, S, L, Tag>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    template <class EC, class S, layout_type L, class Tag>
    constexpr auto xfixed_container<EC, S, L, Tag>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class EC, class S, layout_type L, class Tag>
    constexpr auto xfixed_container<EC, S, L, Tag>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class EC, class S, layout_type L, class Tag>
    constexpr auto xfixed_container<EC, S, L, Tag>::backstrides_impl() const noexcept -> const inner_backstrides_type&
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
        // base_type::reshape(shape, l);
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
        m_data = std::move(rhs.data());
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
    constexpr auto xfixed_adaptor<EC, S, L, Tag>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class EC, class S, layout_type L, class Tag>
    constexpr auto xfixed_adaptor<EC, S, L, Tag>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class EC, class S, layout_type L, class Tag>
    constexpr auto xfixed_adaptor<EC, S, L, Tag>::backstrides_impl() const noexcept -> const inner_backstrides_type&
    {
        return m_backstrides;
    }
}

#endif
