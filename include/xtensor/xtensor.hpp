/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_HPP
#define XTENSOR_HPP

#include <cstddef>
#include <utility>
#include <array>
#include <vector>
#include <algorithm>

#include "xcontainer.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /***********************
     * xtensor declaration *
     ***********************/

    template <class T, std::size_t N, class A>
    struct xcontainer_inner_types<xtensor<T, N, A>>
    {
        using container_type = std::vector<T, A>;
        using shape_type = std::array<typename container_type::size_type, N>;
        using strides_type = shape_type;
        using temporary_type = xtensor<T, N, A>;
    };

    /**
     * @class xtensor
     * @brief Dense multidimensional container with tensor
     * semantic and fixed dimension.
     *
     * The xtensor class implements a dense multidimensional container
     * with tensor semantic and fixed dimension
     *
     * @tparam T The type of objects stored in the container.
     * @tparam N The dimension of the container.
     */
    template <class T, size_t N, class A>
    class xtensor : public xcontainer<xtensor<T, N, A>>,
                    public xcontainer_semantic<xtensor<T, N, A>>
    {

    public:

        using self_type = xtensor<T, N, A>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        xtensor();
        explicit xtensor(const shape_type& shape, layout l = layout::row_major);
        explicit xtensor(const shape_type& shape, const_reference value, layout l = layout::row_major);
        explicit xtensor(const shape_type& shape, const strides_type& strides);
        explicit xtensor(const shape_type& shape, const strides_type& strides, const_reference value);

        ~xtensor() = default;

        xtensor(const xtensor&) = default;
        xtensor& operator=(const xtensor&) = default;

        xtensor(xtensor&&) = default;
        xtensor& operator=(xtensor&&) = default;

        template <class E>
        xtensor(const xexpression<E>& e);

        template <class E>
        xtensor& operator=(const xexpression<E>& e);

    private:

        container_type m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        friend class xcontainer<xtensor<T, N, A>>;
    };

    /*******************************
     * xtensor_adaptor declaration *
     *******************************/

    template <class C, std::size_t N, class A = std::allocator<typename C::value_type>>
    class xtensor_adaptor;

    template <class C, std::size_t N, class A>
    struct xcontainer_inner_types<xtensor_adaptor<C, N, A>>
    {
        using container_type = C;
        using shape_type = std::array<typename container_type::size_type, N>;
        using strides_type = shape_type;
        using temporary_type = xtensor<typename C::value_type, N, A>;
    };

    /**
     * @class xtensor_adaptor
     * @brief Dense multidimensional container adaptor with
     * tensor semantic and fixed dimension.
     *
     * The xtensor_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantic and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * tensor semantic to stl-like containers.
     *
     * @tparam C The container type to adapt.
     * @tparam N The dimension of the adaptor.
     */
    template <class C, std::size_t N, class A>
    class xtensor_adaptor : public xcontainer<xtensor_adaptor<C, N, A>>,
                            public xadaptor_semantic<xtensor_adaptor<C, N, A>>
    {

    public:

        using self_type = xtensor_adaptor<C, N, A>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xadaptor_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        xtensor_adaptor(container_type& data);
        xtensor_adaptor(container_type& data, const shape_type& shape, layout l = layout::row_major);
        xtensor_adaptor(container_type& data, const shape_type& shape, const strides_type& strides);

        ~xtensor_adaptor() = default;

        xtensor_adaptor(const xtensor_adaptor&) = default;
        xtensor_adaptor& operator=(const xtensor_adaptor&);

        xtensor_adaptor(xtensor_adaptor&&) = default;
        xtensor_adaptor& operator=(xtensor_adaptor&&);

        template <class E>
        xtensor_adaptor& operator=(const xexpression<E>& e);

    private:

        container_type& m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type& tmp);

        friend class xcontainer<xtensor_adaptor<C, N, A>>;
        friend class xadaptor_semantic<xtensor_adaptor<C, N, A>>;
    };

    /**************************
     * xtensor implementation *
     **************************/

     /**
      * @name Constructors
      */
     //@{
     /**
      * Allocates an uninitialized xtensor that holds 0 element.
      */
    template <class T, std::size_t N, class A>
    inline xtensor<T, N, A>::xtensor()
        : base_type(), m_data(1, value_type())
    {
    }

    /**
     * Allocates an uninitialized xtensor with the specified shape and
     * layout.
     * @param shape the shape of the xtensor
     * @param l the layout of the xtensor
     */
    template <class T, std::size_t N, class A>
    inline xtensor<T, N, A>::xtensor(const shape_type& shape, layout l)
        : base_type()
    {
        base_type::reshape(shape, l);
    }

    /**
     * Allocates an xtensor with the specified shape and layout. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xtensor
     * @param value the value of the elements
     * @param l the layout of the xtensor
     */
    template <class T, std::size_t N, class A>
    inline xtensor<T, N, A>::xtensor(const shape_type& shape, const_reference value, layout l)
        : base_type()
    {
        base_type::reshape(shape, l);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized xtensor with the specified shape and strides.
     * @param shape the shape of the xtensor
     * @param strides the strides of the xtensor
     */
    template <class T, std::size_t N, class A>
    inline xtensor<T, N, A>::xtensor(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::reshape(shape, strides);
    }

    /**
     * Allocates an uninitialized xtensor with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xtensor
     * @param strides the strides of the xtensor
     * @param value the value of the elements
     */
    template <class T, std::size_t N, class A>
    inline xtensor<T, N, A>::xtensor(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        base_type::reshape(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class T, std::size_t N, class A>
    template <class E>
    inline xtensor<T, N, A>::xtensor(const xexpression<E>& e)
        : base_type()
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class T, std::size_t N, class A>
    template <class E>
    inline auto xtensor<T, N, A>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class T, std::size_t N, class A>
    inline auto xtensor<T, N, A>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class T, std::size_t N, class A>
    inline auto xtensor<T, N, A>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    /*******************
     * xtensor_adaptor *
     *******************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xtensor_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class C, std::size_t N, class A>
    inline xtensor_adaptor<C, N, A>::xtensor_adaptor(container_type& data)
        : base_type(), m_data(data)
    {
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and layout.
     * @param data the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param l the layout of the xtensor_adaptor
     */
    template <class C, std::size_t N, class A>
    inline xtensor_adaptor<C, N, A>::xtensor_adaptor(container_type& data, const shape_type& shape, layout l)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, l);
    }

    /**
     * Constructs an xtensor_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param data the container to adapt
     * @param shape the shape of the xtensor_adaptor
     * @param strides the strides of the xtensor_adaptor
     */
    template <class C, std::size_t N, class A>
    inline xtensor_adaptor<C, N, A>::xtensor_adaptor(container_type& data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, strides);
    }
    //@}

    template <class C, std::size_t N, class A>
    inline xtensor_adaptor<C, N, A>& xtensor_adaptor<C, N, A>::operator=(const xtensor_adaptor& rhs)
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class C, std::size_t N, class A>
    inline xtensor_adaptor<C, N, A>& xtensor_adaptor<C, N, A>::operator=(xtensor_adaptor&& rhs)
    {
        base_type::operator=(std::move(rhs));
        m_data = rhs.m_data;
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class C, std::size_t N, class A>
    template <class E>
    inline auto xtensor_adaptor<C, N, A>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class C, std::size_t N, class A>
    inline auto xtensor_adaptor<C, N, A>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class C, std::size_t N, class A>
    inline auto xtensor_adaptor<C, N, A>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    template <class C, std::size_t N, class A>
    inline void xtensor_adaptor<C, N, A>::assign_temporary_impl(temporary_type& tmp)
    {
        // TODO (performance improvement) : consider moving tmps shape and strides
        base_type::get_shape() = tmp.shape();
        base_type::get_strides() = tmp.strides();
        base_type::get_backstrides() = tmp.backstrides();
        m_data.resize(tmp.size());
        std::copy(tmp.data().cbegin(), tmp.data().cend(), m_data.begin());
    }
}

#endif
