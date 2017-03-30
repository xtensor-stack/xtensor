/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_HPP
#define XTENSOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "xcontainer.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /***********************
     * xtensor declaration *
     ***********************/

    template <class EC, std::size_t N>
    struct xcontainer_inner_types<xtensor_container<EC, N>>
    {
        using container_type = EC;
        using shape_type = std::array<typename container_type::size_type, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<EC, N>;
    };

    template <class EC, std::size_t N>
    struct xiterable_inner_types<xtensor_container<EC, N>>
        : xcontainer_iterable_types<xtensor_container<EC, N>>
    {
    };

    /**
     * @class xtensor_container
     * @brief Dense multidimensional container with tensor semantic and fixed
     * dimension.
     *
     * The xtensor_container class implements a dense multidimensional container
     * with tensor semantic and fixed dimension
     *
     * @tparam EC The type of the container holding the elements.
     * @tparam N The dimension of the container.
     * @sa xtensor
     * @sa move_reshape
     */
    template <class EC, size_t N>
    class xtensor_container : public xstrided_container<xtensor_container<EC, N>>,
                              public xcontainer_semantic<xtensor_container<EC, N>>
    {
    public:

        using self_type = xtensor_container<EC, N>;
        using base_type = xstrided_container<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        xtensor_container();
        xtensor_container(nested_initializer_list_t<value_type, N> t);
        explicit xtensor_container(const shape_type& shape, layout l = layout::row_major);
        explicit xtensor_container(const shape_type& shape, const_reference value, layout l = layout::row_major);
        explicit xtensor_container(const shape_type& shape, const strides_type& strides);
        explicit xtensor_container(const shape_type& shape, const strides_type& strides, const_reference value);

        ~xtensor_container() = default;

        xtensor_container(const xtensor_container&) = default;
        xtensor_container& operator=(const xtensor_container&) = default;

        xtensor_container(xtensor_container&&) = default;
        xtensor_container& operator=(xtensor_container&&) = default;

        template <class E>
        xtensor_container(const xexpression<E>& e);

        template <class E>
        xtensor_container& operator=(const xexpression<E>& e);

    private:

        xtensor_container(EC&& data, const shape_type& shape, const strides_type& strides);

        container_type m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        friend class xcontainer<xtensor_container<EC, N>>;

        template <class EC2, size_t N1, size_t N2>
        friend xtensor_container<EC2, N2> move_reshape(xtensor_container<EC2, N1>&& t,
                                                       const std::array<typename EC2::size_type, N2>& shape,
                                                       const std::array<typename EC2::size_type, N2>& strides);
    };

    /*****************************************
     * xtensor_container_adaptor declaration *
     *****************************************/

    template <class EC, std::size_t N>
    class xtensor_adaptor;

    template <class EC, std::size_t N>
    struct xcontainer_inner_types<xtensor_adaptor<EC, N>>
    {
        using container_type = EC;
        using shape_type = std::array<typename container_type::size_type, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xtensor_container<EC, N>;
    };

    template <class EC, std::size_t N>
    struct xiterable_inner_types<xtensor_adaptor<EC, N>>
        : xcontainer_iterable_types<xtensor_adaptor<EC, N>>
    {
    };

    /**
     * @class xtensor_adaptor
     * @brief Dense multidimensional container adaptor with tensor semantic
     * and fixed dimension.
     *
     * The xtensor_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantic and fixed dimension. It
     * is used to provide a multidimensional container semantic and a
     * tensor semantic to stl-like containers.
     *
     * @tparam EC The container type to adapt.
     * @tparam N The dimension of the adaptor.
     * @sa move_reshape
     */
    template <class EC, std::size_t N>
    class xtensor_adaptor : public xstrided_container<xtensor_adaptor<EC, N>>,
                            public xadaptor_semantic<xtensor_adaptor<EC, N>>
    {
    public:

        using self_type = xtensor_adaptor<EC, N>;
        using base_type = xstrided_container<self_type>;
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

        xtensor_adaptor(EC&& data, const shape_type& shape, const strides_type& strides);

        container_type& m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type& tmp);

        friend class xcontainer<xtensor_adaptor<EC, N>>;
        friend class xadaptor_semantic<xtensor_adaptor<EC, N>>;

        template <class EC2, size_t N1, size_t N2>
        friend xtensor_adaptor<EC2, N2> move_reshape(xtensor_adaptor<EC2, N1>&& t,
                                                     const std::array<typename EC2::size_type, N2>& shape,
                                                     const std::array<typename EC2::size_type, N2>& strides);
    };

    /****************
     * move_reshape *
     ****************/

    template <template <class, size_t> class C, class EC, size_t N1, size_t N2>
    C<EC, N2> move_reshape(C<EC, N1>& t, const std::array<typename EC::size_type, N2>& shape);

    template <template <class, size_t> class C, class EC, size_t N1, size_t N2>
    C<EC, N2> move_reshape(C<EC, N1>&& t, const std::array<typename EC::size_type, N2>& shape);

    template <template <class, size_t> class C, class EC, size_t N1, size_t N2>
    C<EC, N2> move_reshape(C<EC, N1>& t, const std::array<typename EC::size_type, N2>& shape,
                                         const std::array<typename EC::size_type, N2>& strides);

    template <class EC, size_t N1, size_t N2>
    xtensor_container<EC, N2> move_reshape(xtensor_container<EC, N1>&& t,
                                           const std::array<typename EC::size_type, N2>& shape,
                                           const std::array<typename EC::size_type, N2>& strides);

    template <class EC, size_t N1, size_t N2>
    xtensor_adaptor<EC, N2> move_reshape(xtensor_adaptor<EC, N1>&& t,
                                         const std::array<typename EC::size_type, N2>& shape,
                                         const std::array<typename EC::size_type, N2>& strides);

    /************************************
     * xtensor_container implementation *
     ************************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xtensor_container that holds 0 element.
     */
    template <class EC, std::size_t N>
    inline xtensor_container<EC, N>::xtensor_container()
        : base_type(), m_data(1, value_type())
    {
    }

    /**
     * Allocates an xtensor_container with nested initializer lists.
     */
    template <class EC, std::size_t N>
    inline xtensor_container<EC, N>::xtensor_container(nested_initializer_list_t<value_type, N> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and
     * layout.
     * @param shape the shape of the xtensor_container
     * @param l the layout of the xtensor_container
     */
    template <class EC, std::size_t N>
    inline xtensor_container<EC, N>::xtensor_container(const shape_type& shape, layout l)
        : base_type()
    {
        base_type::reshape(shape, l);
    }

    /**
     * Allocates an xtensor_container with the specified shape and layout. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xtensor_container
     * @param value the value of the elements
     * @param l the layout of the xtensor_container
     */
    template <class EC, std::size_t N>
    inline xtensor_container<EC, N>::xtensor_container(const shape_type& shape, const_reference value, layout l)
        : base_type()
    {
        base_type::reshape(shape, l);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and strides.
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     */
    template <class EC, std::size_t N>
    inline xtensor_container<EC, N>::xtensor_container(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::reshape(shape, strides);
    }

    /**
     * Allocates an uninitialized xtensor_container with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xtensor_container
     * @param strides the strides of the xtensor_container
     * @param value the value of the elements
     */
    template <class EC, std::size_t N>
    inline xtensor_container<EC, N>::xtensor_container(const shape_type& shape, const strides_type& strides, const_reference value)
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
    template <class EC, std::size_t N>
    template <class E>
    inline xtensor_container<EC, N>::xtensor_container(const xexpression<E>& e)
        : base_type()
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class EC, std::size_t N>
    template <class E>
    inline auto xtensor_container<EC, N>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, std::size_t N>
    inline xtensor_container<EC, N>::xtensor_container(EC&& data,
                                                       const shape_type& shape,
                                                       const strides_type& strides)
        : base_type(), m_data(std::move(data))
    {
        base_type::reshape(shape, strides);
    }

    template <class EC, std::size_t N>
    inline auto xtensor_container<EC, N>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, std::size_t N>
    inline auto xtensor_container<EC, N>::data_impl() const noexcept -> const container_type&
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
    template <class EC, std::size_t N>
    inline xtensor_adaptor<EC, N>::xtensor_adaptor(container_type& data)
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
    template <class EC, std::size_t N>
    inline xtensor_adaptor<EC, N>::xtensor_adaptor(container_type& data, const shape_type& shape, layout l)
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
    template <class EC, std::size_t N>
    inline xtensor_adaptor<EC, N>::xtensor_adaptor(container_type& data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, strides);
    }
    //@}

    template <class EC, std::size_t N>
    inline auto xtensor_adaptor<EC, N>::operator=(const xtensor_adaptor& rhs) -> self_type&
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class EC, std::size_t N>
    inline auto xtensor_adaptor<EC, N>::operator=(xtensor_adaptor&& rhs) -> self_type&
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
    template <class EC, std::size_t N>
    template <class E>
    inline auto xtensor_adaptor<EC, N>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class EC, std::size_t N>
    inline xtensor_adaptor<EC, N>::xtensor_adaptor(EC&& data,
                                                   const shape_type& shape,
                                                   const strides_type& strides)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, strides);
    }

    template <class EC, std::size_t N>
    inline auto xtensor_adaptor<EC, N>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class EC, std::size_t N>
    inline auto xtensor_adaptor<EC, N>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    template <class EC, std::size_t N>
    inline void xtensor_adaptor<EC, N>::assign_temporary_impl(temporary_type& tmp)
    {
        // TODO (performance improvement) : consider moving tmps shape and strides
        base_type::shape_impl() = tmp.shape();
        base_type::strides_impl() = tmp.strides();
        base_type::backstrides_impl() = tmp.backstrides();
        m_data.resize(tmp.size());
        std::copy(tmp.data().cbegin(), tmp.data().cend(), m_data.begin());
    }

    /*******************************
     * move_reshape implementation *
     *******************************/

    template <template <class, size_t> class C, class EC, size_t N1, size_t N2>
    inline C<EC, N2> move_reshape(C<EC, N1>& t, const std::array<typename EC::size_type, N2>& shape)
    {
        return move_reshape(std::move(t), shape);
    }

    template <template <class, size_t> class C, class EC, size_t N1, size_t N2>
    inline C<EC, N2> move_reshape(C<EC, N1>&& t, const std::array<typename EC::size_type, N2>& shape)
    {
        std::array<typename EC::size_type, N2> strides;
        compute_strides(shape, layout::row_major, strides);
        return move_reshape(std::move(t), shape, strides);
    }

    template <template <class, size_t> class C, class EC, size_t N1, size_t N2>
    inline C<EC, N2> move_reshape(C<EC, N1>& t, const std::array<typename EC::size_type, N2>& shape,
                                                const std::array<typename EC::size_type, N2>& strides)
    {
        return move_reshape(std::move(t), shape, strides);
    }

    /**
     * Moves the data of the specified tensor into a new tensor and reshapes it with the given
     * shape and strides. The original tensor is invalidated. This emulates the reshape of an xarray,
     * where the number of dimensions can be changed.
     *
     * @param rhs the tensor to reshape. May be passed as lvalue or rvalue reference.
     * @param shape the new shape.
     * @param strides the new strides. If omitted, the strides are computed so the new tensor
     *                has a row major layout.
     */
    template <class EC, size_t N1, size_t N2>
    inline xtensor_container<EC, N2> move_reshape(xtensor_container<EC, N1>&& rhs,
                                                  const std::array<typename EC::size_type, N2>& shape,
                                                  const std::array<typename EC::size_type, N2>& strides)
    {
        return xtensor_container<EC, N2>(std::move(rhs.m_data), shape, strides);
    }

    /**
     * Moves the data of the specified tensor adaptor into a new tensor adaptr and reshapes it with the given
     * shape and strides. The original tensor is invalidated. This emulates the reshape of an xarray,
     * where the number of dimensions can be changed.
     *
     * @param rhs the tensor adaptor to reshape. May be passed as lvalue or rvalue reference.
     * @param shape the new shape.
     * @param strides the new strides. If omitted, the strides are computed so the new tensor
     *                adaptor has a row major layout.
     */
    template <class EC, size_t N1, size_t N2>
    inline xtensor_adaptor<EC, N2> move_reshape(xtensor_adaptor<EC, N1>&& rhs,
                                                const std::array<typename EC::size_type, N2>& shape,
                                                const std::array<typename EC::size_type, N2>& strides)
    {
        return xtensor_adaptor<EC, N2>(std::move(rhs.m_data), shape, strides);
    }
}

#endif
