/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XOPTIONAL_ASSEMBLY_HPP
#define XOPTIONAL_ASSEMBLY_HPP

#include "xconcepts.hpp"
#include "xoptional.hpp"
#include "xoptional_assembly_base.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /**********************************
     * xoptional_assembly declaration *
     **********************************/

    template <class VE, class FE>
    class xoptional_assembly;

    template <class VE, class FE>
    struct xcontainer_inner_types<xoptional_assembly<VE, FE>>
    {
        using value_expression = VE;
        using flag_expression = FE;
        using temporary_type = xoptional_assembly<VE, FE>;
    };

    template <class VE, class FE>
    struct xiterable_inner_types<xoptional_assembly<VE, FE>>
    {
        using assembly_type = xoptional_assembly<VE, FE>;
        using inner_shape_type = typename VE::inner_shape_type;
        using stepper = xoptional_assembly_stepper<assembly_type, false>;
        using const_stepper = xoptional_assembly_stepper<assembly_type, true>;
    };

    template <class VE, class FE>
    class xoptional_assembly : public xoptional_assembly_base<xoptional_assembly<VE, FE>>,
                               public xcontainer_semantic<xoptional_assembly<VE, FE>>
    {
    public:

        using self_type = xoptional_assembly<VE, FE>;
        using base_type = xoptional_assembly_base<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using value_expression = typename base_type::value_expression;
        using flag_expression = typename base_type::flag_expression;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        xoptional_assembly() = default;
        explicit xoptional_assembly(const shape_type& shape, layout_type l = base_type::static_layout);
        explicit xoptional_assembly(const shape_type& shape, const value_type& value, layout_type l = base_type::static_layout);
        explicit xoptional_assembly(const shape_type& shape, const strides_type& strides);
        explicit xoptional_assembly(const shape_type& shape, const strides_type& strides, const value_type& value);

        xoptional_assembly(const value_type& value);
        xoptional_assembly(nested_initializer_list_t<value_type, 1> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 2> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 3> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 4> t);
        xoptional_assembly(nested_initializer_list_t<value_type, 5> t);

        xoptional_assembly(const VE& ve);
        xoptional_assembly(VE&& ve);

        template <class OVE, class OFE, XTENSOR_REQUIRE<is_xexpression<OVE>::value && is_xexpression<OFE>::value>>
        xoptional_assembly(OVE&& ove, OFE&& ofe);

        template <class S = shape_type>
        static xoptional_assembly from_shape(S&& s);

        ~xoptional_assembly() = default;

        xoptional_assembly(const xoptional_assembly&) = default;
        xoptional_assembly& operator=(const xoptional_assembly&) = default;

        xoptional_assembly(xoptional_assembly&&) = default;
        xoptional_assembly& operator=(xoptional_assembly&&) = default;

        template <class E>
        xoptional_assembly(const xexpression<E>& e);

        template <class E>
        xoptional_assembly& operator=(const xexpression<E>& e);

    private:

        value_expression& value_impl() noexcept;
        const value_expression& value_impl() const noexcept;

        flag_expression& has_value_impl() noexcept;
        const flag_expression& has_value_impl() const noexcept;

        value_expression m_value;
        flag_expression m_has_value;

        friend class xoptional_assembly_base<xoptional_assembly<VE, FE>>;
    };

    /*************************************
     * xoptional_assembly implementation *
     *************************************/

    namespace detail
    {
        template <class T, class S>
        inline void nested_optional_copy(T&& iter, const S& s)
        {
            iter->value() = s.value();
            iter->has_value() = s.has_value();
            ++iter;
        }

        template <class T, class S>
        inline void nested_optional_copy(T&& iter, std::initializer_list<S> s)
        {
            for (auto it = s.begin(); it != s.end(); ++it)
            {
                nested_optional_copy(std::forward<T>(iter), *it);
            }
        }
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const shape_type& shape, layout_type l)
        : m_value(shape, l), m_has_value(shape, l)
    {
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const shape_type& shape, const value_type& value, layout_type l)
        : m_value(shape, value.value(), l), m_has_value(shape, value.has_value(), l)
    {
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const shape_type& shape, const strides_type& strides)
        : m_value(shape, strides), m_has_value(shape, strides)
    {
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const shape_type& shape, const strides_type& strides, const value_type& value)
        : m_value(shape, strides, value.value()), m_has_value(shape, strides, value.has_value())
    {
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const value_type& value)
        : m_value(value.value()), m_has_value(value.has_value())
    {
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 1> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
            : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 2> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
            : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 3> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
            : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 4> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
            : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(nested_initializer_list_t<value_type, 5> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t));
        bool condition = VE::static_layout == layout_type::row_major && FE::static_layout == layout_type::row_major;
        condition ? detail::nested_optional_copy(this->storage_begin(), t)
            : nested_copy(this->template begin<layout_type::row_major>(), t);
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const VE& ve)
        : m_value(ve), m_has_value(ve.shape(), true, ve.layout())
    {
    }

    template <class VE, class FE>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(VE&& ve)
        : m_value(std::move(ve))
    {
        m_has_value.reshape(ve.shape(), true, ve.layout());
    }

    template <class VE, class FE>
    template <class OVE, class OFE, typename>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(OVE&& ove, OFE&& ofe)
        : m_value(std::forward<OVE>(ove)), m_has_value(std::forward<OFE>(ofe))
    {
    }

    template <class VE, class FE>
    template <class S>
    inline xoptional_assembly<VE, FE> xoptional_assembly<VE, FE>::from_shape(S&& s)
    {
        shape_type shape = xtl::forward_sequence<shape_type>(s);
        return self_type(shape);
    }

    template <class VE, class FE>
    template <class E>
    inline xoptional_assembly<VE, FE>::xoptional_assembly(const xexpression<E>& e)
        : base_type()
    {
        semantic_base::assign(e);
    }

    template <class VE, class FE>
    template <class E>
    inline auto xoptional_assembly<VE, FE>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::value_impl() noexcept -> value_expression&
    {
        return m_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::value_impl() const noexcept -> const value_expression&
    {
        return m_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::has_value_impl() noexcept -> flag_expression&
    {
        return m_has_value;
    }

    template <class VE, class FE>
    inline auto xoptional_assembly<VE, FE>::has_value_impl() const noexcept -> const flag_expression&
    {
        return m_has_value;
    }
}

#endif
