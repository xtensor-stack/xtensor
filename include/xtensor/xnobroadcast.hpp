#ifndef XTENSOR_NOBROADCAST_HPP
#define XTENSOR_NOBROADCAST_HPP

#include "xsemantic.hpp"
#include "xfunction.hpp"
#include "xassign.hpp"

namespace xt
{
	template<class A>
	class nobroadcast_proxy 
	{
	public :

		nobroadcast_proxy(A a) noexcept;

		template<class E>
		A operator=(const xexpression<E> & e);

        template <class E>
        A operator+=(const xexpression<E>& e);

        template <class E>
        A operator-=(const xexpression<E>& e);

        template<class E>
        A operator*=(const xexpression<E>& e);

        template<class E>
        A operator/=(const xexpression<E>& e);

        template<class E>
        A operator%=(const xexpression<E>& e);

        template<class E>
        A operator&=(const xexpression<E>& e);

        template<class E>
        A operator|=(const xexpression<E>& e);

        template<class E>
        A operator^=(const xexpression<E>& e);

	private :
		A m_array;
	};


	template <class A>
	inline nobroadcast_proxy<A>::nobroadcast_proxy(A a) noexcept
		: m_array(std::forward<A>(a))
	{
	}

	template <class A>
	template <class E>
	inline A nobroadcast_proxy<A>::operator=(const xexpression<E> & e)
	{
		/*resize_no_broadcast(m_array, e);
		assign_data(m_array, e);
		std::cout << m_array << std::endl;
		return m_array;*/
        //m_array.assign(e);
        //xexpression_assigner<tag>::assign_xexpression(e1, e2);

        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner<tag>;

        base_type::resize_no_broadcast(m_array, e.derived_cast());
        base_type::assign_data(m_array, e, true);

        return m_array;
	}

    template <class A>
    template <class E>
    inline A nobroadcast_proxy<A>::operator+=(const xexpression<E>& e)
    {
        //return m_array.plus_assign(e);
        //-------------------------------------
        
        //xexpression_assigner<tag>::computed_assign(e1, e2);
        //-------------------------------------


        //xexpression_assigner<tag>::computed_assign(xexpression<A>& m_array, const xexpression<e>& e)
        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner_base<tag>;

        base_type::assign_data(m_array, m_array + e.derived_cast(), true);

        return m_array;
    }

    template <class A>
    template <class E>
    inline A nobroadcast_proxy<A>::operator-=(const xexpression<E>& e)
    {
        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner_base<tag>;

        base_type::assign_data(m_array, m_array - e.derived_cast(), true);

        return m_array;
    }

    template <class A>
    template <class E>
    inline A nobroadcast_proxy<A>::operator*=(const xexpression<E>& e)
    {
        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner_base<tag>;

        base_type::assign_data(m_array, m_array * e.derived_cast(), true);

        return m_array;
    }

    template <class A>
    template <class E>
    inline A nobroadcast_proxy<A>::operator/=(const xexpression<E>& e)
    {
        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner_base<tag>;

        base_type::assign_data(m_array, m_array / e.derived_cast(), true);

        return m_array;
    }

    template <class A>
    template <class E>
    inline A nobroadcast_proxy<A>::operator%=(const xexpression<E>& e)
    {
        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner_base<tag>;

        base_type::assign_data(m_array, m_array % e.derived_cast(), true);

        return m_array;
    }

    template <class A>
    template <class E>
    inline A nobroadcast_proxy<A>::operator&=(const xexpression<E>& e)
    {
        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner_base<tag>;

        base_type::assign_data(m_array, m_array & e.derived_cast(), true);

        return m_array;
    }

    template <class A>
    template <class E>
    inline A nobroadcast_proxy<A>::operator|=(const xexpression<E>& e)
    {
        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner_base<tag>;

        base_type::assign_data(m_array, m_array | e.derived_cast(), true);

        return m_array;
    }

    template <class A>
    template <class E>
    inline A nobroadcast_proxy<A>::operator^=(const xexpression<E>& e)
    {
        using tag = xexpression_tag_t<A, E>;
        using base_type = xexpression_assigner_base<tag>;

        base_type::assign_data(m_array, m_array ^ e.derived_cast(), true);

        return m_array;
    }

	template <class A>
	inline nobroadcast_proxy<xtl::closure_type_t<A>>
		nobroadcast(A&& a) noexcept
	{
		return nobroadcast_proxy<xtl::closure_type_t<A>>(a);
	}
}

#endif