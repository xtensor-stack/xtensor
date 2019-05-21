#ifndef XTENSOR_NOBROADCAST_HPP
#define XTENSOR_NOBROADCAST_HPP

#include "xsemantic.hpp"

namespace xt
{
	template<class A>
	class nobroadcast_proxy
	{
	public :
		nobroadcast_proxy(A a) noexcept;

		template<class E>
		A operator=(const xexpression<E> & e);

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
		std::cout << m_array << std::endl;*/
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