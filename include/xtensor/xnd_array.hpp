/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XND_ARRAY_HPP
#define XTENSOR_XND_ARRAY_HPP

#include "xarray.hpp"
#include "xio.hpp"

namespace xt
{
    namespace detail
    {
        class xnd_array_impl;

        template <class CTE>
        class xnd_expression_wrapper;
    }

    // value semantic
    class xnd_array
    {
    public:

        using implementation_type = detail::xnd_array_impl;

        xnd_array() = default;

        template <class E>
        xnd_array(E&& expr);

        xnd_array(implementation_type* holder);
        xnd_array(const xnd_array& holder);
        xnd_array(xnd_array&& holder);

        xnd_array& operator=(const xnd_array&);
        xnd_array& operator=(xnd_array&&);

        void swap(xnd_array&);

        template <class T>
        T get();

        void print() const;
        xnd_array astype(const std::string&);
        void set_type(const std::string&);

    private:

        void init_pointer_from_dtype(const std::string&);
        void check_holder() const;

        std::unique_ptr<implementation_type> p_holder;
        std::string m_dtype;
    };

    /// @cond DOXYGEN_INCLUDE_SFINAE
    void print(const xnd_array& o);
    xnd_array astype(const std::string& dtype, xnd_array& o);
    /// @endcond

    namespace detail
    {
        class xnd_array_impl  // entity semantic
        {
        public:

            xnd_array_impl(xnd_array_impl&&) = delete;

            xnd_array_impl& operator=(const xnd_array_impl&) = delete;
            xnd_array_impl& operator=(xnd_array_impl&&) = delete;

            virtual xnd_array_impl* clone() const = 0;
            virtual void print() const = 0;
            virtual void get(void*&) = 0;
            virtual ~xnd_array_impl() = default;

        protected:

            xnd_array_impl() = default;
            xnd_array_impl(const xnd_array_impl&) = default;
        };

        template <class CTE>
        class xnd_expression_wrapper : public xnd_array_impl
        {
        public:
            template <class E>
            xnd_expression_wrapper(E&& expr);

            xnd_expression_wrapper* clone() const;

            void print() const;
            void get(void*&);

            ~xnd_expression_wrapper() = default;

        protected:

            xnd_expression_wrapper(const xnd_expression_wrapper&);

        private:

            CTE m_expression;
        };
    }

    template <class E>
    inline xnd_array::xnd_array(E&& expr)
        : p_holder(new detail::xnd_expression_wrapper<E>(std::forward<E>(expr)))
    {
    }

    inline xnd_array::xnd_array(implementation_type* holder)
        : p_holder(holder)
    {
    }

    inline xnd_array::xnd_array(const xnd_array& holder)
        : p_holder(holder.p_holder->clone()),
          m_dtype(holder.m_dtype)
    {
    }

    inline xnd_array::xnd_array(xnd_array&& holder)
        : p_holder(std::move(holder.p_holder)),
          m_dtype(std::move(holder.m_dtype))
    {
    }

    inline xnd_array& xnd_array::operator=(const xnd_array& holder)
    {
        xnd_array tmp(holder);
        swap(tmp);
        return *this;
    }

    inline xnd_array& xnd_array::operator=(xnd_array&& holder)
    {
        swap(holder);
        return *this;
    }

    inline void xnd_array::swap(xnd_array& holder)
    {
        std::swap(p_holder, holder.p_holder);
        std::swap(m_dtype, holder.m_dtype);
    }

    inline void xnd_array::print() const
    {
        if (p_holder != nullptr)
        {
            p_holder->print();
        }
        std::cout << "dtype: " << m_dtype << std::endl;
    }

    template <class T>
    inline T xnd_array::get()
    {
        void* p;
        p_holder->get(p);
        T a;
        a = *(T*)p;
        return a;
    }

    inline void xnd_array::set_type(const std::string& dtype)
    {
        m_dtype = dtype;
    }

    inline xnd_array xnd_array::astype(const std::string& dtype)
    {
        if (p_holder == nullptr)
        {
            init_pointer_from_dtype(dtype);
        }
        else
        {
            xnd_array new_array;
            if (m_dtype == "int32")
            {
                auto a = get<xarray<int32_t>>();
                if (dtype == "float64")
                {
                    new_array = xt::cast<double>(a);
                    new_array.m_dtype = "float64";
                    new_array.print();
                    return new_array;
                }
            }
            else if (m_dtype == "float64")
            {
                auto a = get<xarray<double>>();
                if (dtype == "int32")
                {
                    new_array = xt::cast<int32_t>(a);
                    new_array.m_dtype = "int32";
                    return new_array;
                }
            }
        }
    }

    inline void xnd_array::init_pointer_from_dtype(const std::string& dtype)
    {
        if (dtype == "int32")
        {
            using dtype_t = int32_t;
            xt::xarray<dtype_t> empty_arr;
            p_holder.reset(new detail::xnd_expression_wrapper<xt::xarray<dtype_t>>(std::move(empty_arr)));
            m_dtype = "int32";
        }
        else if (dtype == "float64")
        {
            using dtype_t = double;
            xt::xarray<dtype_t> empty_arr;
            p_holder.reset(new detail::xnd_expression_wrapper<xt::xarray<dtype_t>>(std::move(empty_arr)));
            m_dtype = "float64";
        }
        else
        {
            XTENSOR_THROW(std::runtime_error, "Unsupported data type: " + dtype);
        }
    }

    inline void xnd_array::check_holder() const
    {
        if (p_holder == nullptr)
        {
            XTENSOR_THROW(std::runtime_error, "The holder does not contain an expression");
        }
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
    inline void print(const xnd_array& o)
    {
        o.print();
    }

    inline xnd_array astype(const std::string& dtype, xnd_array& o)
    {
        return o.astype(dtype);
    }
    /// @endcond

    namespace detail
    {
        template <class CTE>
        template <class E>
        inline xnd_expression_wrapper<CTE>::xnd_expression_wrapper(E&& expr)
            : xnd_array_impl(),
              m_expression(std::forward<E>(expr))
        {
        }

        template <class CTE>
        inline xnd_expression_wrapper<CTE>* xnd_expression_wrapper<CTE>::clone() const
        {
            return new xnd_expression_wrapper<CTE>(*this);
        }

        template <class CTE>
        inline void xnd_expression_wrapper<CTE>::print() const
        {
            std::cout << m_expression << std::endl;
        }

        template <class CTE>
        inline void xnd_expression_wrapper<CTE>::get(void*& p)
        {
            p = (void*)&m_expression;
        }

        template <class CTE>
        inline xnd_expression_wrapper<CTE>::xnd_expression_wrapper(const xnd_expression_wrapper& wrapper)
            : xnd_array_impl(),
              m_expression(wrapper.m_expression)
        {
        }
    }
}

#endif
