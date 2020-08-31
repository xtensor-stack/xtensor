#ifndef XTENSOR_ZARRAY_HPP
#define XTENSOR_ZARRAY_HPP

#include <memory>

#include <xtl/xmultimethods.hpp>

#include "xarray.hpp"

namespace xt
{

    class zarray_impl;
    
    /**********
     * zarray *
     **********/

    class zarray
    {
    public:

        using implementation_ptr = std::unique_ptr<zarray_impl>;

        zarray() = default;
        ~zarray() = default;

        template <class E>
        zarray(E&& e);

        zarray(implementation_ptr&& impl);

        zarray(const zarray& rhs);
        zarray& operator=(const zarray& rhs);

        zarray(zarray&& rhs);
        zarray& operator=(zarray&& rhs);

        void swap(zarray& rhs);

        zarray_impl& get_implementation();
        const zarray_impl& get_implementation() const;

        template <class T>
        xarray<T>& get_array();

        template <class T>
        const xarray<T>& get_array() const;

    private:

        implementation_ptr p_impl;
    };

    /***************
     * zarray_impl *
     ***************/

    class zarray_impl
    {
    public:

        using self_type = zarray_impl;

        virtual ~zarray_impl() = default;

        zarray_impl(zarray_impl&&) = delete;
        zarray_impl& operator=(const zarray_impl&) = delete;
        zarray_impl& operator=(zarray_impl&&) = delete;

        virtual self_type* clone() const = 0;

        XTL_IMPLEMENT_INDEXABLE_CLASS()

    protected:

        zarray_impl() = default;
        zarray_impl(const zarray_impl&) = default;
    };

    /****************
     * ztyped_array *
     ****************/

    template <class T>
    class ztyped_array : public zarray_impl
    {
    public:

        virtual ~ztyped_array() = default;

        virtual xarray<T>& get_array() = 0;
        virtual const xarray<T>& get_array() const = 0;

        XTL_IMPLEMENT_INDEXABLE_CLASS()

    protected:

        ztyped_array() = default;
        ztyped_array(const ztyped_array&) = default;
    };

    /***********************
     * zexpression_wrapper *
     ***********************/

    template <class CTE>
    class zexpression_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>
    {
    public:

        using self_type = zexpression_wrapper<CTE>;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;

        template <class E>
        zexpression_wrapper(E&& e);

        virtual ~zexpression_wrapper() = default;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        self_type* clone() const override;

    private:

        zexpression_wrapper(const zexpression_wrapper&) = default;

        void compute_cache() const;

        CTE m_expression;
        mutable xarray<value_type> m_cache;
        mutable bool m_cache_initialized;
    };

    /******************
     * zarray_wrapper *
     ******************/

    template <class CTE>
    class zarray_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>
    {
    public:

        using self_type = zarray_wrapper;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;

        template <class E>
        zarray_wrapper(E&& e);

        virtual ~zarray_wrapper() = default;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        self_type* clone() const override;

    private:

        zarray_wrapper(const zarray_wrapper&) = default;

        CTE m_array;
    };

    /*************************
     * zarray implementation *
     *************************/

    namespace detail
    {
        template <class E>
        struct is_xarray : std::false_type
        {
        };

        template <class T, layout_type L, class A, class SA>
        struct is_xarray<xarray<T, L, A, SA>> : std::true_type
        {
        };

        template <class E>
        struct zwrapper_builder
        {
            using closure_type = xtl::closure_type_t<E>;
            using wrapper_type = std::conditional_t<is_xarray<std::decay_t<E>>::value,
                                                    zarray_wrapper<closure_type>,
                                                    zexpression_wrapper<closure_type>>;

            template <class OE>
            static wrapper_type* run(OE&& e)
            {
                return new wrapper_type(std::forward<OE>(e));
            }
        };

        template <class E>
        inline auto build_zarray(E&& e)
        {
            return zwrapper_builder<E>::run(std::forward<E>(e));
        }
    }

    template <class E>
    inline zarray::zarray(E&& e)
        : p_impl(detail::build_zarray(std::forward<E>(e)))
    {
    }

    inline zarray::zarray(implementation_ptr&& impl)
        : p_impl(std::move(impl))
    {
    }

    inline zarray::zarray(const zarray& rhs)
        : p_impl(rhs.p_impl->clone())
    {
    }

    inline zarray& zarray::operator=(const zarray& rhs)
    {
        zarray tmp(rhs);
        swap(tmp);
        return *this;
    }

    inline zarray::zarray(zarray&& rhs)
        : p_impl(std::move(rhs.p_impl))
    {
    }

    inline zarray& zarray::operator=(zarray&& rhs)
    {
        swap(rhs);
        return *this;
    }

    inline void zarray::swap(zarray& rhs)
    {
        std::swap(p_impl, rhs.p_impl);
    }

    inline zarray_impl& zarray::get_implementation()
    {
        return *p_impl;
    }

    inline const zarray_impl& zarray::get_implementation() const
    {
        return *p_impl;
    }

    template <class T>
    inline xarray<T>& zarray::get_array()
    {
        return dynamic_cast<ztyped_array<T>*>(p_impl.get())->get_array();
    }

    template <class T>
    inline const xarray<T>& zarray::get_array() const
    {
        return dynamic_cast<const ztyped_array<T>*>(p_impl.get())->get_array();
    }
    
    /***********************
     * zexpression_wrapper *
     ***********************/
        
    template <class CTE>
    template <class E>
    inline zexpression_wrapper<CTE>::zexpression_wrapper(E&& e)
        : base_type()
        , m_expression(std::forward<E>(e))
        , m_cache()
        , m_cache_initialized(false)
    {
    }

    template <class CTE>
    inline auto zexpression_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    inline auto zexpression_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    inline auto zexpression_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }

    template <class CTE>
    inline void zexpression_wrapper<CTE>::compute_cache() const
    {
        if (!m_cache_initialized)
        {
            m_cache = m_expression;
            m_cache_initialized = true;
        }
    }

    /******************
     * zarray_wrapper *
     ******************/

    template <class CTE>
    template <class E>
    inline zarray_wrapper<CTE>::zarray_wrapper(E&& e)
        : base_type()
        , m_array(std::forward<E>(e))
    {
    }

    template <class CTE>
    inline auto zarray_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        return m_array;
    }

    template <class CTE>
    inline auto zarray_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        return m_array;
    }

    template <class CTE>
    inline auto zarray_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }
}

#endif

