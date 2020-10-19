.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _xsimd:

xsimd
=====

Alignment of fixed-size members
-------------------------------

.. note::

    If you are using ``C++ >= 17`` you should not have to worry about this.

If you define a structure having members of fixed-size xtensor types,
you must ensure that the buffers properly aligned.
For this you can use the macro ``XTENSOR_SELECT_ALIGN`` available in
``xtensor/xtensor_config.hpp``.
Consider the following example:

.. code-block:: cpp

    template <typename T>
    class alignas(XTENSOR_SELECT_ALIGN(T)) Foo
    {
    public:

        using allocator_type = std::conditional_t<XTENSOR_SELECT_ALIGN(T) != 0,
                                                  xt_simd::aligned_allocator<T, XTENSOR_SELECT_ALIGN(T)>,
                                                  std::allocator<T>>;

        Foo(T fac) : m_fac(fac)
        {
            m_bar.fill(fac);
        }

        auto get() const
        {
            return m_bar;
        }

    private:

        xt::xtensor_fixed<T, xt::xshape<10, 10>> m_bar;
        T m_fac;
    };

Whereby it is important to store the fixed-sized xtensor type (in this case ``xt::xtensor_fixed<T, xt::xshape<10, 10>>``) as first member.
