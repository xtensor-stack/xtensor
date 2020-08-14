.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Extending xtensor
=================

``xtensor`` provides means to plug external data structures into its expression engine without
copying any data.

Adapting one-dimensional containers
-----------------------------------

You may want to use your own one-dimensional container as a backend for tensor data containers
and even for the shape or the strides. This is the simplest structure to plug into ``xtensor``.
In the following example, we define new container and adaptor types for user-specified storage and shape types.

.. code::

    // Assuming container_type and shape_type are third-party library containers
    using my_array_type = xt::xarray_container<container_type, shape_type>;
    using my_adaptor_type = xt::xarray_adaptor<container_type, shape_type>;

    // Or, working with a fixed number of dimensions
    using my_tensor_type = xt::xtensor_container<container_type, 3>;
    using my_adaptor_type = xt::xtensor_adaptor<container_type, 3>;

These new types will have all the features of the core ``xt::xtensor`` and ``xt::xarray`` types.
``xt::xarray_container`` and ``xt::xtensor_container`` embed the data container, while
``xt::xarray_adaptor`` and ``xt::xtensor_adaptor`` hold a reference on an already initialized
container.

A requirement for the user-specified containers is to provide a minimal ``std::vector``-like interface, that is:

- usual typedefs for STL sequences
- random access methods (``operator[]``, ``front``, ``back`` and ``data``)
- iterator methods (``begin``, ``end``, ``cbegin``, ``cend``)
- ``size`` and ``reshape``, ``resize`` methods

``xtensor`` does not require that the container has a contiguous memory layout, only that it
provides the aforementioned interface. In fact, the container could even be backed by a
file on the disk, a database or a binary message.

Structures that embed shape and strides
---------------------------------------

Some structures may gather data container, shape and strides, making them impossible to plug
into ``xtensor`` with the method above. This section illustrates how to adapt such structures
with the following simple example:

.. code::

    template <class T>
    struct raw_tensor
    {
        using container_type = std::vector<T>;
        using shape_type = std::vector<std::size_t>;
        container_type m_data;
        shape_type m_shape;
        shape_type m_strides;
        shape_type m_backstrides;
        static constexpr layout_type layout = layout_type::dynamic;
    };

    // This is the adaptor we need to define to plug raw_tensor in xtensor
    template <class T>
    class raw_tensor_adaptor;

Define inner types
~~~~~~~~~~~~~~~~~~

The following tells ``xtensor`` which types must be used for getting shape, strides, and data:

.. code::

    template <class T>
    struct xcontainer_inner_types<raw_tensor_adaptor<T>>
    {
        using container_type = typename raw_tensor<T>::container_type;
        using inner_shape_type = typename raw_tensor<T>::shape_type;
        using inner_strides_type = inner_shape_type;
        using inner_backstrides_type = inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = inner_shape_type;
        using backstrides_type = inner_shape_type;
        static constexpr layout_type layout = raw_tensor<T>::layout;
    };

The ``inner_XXX_type`` are the types used to store and read the shape, strides and backstrides, while the
other ones are used for reshaping. Most of the time, they will be the same; differences come when inner
types cannot be instantiated out of the box (because they are linked to python buffer for instance).

Next, bring all the iterable features with this simple definition:

.. code::

    template <class T>
    struct xiterable_inner_types<raw_tensor_adaptor<T>>
        : xcontainer_iterable_types<raw_tensor_adaptor<T>>
    {
    };

Inherit
~~~~~~~

Next step is to inherit from the ``xcontainer`` and the ``xcontainer_semantic`` classes:

.. code::

    template <class T>
    class raw_tensor_adaptor : public xcontainer<raw_tensor_adaptor<T>>,
                               public xcontainer_semantic<raw_tensor_adaptor<T>>
    {
        ...
    };

Thanks to definition of the previous structures, inheriting from ``xcontainer`` brings almost all the container
API available in the other entities of ``xtensor``, while  inheriting from ``xtensor_semantic`` brings the support
for mathematical operations.

Define semantic
~~~~~~~~~~~~~~~

``xtensor`` classes have full value semantic, so you may define the constructors specific to your structures,
and use the default copy and move constructors and assign operators. Note these last ones *must* be declared as
they are declared as ``protected`` in the base class.

.. code::

    template <class T>
    class raw_tensor_adaptor : public xcontainer<raw_tensor_adaptor<T>>,
                               public xcontainer_semantic<raw_tensor_adaptor<T>>
    {
    
    public:

        using self_type = raw_tensor_adaptor<T>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;

        // ... specific constructors here

        raw_tensor_adaptor(const raw_tensor_adaptor&) = default;
        raw_tensor_adaptor& operator=(const raw_tensor_adaptor&) = default;

        raw_tensor_adaptor(raw_tensor_adaptor&&) = default;
        raw_tensor_adaptor& operator=(raw_tensor_adaptor&&) = default;

        template <class E>
        raw_tensor_type(const xexpression<E>& e)
            : base_type()
        {
            semantic_base::assign(e);
        }

        template <class E>
        self_type& operator=(const xexpression<E>& e)
        {
            return semantic_base::operator=(e);
        }
    };
    
The last two methods are extended copy constructor and assign operator. They allow writing things like

.. code::

    using tensor_type = raw_tensor_adaptor<double>;
    tensor_type a, b, c;
    // .... init a, b and c
    tensor_type d = a + b - c;

Implement the resize methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next methods to define are the overloads of ``resize``. ``xtensor`` provides utility functions to compute
strides based on the shape and the layout, so the implementation of the ``resize`` overloads is straightforward:

.. code::

    #include "xtensor/xstrides.hpp" // for utility functions

    template <class T>
    void resize(const shape_type& shape)
    {
        if(m_shape != shape)
            resize(shape, layout::row_major);
    }

    template <class T>
    void resize(const shape_type& shape, layout l)
    {
        m_raw.m_shape = shape;
        m_raw.m_strides.resize(shape.size());
        m_raw.m_backstrides.resize(shape.size());
        size_type data_size = compute_strides(m_shape, l, m_strides, m_backstrides);
        m_raw.m_data.resize(data_size);
    }

    template <class T>
    void resize(const shape_type& shape, const strides_type& strides)
    {
        m_raw.m_shape = shape;
        m_raw.m_strides = strides;
        m_raw.m_backstrides.resize(shape.size());
        adapt_strides(m_raw.m_shape, m_raw.m_strides, m_raw.m_backstrides);
        m_raw.m_data.resize(compute_size(m_shape));
    }

Implement private accessors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``xcontainer`` assume the following methods are implemented in its inheriting class:

.. code::

    inner_shape_type& shape_impl();
    const inner_shape_type& shape_impl() const;

    inner_strides_type& strides_impl();
    const inner_strides_type& strides_impl() const;

    inner_backstrides_type& backstrides_impl();
    const inner_backstrides_type& backstrides_impl() const;

However, since ``xcontainer`` provides a public API for getting the shape and the strides,
these methods should be declared ``protected`` or ``private`` and ``xcontainer`` should
be declared as a friend class so that it can access them.

Embedding a full tensor structure
---------------------------------

You may need to plug structures that already provide n-dimensional access methods, instead of a one-dimensional
container with a strided index scheme. This section illustrates how to adapt such structures with the following (minimal) API:

.. code::

    template <class T>
    class table
    {

    public:

        using shape_type = std::vector<std::size_t>;

        const shape_type& shape() const;

        template <class... Args>
        T& operator()(Args... args);

        template <class... Args>
        const T& operator()(Args... args) const;

        template <class It>
        T& element(It first, It last);

        template <class It>
        const T& element(It first, It last) const;
    };

    // This is the adaptor we need to define to plug table in xtensor
    template <class T>
    class table_adaptor;

Define inner types
~~~~~~~~~~~~~~~~~~

The following definitions are required:

.. code::

    template <class T>
    struct xcontainer_inner_types<table_adaptor<T>>
    {
        using temporary_type = xarray<T>;
    };

    template <class T>
    struct xiterable_inner_types<table_adaptor<T>>
    {
        using inner_shape_type = typename table<T>::shape_type;
        using stepper = xindexed_stepper<table<T>, false>;
        using const_stepper = xindexed_stepper<table<T>, true>;
    };

Inheritance
~~~~~~~~~~~

Next step is to inherit from the ``xiterable`` and ``xcontainer_semantic`` classes,
and to define a bunch of typedefs.

.. code::

    template<class T>
    class table_adaptor : public xiterable<table_adaptor<T>>,
                          public xcontainer_semantic<table_adaptor<T>>
    {

    public:

        using self_type = table_adaptor<T>;
        using semantic_base = xcontainer_semantic<self_type>;

        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using inner_shape_type = typename table<T>::shape_type;
        using inner_stride_stype = inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = inner_strides_type;

        using iterable_base = xiterable<self_type>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;
    };

The iterator and stepper used here may not be the most optimal for ``table``, however they
are guaranteed to work as long as ``table`` provides an access operator based on indices.

NOTE: we inherit from ``xcontainer_semantic`` because we assume the ``table_adaptor`` class
embeds an instance of ``table``. If it took a reference on it, we would inherit from
``xadaptor_semantic`` instead.

Define semantic
~~~~~~~~~~~~~~~

As for one-dimensional containers adaptors, you must define constructors and at least declare
default copy and move constructors and assignment operators. You also must define the extended copy
constructor and assign operator.

.. code::

    template <class T>
    class table_adaptor : public xiterable<table_adaptor<T>>,
                          public xcontainer_semantic<table_adaptor<T>>
    {

    public:

        // .... typedefs
        // .... specific constructors

        table_adaptor(const table_adaptor&) = default;
        table_adaptor& operator=(const table_adaptor&) = default;

        table_adaptor(table_adaptor&&) = default;
        table_adaptor& operator=(table_adaptor&&) = default;

        template <class E>
        table_adaptor(const xexpression<E>& e)
        {
            semantic_base::assign(e);
        }

        template <class E>
        self_type& operator=(const xexpression<E>& e)
        {
            return semantic_base::operator=(e);
        }
    };
    
Implement access operators
~~~~~~~~~~~~~~~~~~~~~~~~~~

``xtensor`` requires that the following access operators are defined

.. code::

    template <class... Args>
    reference operator()(Args... args)
    {
        // Should forward to table<T>:operator()(args...)
    }

    template <class... Args>
    const_reference operator()(Args... args) const
    {
        // Should forward to table<T>::operator()(args...)
    }

    reference operator[](const xindex& index)
    {
        return element(index.cbegin(), index.cend());
    }

    const_reference operator[](const xindex& index) const
    {
        return element(index.cbegin(), index.cend());
    }

    reference operator[](size_type i)
    {
        return operator()(i);
    }

    const_reference operator[](size_type i) const
    {
        return operator()(i);
    }

    template <class It>
    reference element(It first, It last)
    {
        // Should forward to table<T>::element(first, last)
    }

    template <class It>
    const_reference element(It first, It last)
    {
        // Should forward to table<T>::element(first, last)
    }

Implement broadcast mechanic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This part is relatively straightforward:

.. code::

    size_type dimension() const
    {
        return shape().size();
    }

    const shape_type& shape() const
    {
        // Should forward to table<T>::shape()
    }

    template <class S>
    bool broadcast_shape(const S& s) const
    {
        // Available in "xtensor/xtrides.hpp"
        return xt::broadcast_shape(shape(), s);
    }

Implement resize overloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is very similar to what must be done for one-dimensional containers,
except you may ignore the layout and the strides in the implementation.
However, these overloads are still required.

Provide a stepper API
~~~~~~~~~~~~~~~~~~~~~

The last required step is to provide a stepper API, on which are built
iterators.

.. code::

    template <class ST>
    stepper stepper_begin(const ST& s)
    {
        size_type offset = s.size() - dimension();
        return stepper(this, offset);
    }

    template <class ST>
    stepper stepper_end(const ST& s)
    {
        size_type offset = s.size() - dimension();
        return stepper(this, offset, true);
    }

    template <class ST>
    const_stepper stepper_begin(const ST& s) const
    {
        size_type offset = s.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class ST>
    const_stepper stepper_end(const ST& s) const
    {
        size_type offset = s.size() - dimension();
        return const_stepper(this, offset, true);
    }

