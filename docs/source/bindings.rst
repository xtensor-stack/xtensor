.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Designing language bindings with xtensor
========================================

xtensor and its :ref:`related-projects` make it easy to implement a feature once in C++ and expose it
to the main languages of data science, such as Python, Julia and R with little extra work. Although,
if that sounds simple in principle, difficulties may appear when it comes to define the API of the
C++ library. 
The following illustrates the different options we have with the case of a single function ``compute``
that must be callable from all the languages.

Generic API
-----------

Since the xtensor bindings provide different container types for holding tensors (pytensor, rtensor
and jltensor), if we want our function to be callable from all the languages, it must accept a generic
argument:

.. code::

    template <class E>
    void compute(E&& e);

However, this is a bit too generic and we may want to enforce that this function only accepts xtensor
arguments. Since all xtensor containers inherit from the “xexpression” CRTP base class, we can easily
express that constraint with the following signature:

.. code::

    template <class E>
    void compute(const xexpression<E>& e)
    {
        // Now the implementation must use e() instead of e
    }

Notice that with this change, we lose the ability to call the function with non-constant references or
rvalue references. If we want them back, we need to add the following overloads:

.. code::

    template <class E>
    void compute(xexpression<E>& e);
    
    template <class E>
    void compute(xexpression<E>&& e);

In the following we assume that the constant reference overload is enough. We can now expose the compute
function to the other languages, let’s illustrate this with Python bindings:

.. code::

    PYBIND11_MODULE(pymod, m)
    {
        xt::import_numpy();

        m.def("compute", &compute<pytensor<double, 2>>);
    }

Full qualified API
------------------

Accepting any kind of expression can still be too permissive; assume we want to restrict this function to
2-dimensional tensor containers only. In that case, a solution is to provide an API function that forwards
the call to a common generic implementation:

.. code::

    namespace detail
    {
        template <class E>
        void compute_impl(E&&);
    }

    template <class T>
    void compute(const xtensor<T, 2>& t)
    {
        detail::compute_impl(t);
    }

Exposing it to the Python is just as simple:

.. code::

    template <class T>
    void compute(const pytensor<T, 2>& t)
    {
        detail::compute_impl(t);
    }

    PYBIND11_MODULE(pymod, m)
    {
        xt::import_numpy();

        m.def("compute", &compute<double>);
    }

Although this solution is really simple, it requires writing four additional functions for the API. Besides,
if later, you decide to support array containers, you need to add four more functions. Therefore this solution
should be considered for libraries with a small number of functions to expose, and whose APIs are unlikely to
change in the future.

Container selection
-------------------

A way to keep the restriction on the parameter type while limiting the required amount of typing in the bindings
is to rely on additional structures that will “select” the right type for us.

The idea is to define a structure for selecting the type of containers (tensor, array) and a structure to select
the library implementation of that container (xtensor, pytensor in the case of a tensor container):

.. code::

    // library container selector
    struct xtensor_c
    {
    };
    
    // container selector, must be specialized for each
    // library container selector
    template <class C, class T, std::size_t N>
    struct tensor_container;

    // Specialization for xtensor library (or C++)
    template <class T, std::size_t N>
    struct tensor_container<xtensor_c, T, N>
    {
        using type = xt::xtensor<T, N>;
    };

    template <class C, class T, std::size_t N>
    using tensor_container_t = typename tensor_container<C, T, N>::type;

The function signature then becomes

.. code::

    template <class T, class C = xtensor_c>
    void compute(const tensor_container_t<C, T, 2>& t);

The Python bindings only require that we specialize the ``tensor_container`` structure

.. code::

    struct pytensor_c
    {
    };
    
    template <class T, std::size_t N>
    struct tensor_container<pytensor_c, T, N>
    {
        using type = pytensor<T, N>;
    };

    PYBIND11_MODULE(pymod, m)
    {
        xt::import_numpy();

        m.def("compute", &compute<double, pytensor_c>);
    }

Even if we need to specialize the “tensor_container” structure for each language, the specialization can be
reused for other functions and thus reduce the amount of typing required. This comes at a cost though: we’ve
lost type inference on the C++ side.

.. code::

    xt::xtensor<double, 2> t {{1., 2., 3.}, {4., 5., 6.}};

    compute<double>(t);  // works
    compute(t);          // error (couldn't infer template argument 'T')

Besides, if later we want to support arrays, we need to add an “array_container” structure and its specializations,
and an overload of the compute function:

.. code::

    template <class C, class T>
    struct array_container;

    template <class C, class T>
    struct array_container<xtensor_c, T>
    {
        using type = xt::xarray<T>;
    };

    template <class C, class T>
    using array_container_t = typename array_container<C, T>::type;

    template <class T, class C = xtensor_c>
    void compute(const array_container_t<C, T>& t);

Type restriction with SFINAE
----------------------------

The major drawback of the previous option is the loss of type inference in C++. The only means to get it back
is to reintroduce a generic parameter type. However, we can make the compiler generate an invalid type so the
function is removed from the overload resolution set when the actual type of the argument does not satisfy
some constraint. This principle is known as SFINAE (Substitution Failure Is Not An Error). Modern C++ provide
metafunctions to help us make use of SFINAE:

.. code::

    template <class C>
    struct is_tensor : std::false_type
    {
    };

    template <class T, std::size_t N, layout_type L, class Tag>
    struct is_tensor<xtensor<T, N, L, Tag>> : std::true_type
    {
    };

    template <class T, template <class> class C = is_tensor, 
              std::enable_if_t<C<T>::value, bool> = true>
    void compute(const T& t);

Here when ``C<T>::value`` is true, the ``enable_if_t`` invocation generates the bool type. Otherwise, it does 
not generate anything, leading to an invalid function declaration. The compiler removes this declaration from
the overload resolution set and no error happens if another “compute” overload is a good match for the call.
Otherwise, the compiler emits an error.

The default value is here to avoid the need to pass a boolean value when invoking the ``compute`` function; this
value is of no use, we only rely on the SFINAE trick.

This declaration has a slight problem: adding ``enable_if_t`` to the signature of each function we want to expose
is cumbersome. Let’s make this part more expressive:

.. code::

    template <template<class> class C, class T>
    using check_constraints = std::enable_if_t<C<T>::value, bool>;
    template <class T, template <class> class C = is_tensor,
              check_constraints<C, T> = true>
    void compute(const T& t);

All good, we have type inference and an expressive syntax for declaring our function. Besides, if we want to relax
the constraint so the function can accept both tensors and arrays, all we have to do is to replace the default value
for C:

.. code::

    // Equivalent to is_tensor<T>::value || is_array<T>::value
    template <class T>
    sturct is_container : xtl::disjunction<is_tensor<T>, is_array<T>>
    {
    };

    template <class T, template <class> class C = is_container,
              check_constraints<C, T> = true>
    void compute(const T& t);

This is far more flexible than the previous option. This flexibility comes at a minor cost: exposing the function to
the Python is slightly more verbose:

.. code::

    template <class T, std::size_t N, layout_type L>
    struct is_tensor<pytensor<T, N, L>> : std::true_type
    {
    };

    PYBIND11_MODULE(pymod, m)
    {
        xt::import_numpy();

        m.def("compute", &compute<pytensor<double, 2>>);
    }

Conclusion
----------

Each solution has its pros and cons and choosing one of them should be done according to the flexibility you want to
impose on your API and the constraints you are imposed by the implementation. For instance, a method that requires a
lot of typing in the bindings might not suit for libraries with a huge amount of functions to expose, while a full
generic API might be problematic if the implementation expects containers only. Below is a summary of the advantages
and drawbacks of the different options:

- Generic API: full genericity, no additional typing required in the bindings, but maybe too permissive.
- Full qualified API: simple, accepts only the specified parameter type, but requires a lot of typing for the bindings.
- Container selection: quite simple, requires less typing than the previous method, but loses type inference on the C++ side and lacks some flexibility.
- Type restriction with SFINAE: more flexible than the previous option, gets type inference back, but slightly more complex to implement.

