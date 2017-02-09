/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XVIEW_HPP
#define XVIEW_HPP

#include <cstddef>
#include <utility>
#include <type_traits>
#include <tuple>
#include <array>
#include <algorithm>

#include "xtensor_forward.hpp"
#include "xbroadcast.hpp"
#include "xiterator.hpp"
#include "xsemantic.hpp"
#include "xview_utils.hpp"

namespace xt
{

    /*********************
     * xview declaration *
     *********************/

    template <class CT, class... S>
    struct xcontainer_inner_types<xview<CT, S...>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = view_temporary_type_t<xexpression_type, S...>;
    };

    template <bool is_const, class CT, class... S>
    class xview_stepper;

    template <class ST, class... S>
    struct xview_shape_type;


    /**
     * @class xview
     * @brief Multidimensional view with tensor semantic.
     *
     * The xview class implements a multidimensional view with tensor
     * semantic. It is used to adapt the shape of an xexpression without
     * changing it. xview is not meant to be used directly, but
     * only with the \ref view helper functions.
     *
     * @tparam CT the closure type of the \ref xexpression to adapt
     * @tparam S the slices type describing the shape adaptation
     *
     * @sa view, range, all, newaxis
     */
    template <class CT, class... S>
    class xview : public xview_semantic<xview<CT, S...>>
    {

    public:

        using self_type = xview<CT, S...>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename xexpression_type::value_type;
        using reference = typename xexpression_type::reference;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using shape_type = typename xview_shape_type<typename xexpression_type::shape_type, S...>::type;

        using slice_type = std::tuple<S...>;

        using stepper = xview_stepper<false, CT, S...>;
        using const_stepper = xview_stepper<true, CT, S...>;

        using iterator = xiterator<stepper, shape_type>;
        using const_iterator = xiterator<const_stepper, shape_type>;
        
        using storage_iterator = iterator;
        using const_storage_iterator = const_iterator;

        template <class... SL>
        xview(CT e, SL&&... slices) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type dimension() const noexcept;

        const shape_type& shape() const noexcept;
        const slice_type& slices() const noexcept;

        template <class... Args>
        reference operator()(Args... args);
        reference operator[](const xindex& index);
        template <class It>
        reference element(It first, It last);

        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;
        template <class It>
        const_reference element(It first, It last) const;

        template <class ST>
        bool broadcast_shape(ST& shape) const;

        template <class ST>
        bool is_trivial_broadcast(const ST& strides) const;

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        template <class ST>
        xiterator<stepper, ST> xbegin(const ST& shape);
        template <class ST>
        xiterator<stepper, ST> xend(const ST& shape);

        template <class ST>
        xiterator<const_stepper, ST> xbegin(const ST& shape) const;
        template <class ST>
        xiterator<const_stepper, ST> xend(const ST& shape) const;
        template <class ST>
        xiterator<const_stepper, ST> cxbegin(const ST& shape) const;
        template <class ST>
        xiterator<const_stepper, ST> cxend(const ST& shape) const;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape) const;

        storage_iterator storage_begin();
        storage_iterator storage_end();

        const_storage_iterator storage_begin() const;
        const_storage_iterator storage_end() const;

        const_storage_iterator storage_cbegin() const;
        const_storage_iterator storage_cend() const;

    private:

        // VS 2015 workaround (yes, really)
        template <std::size_t I>
        struct lesser_condition
        {
            static constexpr bool value = (I + newaxis_count_before<S...>(I + 1) < sizeof...(S));
        };

        CT m_e;
        slice_type m_slices;
        shape_type m_shape;

        template <typename std::decay_t<CT>::size_type... I, class... Args>
        reference access_impl(std::index_sequence<I...>, Args... args);

        template <typename std::decay_t<CT>::size_type... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const;

        template <typename std::decay_t<CT>::size_type I, class... Args>
        std::enable_if_t<lesser_condition<I>::value, size_type> index(Args... args) const;

        template <typename std::decay_t<CT>::size_type I, class... Args >
        std::enable_if_t<!lesser_condition<I>::value, size_type> index(Args... args) const;

        template<typename std::decay_t<CT>::size_type, class T>
        size_type sliced_access(const xslice<T>& slice) const;

        template<typename std::decay_t<CT>::size_type I, class T, class Arg, class... Args>
        size_type sliced_access(const xslice<T>& slice, Arg arg, Args... args) const;

        template<typename std::decay_t<CT>::size_type I, class T, class... Args>
        disable_xslice<T, size_type> sliced_access(const T& squeeze, Args...) const;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        template <class It>
        base_index_type make_index(It first, It last) const;

        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xview<CT, S...>>;
    };

    template <class E, class... S>
    auto view(E&& e, S&&... slices);

    /*****************************
     * xview_stepper declaration *
     *****************************/

    namespace detail
    {
        template <class V>
        struct get_stepper_impl
        {
            using xexpression_type = typename V::xexpression_type;
            using type = typename xexpression_type::stepper;
        };

        template <class V>
        struct get_stepper_impl<const V>
        {
            using xexpression_type = typename V::xexpression_type;
            using type = typename xexpression_type::const_stepper;
        };
    }

    template <class V>
    using get_stepper = typename detail::get_stepper_impl<V>::type;

    template <bool is_const, class CT, class... S>
    class xview_stepper
    {

    public:

        using view_type = std::conditional_t<is_const,
                                             const xview<CT, S...>,
                                             xview<CT, S...>>;
        using substepper_type = get_stepper<view_type>;

        using value_type = typename substepper_type::value_type;
        using reference = typename substepper_type::reference;
        using pointer = typename substepper_type::pointer;
        using difference_type = typename substepper_type::difference_type;
        using size_type = typename view_type::size_type;

        using shape_type = typename substepper_type::shape_type;

        xview_stepper() = default;
        xview_stepper(view_type* view, substepper_type it,
                      size_type offset, bool end = false);

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        bool equal(const xview_stepper& rhs) const;

    private:

        bool is_newaxis_slice(size_type index) const noexcept;

        template <class F>
        void common_step(size_type dim, size_type n, F f);

        view_type* p_view;
        substepper_type m_it;
        size_type m_offset;
    };

    template <bool is_const, class CT, class... S>
    bool operator==(const xview_stepper<is_const, CT, S...>& lhs,
                    const xview_stepper<is_const, CT, S...>& rhs);

    template <bool is_const, class CT, class... S>
    bool operator!=(const xview_stepper<is_const, CT, S...>& lhs,
                    const xview_stepper<is_const, CT, S...>& rhs);


    // meta-function returning the shape type for an xview 
    template <class ST, class... S>
    struct xview_shape_type
    {
        using type = ST;
    };

    template <class I, std::size_t L, class... S>
    struct xview_shape_type<std::array<I, L>, S...>
    {
        using type = std::array<I, L - integral_count<S...>() + newaxis_count<S...>()>;
    };

    /************************
     * xview implementation *
     ************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs a view on the specified xexpression.
     * Users should not call directly this constructor but
     * use the view function instead.
     * @param e the xexpression to adapt
     * @param slices the slices list describing the view
     * @sa view
     */
    template <class CT, class... S>
    template <class... SL>
    inline xview<CT, S...>::xview(CT e, SL&&... slices) noexcept
        : m_e(e), m_slices(std::forward<SL>(slices)...),
          m_shape(make_sequence<shape_type>(m_e.dimension() - integral_count<S...>() + newaxis_count<S...>(), 0))
    {
        auto func = [](const auto& s) noexcept { return get_size(s); };
        for (size_type i = 0; i != dimension(); ++i)
        {
            size_type index = integral_skip<S...>(i);
            m_shape[i] = index < sizeof...(S) ?
                apply<std::size_t>(index, func, m_slices) : m_e.shape()[index - newaxis_count_before<S...>(index)];
        }
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class CT, class... S>
    template <class E>
    inline auto xview<CT, S...>::operator=(const xexpression<E>& e) -> self_type&
    {
        bool cond = (e.derived_cast().shape().size() == dimension())
                    && std::equal(shape().begin(), shape().end(), e.derived_cast().shape().begin());
        if(!cond)
        {
            semantic_base::operator=(broadcast(e.derived_cast(), shape()));
        }
        else
        {
            semantic_base::operator=(e);
        }
        return *this;
    }
    //@}

    template <class CT, class... S>
    template <class E>
    inline auto xview<CT, S...>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(storage_begin(), storage_end(), e);
        return *this;
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of dimensions of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::dimension() const noexcept -> size_type
    {
        return m_e.dimension() - integral_count<S...>() + newaxis_count<S...>();
    }

    /**
     * Returns the shape of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the slices of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::slices() const noexcept -> const slice_type&
    {
        return m_slices;
    }
    //@}

    /**
     * @name Data
     */
    //@{
    /**
     * Returns a reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater
     * than the number of dimensions of the view.
     */
    template <class CT, class... S>
    template <class... Args>
    inline auto xview<CT, S...>::operator()(Args... args) -> reference
    {
        return access_impl(std::make_index_sequence<(sizeof...(Args) + integral_count<S...>() > newaxis_count<S...>() ?
                                                        sizeof...(Args) + integral_count<S...>() - newaxis_count<S...>() :
                                                        0)>(),
                           args...);
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::operator[](const xindex& index) -> reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class... S>
    template <class It>
    inline auto xview<CT, S...>::element(It first, It last) -> reference
    {
        auto index = make_index(first, last);
        return m_e.element(index.cbegin(), index.cend());
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices must be
     * unsigned integers, the number of indices should be equal or greater than the number
     * of dimensions of the view.
     */
    template <class CT, class... S>
    template <class... Args>
    inline auto xview<CT, S...>::operator()(Args... args) const -> const_reference
    {
        return access_impl(std::make_index_sequence<(sizeof...(Args) + integral_count<S...>() > newaxis_count<S...>() ?
                                                        sizeof...(Args) + integral_count<S...>() - newaxis_count<S...>() :
                                                        0)>(),
                           args...);
    }

    template <class CT, class... S>
    inline auto xview<CT, S...>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class... S>
    template <class It>
    inline auto xview<CT, S...>::element(It first, It last) const -> const_reference
    {
        auto index = make_index(first, last);
        return m_e.element(index.cbegin(), index.cend());
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the view to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class... S>
    template <class ST>
    inline bool xview<CT, S...>::broadcast_shape(ST& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the view to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class... S>
    template <class ST>
    inline bool xview<CT, S...>::is_trivial_broadcast(const ST& /*strides*/) const
    {
        return false;
    }
    //@}

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type... I, class... Args>
    inline auto xview<CT, S...>::access_impl(std::index_sequence<I...>, Args... args) -> reference
    {
        return m_e(index<I>(args...)...);
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type... I, class... Args>
    inline auto xview<CT, S...>::access_impl(std::index_sequence<I...>, Args... args) const -> const_reference
    {
        return m_e(index<I>(args...)...);
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type I, class... Args>
    inline auto xview<CT, S...>::index(Args... args) const -> std::enable_if_t<lesser_condition<I>::value, size_type>
    {
        return sliced_access<I - integral_count_before<S...>(I) + newaxis_count_before<S...>(I + 1)>
            (std::get<I + newaxis_count_before<S...>(I + 1)>(m_slices), args...);
    }

    template <class CT, class... S>
    template <typename std::decay_t<CT>::size_type I, class... Args>
    inline auto xview<CT, S...>::index(Args... args) const -> std::enable_if_t<!lesser_condition<I>::value, size_type>
    {
        return argument<I - integral_count<S...>() + newaxis_count<S...>()>(args...);
    }

    template <class CT, class... S>
    template<typename std::decay_t<CT>::size_type I, class T>
    inline auto xview<CT, S...>::sliced_access(const xslice<T>& slice) const -> size_type
    {
        return slice.derived_cast()(0);
    }

    template <class CT, class... S>
    template<typename std::decay_t<CT>::size_type I, class T, class Arg, class... Args>
    inline auto xview<CT, S...>::sliced_access(const xslice<T>& slice, Arg arg, Args... args) const -> size_type
    {
        return slice.derived_cast()(argument<I>(arg, args...));
    }

    template <class CT, class... S>
    template<typename std::decay_t<CT>::size_type I, class T, class... Args>
    inline auto xview<CT, S...>::sliced_access(const T& squeeze, Args...) const -> disable_xslice<T, size_type>
    {
        return squeeze;
    }

    template <class CT, class... S>
    template <class It>
    inline auto xview<CT, S...>::make_index(It first, It last) const -> base_index_type
    {
        auto index = make_sequence<typename xexpression_type::shape_type>(m_e.dimension(), 0);
        auto func1 = [&first](const auto& s)
        {
            return get_slice_value(s, first);
        };
        auto func2 = [](const auto& s)
        {
            return xt::value(s, 0);
        };
        for (size_type i = 0; i != m_e.dimension(); ++i)
        {
            size_type k = newaxis_skip<S...>(i);
            std::advance(first, k - i);
            if (first != last)
            {
                index[i] = k < sizeof...(S) ?
                    apply<size_type>(k, func1, m_slices) : *first++;
            }
            else
            {
                index[i] = k < sizeof...(S) ?
                    apply<size_type>(k, func2, m_slices) : 0;
            }
        }
        return index;
    }

    template <class CT, class... S>
    inline void xview<CT, S...>::assign_temporary_impl(temporary_type& tmp)
    {
        std::copy(tmp.storage_cbegin(), tmp.storage_cend(), begin());
    }

    namespace detail
    {
        template <class E, class... S>
        inline std::size_t get_underlying_shape_index(std::size_t I)
        {
            return I - newaxis_count_before<get_slice_type<E, S>...>(I);
        }

        template <class E, std::size_t... I, class... S>
        inline auto make_view_impl(E&& e, std::index_sequence<I...>, S&&... slices)
        {
            using view_type = xview<closure_t<E>, get_slice_type<std::decay_t<E>, S>...>;
            return view_type(std::forward<E>(e),
                get_slice_implementation(e, std::forward<S>(slices), get_underlying_shape_index<std::decay_t<E>, S...>(I))...
            );
        }
    }

    /**
     * Constructs and returns a view on the specified xexpression. Users
     * should not directly construct the slices but call helper functions
     * instead.
     * @param e the xexpression to adapt
     * @param slices the slices list describing the view
     * @sa range, all, newaxis
     */
    template <class E, class... S>
    inline auto view(E&& e, S&&... slices)
    {
        return detail::make_view_impl(std::forward<E>(e), std::make_index_sequence<sizeof...(S)>(), std::forward<S>(slices)...);
    }

    /****************
     * iterator api *
     ****************/

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::begin() -> iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns an iterator to the element following the last element
     * of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::end() -> iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::begin() const -> const_iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::end() const -> const_iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::cbegin() const -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::cend() const -> const_iterator
    {
        return end();
    }

    /**
     * Returns an iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::xbegin(const ST& shape) -> xiterator<stepper, ST>
    {
        return xiterator<stepper, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::xend(const ST& shape) -> xiterator<stepper, ST>
    {
        return xiterator<stepper, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::xbegin(const ST& shape) const -> xiterator<const_stepper, ST>
    {
        return xiterator<const_stepper, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::xend(const ST& shape) const -> xiterator<const_stepper, ST>
    {
        return xiterator<const_stepper, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::cxbegin(const ST& shape) const -> xiterator<const_stepper, ST>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::cxend(const ST& shape) const -> xiterator<const_stepper, ST>
    {
        return xend(shape);
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, m_e.stepper_begin(m_e.shape()), offset);
    }

    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::stepper_end(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, m_e.stepper_end(m_e.shape()), offset, true);
    }

    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        const xexpression_type& e = m_e;
        return const_stepper(this, e.stepper_begin(m_e.shape()), offset);
    }

    template <class CT, class... S>
    template <class ST>
    inline auto xview<CT, S...>::stepper_end(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        const xexpression_type& e = m_e;
        return const_stepper(this, e.stepper_end(m_e.shape()), offset, true);
    }

    /************************
     * storage_iterator api *
     ************************/

    /**
     * @name Storage iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the buffer containing
     * the elements of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::storage_begin() -> storage_iterator
    {
        return begin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the buffer containing the elements of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::storage_end() -> storage_iterator
    {
        return end();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::storage_begin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::storage_end() const -> const_storage_iterator
    {
        return cend();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::storage_cbegin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the view.
     */
    template <class CT, class... S>
    inline auto xview<CT, S...>::storage_cend() const -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /********************************
     * xview_stepper implementation *
     ********************************/

    template <bool is_const, class CT, class... S>
    inline xview_stepper<is_const, CT, S...>::xview_stepper(view_type* view, substepper_type it,
                                                           size_type offset, bool end)
        : p_view(view), m_it(it), m_offset(offset)
    {
        if(!end)
        {
            auto func = [](const auto& s) { return xt::value(s, 0); };
            for(size_type i = 0; i < sizeof...(S); ++i)
            {
                if (!is_newaxis_slice(i))
                {
                    size_type s = apply<size_type>(i, func, p_view->slices());
                    size_type index = i - newaxis_count_before<S...>(i);
                    m_it.step(index, s);
                }
            }
        }
    }

    template <bool is_const, class CT, class... S>
    inline auto xview_stepper<is_const, CT, S...>::operator*() const -> reference
    {
        return *m_it;
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::step(size_type dim, size_type n)
    {
        auto func = [this](size_type index, size_type offset) { m_it.step(index, offset); };
        common_step(dim, n, func);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::step_back(size_type dim, size_type n)
    {
        auto func = [this](size_type index, size_type offset) { m_it.step_back(index, offset); };
        common_step(dim, n, func);
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::reset(size_type dim)
    {
        if(dim >= m_offset)
        {
            auto size_func = [](const auto& s) noexcept { return get_size(s); };
            auto step_func = [](const auto& s) noexcept { return step_size(s); };
            size_type index = integral_skip<S...>(dim);
            if (!is_newaxis_slice(index))
            {
                size_type size = index < sizeof...(S) ?
                    apply<size_type>(index, size_func, p_view->slices()) : p_view->shape()[dim];
                if (size != 0) size = size - 1;
                size_type step_size = index < sizeof...(S) ?
                    apply<size_type>(index, step_func, p_view->slices()) : 1;
                index -= newaxis_count_before<S...>(index);
                m_it.step_back(index, step_size * size);
            }
        }
    }

    template <bool is_const, class CT, class... S>
    inline void xview_stepper<is_const, CT, S...>::to_end()
    {
        m_it.to_end();
    }

    template <bool is_const, class CT, class... S>
    inline bool xview_stepper<is_const, CT, S...>::equal(const xview_stepper& rhs) const
    {
        return p_view == rhs.p_view && m_it == rhs.m_it && m_offset == rhs.m_offset;
    }

    template <bool is_const, class CT, class... S>
    inline bool xview_stepper<is_const, CT, S...>::is_newaxis_slice(size_type index) const noexcept
    {
        // A bit tricky but avoids a lot of template instantiations
        return newaxis_count_before<S...>(index + 1) != newaxis_count_before<S...>(index);
    }

    template <bool is_const, class CT, class... S>
    template <class F>
    void xview_stepper<is_const, CT, S...>::common_step(size_type dim, size_type n, F f)
    {
        if (dim >= m_offset)
        {
            auto func = [](const auto& s) noexcept { return step_size(s); };
            size_type index = integral_skip<S...>(dim);
            if (!is_newaxis_slice(index))
            {
                size_type step_size = index < sizeof...(S) ?
                    apply<size_type>(index, func, p_view->slices()) : 1;
                index -= newaxis_count_before<S...>(index);
                f(index, step_size * n);
            }
        }
    }

    template <bool is_const, class CT, class... S>
    inline bool operator==(const xview_stepper<is_const, CT, S...>& lhs,
                           const xview_stepper<is_const, CT, S...>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <bool is_const, class CT, class... S>
    inline bool operator!=(const xview_stepper<is_const, CT, S...>& lhs,
                           const xview_stepper<is_const, CT, S...>& rhs)
    {
        return !(lhs.equal(rhs));
    }
}

#endif
