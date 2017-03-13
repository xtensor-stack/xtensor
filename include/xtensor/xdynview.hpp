/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XDYNVIEW_HPP
#define XDYNVIEW_HPP

#include <utility>
#include <type_traits>
#include <tuple>
#include <algorithm>
#include <typeinfo>

#include "xarray.hpp"
#include "xiterator.hpp"
#include "xslice.hpp"
#include "xview.hpp"

#include "thirdparty/any.hpp"

namespace xt
{

    /************************
     * xdynview declaration *
     ************************/

    template <bool is_const, class E>
    class xdynview_stepper;

    template <class CT>
    class xdynview;

    template <class CT>
    struct xcontainer_inner_types<xdynview<CT>>
    {
        using temporary_type = xarray<typename std::decay_t<CT>::value_type>;
    };

    template <class CT>
    struct xiterable_inner_types<xdynview<CT>>
    {
        using xexpression_type = std::decay_t<CT>;
        using shape_type = std::vector<std::size_t>;
        using stepper = xdynview_stepper<false, CT>;
        using const_stepper = xdynview_stepper<true, CT>;
        using broadcast_iterator = xiterator<stepper, shape_type*>;
        using const_broadcast_iterator = xiterator<const_stepper, shape_type*>;
        using iterator = broadcast_iterator;
        using const_iterator = const_broadcast_iterator;
    };

    /**
     * @class xdynview
     * @brief Multidimensional view with tensor semantic.
     *
     * The xdynview class implements a multidimensional view with tensor
     * semantic. It is used to adapt the shape of an xexpression without
     * changing it.
     *
     * @tparam E the expression type to adapt
     * @tparam S the slices type describing the shape adaptation
     *
     * @sa make_xdynview
     */
    template <class CT>
    class xdynview : public xview_semantic<xdynview<CT>>,
                     public xiterable<xdynview<CT>>
    {

    public:

        using self_type = xdynview<CT>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename xexpression_type::value_type;
        using reference = typename xexpression_type::reference;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using strides_type = std::vector<size_type>;

        using xintegral_t = std::size_t;
        using xall_t = xall<std::size_t>;
        using xrange_t = xrange<std::size_t>;
        using xstepped_range_t = xstepped_range<std::size_t>;
        using xnewaxis_t = xnewaxis<std::size_t>;

        using slice_type = std::vector<linb::any>;

        using iterable_base = xiterable<self_type>;
        using shape_type = typename iterable_base::shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using broadcast_iterator = typename iterable_base::broadcast_iterator;
        using const_broadcast_iterator = typename iterable_base::const_broadcast_iterator;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;

        using closure_type = const self_type&;

        xdynview(CT& e, const slice_type& slices) noexcept;

        template <class OE>
        self_type& operator=(const xexpression<OE>& e);

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

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape) const;

    private:

        CT m_e;
        slice_type m_slices;
        shape_type m_shape;

        template <class It>
        xindex get_idx(It begin, It end) const;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xdynview<CT>>;
    };

    template <class CT>
    xdynview<CT> make_dynview(CT& e, const typename xdynview<CT>::slice_type& slices);

    template <bool is_const, class E>
    class xdynview_stepper
    {

    public:

        using view_type = std::conditional_t<is_const,
                                             const xdynview<E>,
                                             xdynview<E>>;
        using substepper_type = get_stepper<view_type>;

        using value_type = typename substepper_type::value_type;
        using reference = typename substepper_type::reference;
        using pointer = typename substepper_type::pointer;
        using difference_type = typename substepper_type::difference_type;
        using size_type = typename view_type::size_type;

        using shape_type = typename substepper_type::shape_type;

        xdynview_stepper(view_type* view, substepper_type it,
                         size_type offset, bool end = false);

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        bool equal(const xdynview_stepper& rhs) const;

    private:

        view_type* p_view;
        substepper_type m_it;
        size_type m_offset;
    };

    template <bool is_const, class E>
    inline bool operator==(const xdynview_stepper<is_const, E>& lhs,
                           const xdynview_stepper<is_const, E>& rhs);

    template <bool is_const, class E>
    inline bool operator!=(const xdynview_stepper<is_const, E>& lhs,
                           const xdynview_stepper<is_const, E>& rhs);

    /************************
     * xdynview implementation *
     ************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs a view on the specified xexpression.
     * Users should not call directly this constructor but
     * use the make_xdynview function instead.
     * @param e the xexpression to adapt
     * @param slices the slices list describing the view
     * @sa make_xdynview
     */

    namespace detail
    {
        inline std::size_t integral_skip(const std::vector<linb::any>& vec, std::size_t i)
        {
            std::size_t idx = 0;
            for (; i < vec.size(); ++i)
            {
                if (vec[i].type() != typeid(std::size_t)) {
                    return i;
                }
            }
            return i;
        }

        inline std::size_t integral_before(const std::vector<linb::any>& vec, std::size_t end)
        {
            std::size_t count = 0;
            for (std::size_t i = 0; i < end && i < vec.size(); ++i)
            {
                if (vec[i].type() == typeid(std::size_t))
                {
                    count++;
                }
            }
            return count;
        }

        inline std::size_t newaxis_before(const std::vector<linb::any>& vec, std::size_t end)
        {
            std::size_t count = 0;
            for (std::size_t i = 0; i < end && i < vec.size(); ++i)
            {
                if (vec[i].type() == typeid(xnewaxis<std::size_t>))
                {
                    count++;
                }
            }
            return count;
        }

        inline std::size_t is_newaxis_any(const linb::any& slice)
        {
            return slice.type() == typeid(xnewaxis<std::size_t>);
        }


        inline std::size_t integral_count(const std::vector<linb::any>& vec)
        {
            std::size_t count = 0;
            for (const auto& el : vec)
            {
                if (el.type() == typeid(std::size_t))
                {
                    count++;
                }
            }
            return count;
        }

        inline std::size_t newaxis_count(const std::vector<linb::any>& vec)
        {
            std::size_t count = 0;
            for (const auto& el : vec)
            {
                if (el.type() == typeid(xnewaxis<std::size_t>))
                {
                    count++;
                }
            }
            return count;
        }

        inline std::size_t get_size(const linb::any& slice)
        {
            if (slice.type() == typeid(xall<std::size_t>))
            {
                return get_size(linb::any_cast<xall<std::size_t>>(slice));
            }
            else if (slice.type() == typeid(xnewaxis<std::size_t>))
            {
                return get_size(linb::any_cast<xnewaxis<std::size_t>>(slice));
            }
            else if (slice.type() == typeid(xrange<std::size_t>))
            {
                return get_size(linb::any_cast<xrange<std::size_t>>(slice));
            }
            else if (slice.type() == typeid(xstepped_range<std::size_t>))
            {
                return get_size(linb::any_cast<xstepped_range<std::size_t>>(slice));
            }
            else
            {
                std::cout << "UNNORMALIZED SLICE TYPE ERROR " << slice.type().name() << std::endl;
            }
        }

        inline std::size_t get_value(const linb::any& slice, std::size_t at)
        {
            if (slice.type() == typeid(std::size_t))
            {
                return (std::size_t) linb::any_cast<std::size_t>(slice);
            }
            else if (slice.type() == typeid(xall<std::size_t>))
            {
                return (std::size_t) value(linb::any_cast<xall<std::size_t>>(slice), at);
            }
            else if (slice.type() == typeid(xnewaxis<std::size_t>))
            {
                return (std::size_t) value(linb::any_cast<xnewaxis<std::size_t>>(slice), at);
            }
            else if (slice.type() == typeid(xrange<std::size_t>))
            {
                return (std::size_t) value(linb::any_cast<xrange<std::size_t>>(slice), at);
            }
            else if (slice.type() == typeid(xstepped_range<std::size_t>))
            {
                return (std::size_t) value(linb::any_cast<xstepped_range<std::size_t>>(slice), at);
            }
            else
            {
                std::cout << "UNNORMALIZED SLICE TYPE ERROR" << slice.type().name()<< std::endl;
            }
        }

        inline std::size_t get_step_size(const linb::any& slice)
        {
            if (slice.type() == typeid(std::size_t))
            {
                return (std::size_t) 0;
            }
            else if (slice.type() == typeid(xall<std::size_t>))
            {
                return (std::size_t) step_size(linb::any_cast<xall<std::size_t>>(slice));
            }
            else if (slice.type() == typeid(xnewaxis<std::size_t>))
            {
                return (std::size_t) step_size(linb::any_cast<xnewaxis<std::size_t>>(slice));
            }
            else if (slice.type() == typeid(xrange<std::size_t>))
            {
                return (std::size_t) step_size(linb::any_cast<xrange<std::size_t>>(slice));
            }
            else if (slice.type() == typeid(xstepped_range<std::size_t>))
            {
                return (std::size_t) step_size(linb::any_cast<xstepped_range<std::size_t>>(slice));
            }
            else
            {
                std::cout << "UNNORMALIZED SLICE TYPE ERROR"<< slice.type().name() << std::endl;
            }
        }
    }

    template <class CT>
    xdynview<CT> dynview(CT& e, const typename xdynview<CT>::slice_type& slices)
    {
        using view = xdynview<CT>;
        using slice_type = typename view::slice_type;
        using size_type = typename view::size_type;

        size_type newaxis_before = 0;
        slice_type temp_slices(slices.size());

        for (size_type i = 0; i < slices.size(); ++i)
        {
            if (slices[i].type() == typeid(xall_tag))
            {
                temp_slices[i] = typename view::xall_t(e.shape()[i - newaxis_before]);
            }
            else if (slices[i].type() == typeid(xnewaxis_tag))
            {
                temp_slices[i] = typename view::xnewaxis_t();
                newaxis_before++;
            }
            else
            {
                if (slices[i].type() == typeid(xrange<int>))
                {
                    xrange<int> slice = linb::any_cast<xrange<int>>(slices[i]);
                    temp_slices[i] = xrange<std::size_t>(slice(0), slice(0) + slice.size());
                }
                else if (slices[i].type() == typeid(xstepped_range<int>))
                {
                    xstepped_range<int> slice = linb::any_cast<xstepped_range<int>>(slices[i]);
                    temp_slices[i] = xstepped_range<std::size_t>(slice(0), slice(0) + slice.size() * slice.step_size(), slice.step_size());
                }
                else if (slices[i].type() == typeid(int))
                {
                    temp_slices[i] = (std::size_t) linb::any_cast<int>(slices[i]);
                }
                else {
                    temp_slices[i] = slices[i];
                }
            }
        }
        return xdynview<CT>(e, temp_slices);
    }

    template <class CT>
    inline xdynview<CT>::xdynview(CT& e, const slice_type& slices) noexcept
        : m_e(e), m_slices(slices)
    {
        m_shape.resize(dimension());
        m_slices.resize(slices.size());

        for (size_type i = 0; i != dimension(); ++i)
        {
            size_type index = detail::integral_skip(m_slices, i) + detail::integral_before(m_slices, i);
            if (index < m_slices.size())
            {
                if (m_slices[index].type() == typeid(std::size_t))
                {
                    m_shape[i] = 1;
                }
                else
                {
                    m_shape[i] = detail::get_size(m_slices[index]);
                }
            }
            else
            {
                m_shape[i] = m_e.shape()[index];
            }
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
    template <class CT>
    template <class OE>
    inline auto xdynview<CT>::operator=(const xexpression<OE>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of dimensions of the view.
     */
    template <class CT>
    inline auto xdynview<CT>::dimension() const noexcept -> size_type
    {
        return m_e.dimension() - detail::integral_count(m_slices) + detail::newaxis_count(m_slices);
    }

    /**
     * Returns the shape of the view.
     */
    template <class CT>
    inline auto xdynview<CT>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the slices of the view.
     */
    template <class CT>
    inline auto xdynview<CT>::slices() const noexcept -> const slice_type&
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
     * @param args a list of indices specifying the position in the voew. Indices
     * must be unsigned integers, the number of indices should be equal or greater
     * than the number of dimensions of the view.
     */
    template <class CT>
    template <class... Args>
    inline auto xdynview<CT>::operator()(Args... args) -> reference
    {
        std::array<size_type, sizeof...(Args)> arg_array = { static_cast<size_type>(args)... };
        return m_e[get_idx(arg_array.begin(), arg_array.end())];
    }

    template <class CT>
    inline auto xdynview<CT>::operator[](const xindex& index) -> reference
    {
        return m_e[get_idx(index.cbegin(), index.cend())];
    }

    template <class CT>
    template <class It>
    inline auto xdynview<CT>::element(It first, It last) -> reference
    {
        return m_e[get_idx(first, last)];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices must be
     * unsigned integers, the number of indices should be equal or greater than the number
     * of dimensions of the view.
     */
    template <class CT>
    template <class... Args>
    inline auto xdynview<CT>::operator()(Args... args) const -> const_reference
    {
        std::array<size_type, sizeof...(Args)> arg_array = { static_cast<size_type>(args)... };
        return m_e[get_idx(arg_array.begin(), arg_array.end())];
    }
    template <class CT>
    inline auto xdynview<CT>::operator[](const xindex& index) const -> const_reference
    {
        return m_e[get_idx(index.cbegin(), index.cend())];
    }

    template <class CT>
    template <class It>
    inline auto xdynview<CT>::element(It first, It last) const -> const_reference
    {
        return m_e[get_idx(first, last)];
    }

    //@}

    template <class CT>
    inline void xdynview<CT>::assign_temporary_impl(temporary_type& tmp)
    {
        std::copy(tmp.storage_cbegin(), tmp.storage_cend(), this->xbegin());
    }

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the view to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcast is trivial
     */
    template <class CT>
    template <class ST>
    inline bool xdynview<CT>::broadcast_shape(ST& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the view to see wether
     * the broadcast is trivial.integral_count<S
     * @return a boolean indicating whether the broadcast is trivial
     */
    template <class CT>
    template <class ST>
    inline bool xdynview<CT>::is_trivial_broadcast(const ST& /*strides*/) const
    {
        return false;
    }
    //@}

    template <class CT>
    template <class It>
    inline xindex xdynview<CT>::get_idx(It begin, It end) const
    {
        xindex idx;
        bool end_reached = false;
        for (const auto& el : m_slices)
        {
            if (begin == end)
            {
                end_reached = true;
            }
            if (el.type() == typeid(xnewaxis_t))
            {
                if (!end_reached)
                {
                    ++begin;
                }
            }
            else if (el.type() == typeid(xintegral_t))
            {
                idx.push_back((std::size_t) linb::any_cast<xintegral_t>(el));
            }
            else
            {
                std::size_t slice_index = 0;
                if (!end_reached) {
                    slice_index = *begin;
                }
                if (el.type() == typeid(xall_t))
                {
                    idx.push_back((std::size_t) linb::any_cast<xall_t>(el)(slice_index));
                }
                else if (el.type() == typeid(xrange_t))
                {
                    idx.push_back((std::size_t) linb::any_cast<xrange_t>(el)(slice_index));
                }
                else if (el.type() == typeid(xstepped_range_t))
                {
                    idx.push_back((std::size_t) linb::any_cast<xstepped_range_t>(el)(slice_index));
                }
                if (!end_reached)
                {
                    ++begin;
                }
            }
        }
        for (; begin != end; ++ begin)
        {
            idx.push_back(*begin);
        }
        return idx;
    }

    /***************
     * stepper api *
     ***************/

    template <class CT>
    template <class ST>
    inline auto xdynview<CT>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, m_e.stepper_begin(m_e.shape()), offset);
    }

    template <class CT>
    template <class ST>
    inline auto xdynview<CT>::stepper_end(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, m_e.stepper_end(m_e.shape()), offset, true);
    }

    template <class CT>
    template <class ST>
    inline auto xdynview<CT>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        const xexpression_type& e = m_e;
        return const_stepper(this, e.stepper_begin(m_e.shape()), offset);
    }

    template <class CT>
    template <class ST>
    inline auto xdynview<CT>::stepper_end(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        const xexpression_type& e = m_e;
        return const_stepper(this, e.stepper_end(m_e.shape()), offset, true);
    }

    /***********************************
     * xdynview_stepper implementation *
     ***********************************/

    template <bool is_const, class E>
    inline xdynview_stepper<is_const, E>::xdynview_stepper(view_type* view, substepper_type it,
                                                           size_type offset, bool end)
        : p_view(view), m_it(it), m_offset(offset)
    {
        if(!end)
        {
            const auto& slices = p_view->slices();
            for(size_type i = 0; i < slices.size(); ++i)
            {
                if (slices[i].type() == typeid(typename view_type::xnewaxis_t))
                {
                    continue;
                }
                else
                {
                    std::size_t value = detail::get_value(slices[i], 0);
                    m_it.step(i - detail::newaxis_before(slices, i), value);
                }
            }
        }
    }

    template <bool is_const, class E>
    inline auto xdynview_stepper<is_const, E>::operator*() const -> reference
    {
        return *m_it;
    }

    template <bool is_const, class E>
    inline void xdynview_stepper<is_const, E>::step(size_type dim, size_type n)
    {
        if(dim >= m_offset)
        {
            const auto& slices = p_view->slices();
            size_type index = detail::integral_skip(slices, dim) + detail::integral_before(slices, dim);
            if (index >= slices.size() || !detail::is_newaxis_any(slices[index]))
            {
                size_type step_size = index < slices.size() ?
                    detail::get_step_size(slices[index]) : 1;
                m_it.step(index - detail::newaxis_before(slices, index + 1), step_size * n);
            }
        }
    }

    template <bool is_const, class E>
    inline void xdynview_stepper<is_const, E>::step_back(size_type dim, size_type n)
    {
        if(dim >= m_offset)
        {
            const auto& slices = p_view->slices();
            size_type index = detail::integral_skip(slices, dim) + detail::integral_before(slices, dim);
            if (index >= slices.size() || !detail::is_newaxis_any(slices[index]))
            {
                size_type step_size = index < slices.size() ?
                    detail::get_step_size(slices[index]) : 1;
                m_it.step_back(index - detail::newaxis_before(slices, index + 1), step_size * n);
            }
        }
    }

    template <bool is_const, class E>
    inline void xdynview_stepper<is_const, E>::reset(size_type dim)
    {
        if(dim >= m_offset)
        {
            const auto& slices = p_view->slices();
            size_type index = detail::integral_skip(slices, dim) + detail::integral_before(slices, dim);
            if (index >= slices.size() || !detail::is_newaxis_any(slices[index]))
            {
                size_type size = index < slices.size() ?
                    detail::get_size(slices[index]) : p_view->shape()[dim];
                if(size != 0) size = size - 1;

                size_type step_size = index < slices.size() ?
                    detail::get_step_size(slices[index]) : 1;

                m_it.step_back(index - detail::newaxis_before(slices, index + 1), step_size * size);
            }
        }
    }

    template <bool is_const, class E>
    inline void xdynview_stepper<is_const, E>::to_end()
    {
        m_it.to_end();
    }

    template <bool is_const, class E>
    inline bool xdynview_stepper<is_const, E>::equal(const xdynview_stepper& rhs) const
    {
        return p_view == rhs.p_view && m_it == rhs.m_it && m_offset == rhs.m_offset;
    }

    template <bool is_const, class E>
    inline bool operator==(const xdynview_stepper<is_const, E>& lhs,
                           const xdynview_stepper<is_const, E>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <bool is_const, class E>
    inline bool operator!=(const xdynview_stepper<is_const, E>& lhs,
                           const xdynview_stepper<is_const, E>& rhs)
    {
        return !(lhs.equal(rhs));
    }
}

#endif