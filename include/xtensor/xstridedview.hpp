/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSTRIDEDVIEW_HPP
#define XSTRIDEDVIEW_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"

namespace xt
{
    namespace check_policy
    {
        struct none
        {
        };
        struct full
        {
        };
    }

    template <class CT>
    class xstrided_view;

    template <class CT>
    struct xcontainer_inner_types<xstrided_view<CT>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<typename xexpression_type::value_type>;
    };

    template <class CT>
    struct xiterable_inner_types<xstrided_view<CT>>
    {
        using inner_shape_type = typename std::decay_t<CT>::shape_type;
        using const_stepper = xindexed_stepper<xstrided_view<CT>>;
        using stepper = xindexed_stepper<xstrided_view<CT>, false>;
        using const_broadcast_iterator = xiterator<const_stepper, inner_shape_type*>;
        using broadcast_iterator = xiterator<stepper, inner_shape_type*>;
        using const_iterator = const_broadcast_iterator;
        using iterator = broadcast_iterator;
    };

    /*****************
     * xstrided_view *
     *****************/

    /**
     * @class xstrided_view
     * @brief View of an xexpression using strides
     *
     * The xstrided_view class implements a view utilizing an offset and strides 
     * into a multidimensional xcontainer. The xstridedview is currently used 
     * to implement `diagonal`, `flip` and `transpose.
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     *
     * @sa stridedview, transpose, diagonal, flip
     */
    template <class CT>
    class xstrided_view : public xview_semantic<xstrided_view<CT>>,
                          public xexpression_iterable<xstrided_view<CT>>
    {

    public:

        using self_type = xstrided_view<CT>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename xexpression_type::value_type;
        using reference = typename xexpression_type::reference;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using underlying_container_type = typename xexpression_type::container_type;

        using iterable_base = xexpression_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = shape_type;
        using closure_type = const self_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using broadcast_iterator = typename iterable_base::broadcast_iterator;
        using const_broadcast_iterator = typename iterable_base::const_broadcast_iterator;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        template <class I>
        xstrided_view(CT e, I&& shape, I&& strides, std::size_t offset) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;

        reference operator()();
        template <class... Args>
        reference operator()(Args... args);
        reference operator[](const xindex& index);
        reference operator[](size_type i);

        template <class It>
        reference element(It first, It last);

        const_reference operator()() const;
        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        bool broadcast_shape(O& shape) const;

        template <class O>
        bool is_trivial_broadcast(const O& strides) const noexcept;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape) const;

        underlying_container_type& data() noexcept;
        const underlying_container_type& data() const noexcept;

        value_type* raw_data() noexcept;
        const value_type* raw_data() const noexcept;

        size_type raw_data_offset() const noexcept;

    private:

        CT m_e;
        shape_type m_shape;
        strides_type m_strides;
        std::size_t m_offset;

        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xstrided_view<CT>>;
    };

    /*****************************
     * xstrided_view implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xstrided_view, selecting the indices specified by \a indices.
     * The resulting xexpression has a 1D shape with a length of n for n indices.
     * 
     * @param e the underlying xexpression for this view
     * @param indices the indices to select
     */
    template <class CT>
    template <class I>
    inline xstrided_view<CT>::xstrided_view(CT e, I&& shape, I&& strides, std::size_t offset) noexcept
        : m_e(e), m_shape(std::forward<I>(shape)), m_strides(std::forward<I>(strides)), m_offset(offset)
    {
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
    template <class E>
    inline auto xstrided_view<CT>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class CT>
    template <class E>
    inline auto xstrided_view<CT>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT>
    inline void xstrided_view<CT>::assign_temporary_impl(temporary_type& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->xbegin());
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the xstrided_view.
     */
    template <class CT>
    inline auto xstrided_view<CT>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the xstrided_view.
     */
    template <class CT>
    inline auto xstrided_view<CT>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xstrided_view.
     */
    template <class CT>
    inline auto xstrided_view<CT>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class CT>
    inline auto xstrided_view<CT>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }

    template <class CT>
    inline auto xstrided_view<CT>::data() noexcept -> underlying_container_type&
    {
        return m_e.data();
    }

    template <class CT>
    inline auto xstrided_view<CT>::data() const noexcept -> const underlying_container_type&
    {
        return m_e.data();
    }

    template <class CT>
    inline auto xstrided_view<CT>::raw_data() noexcept -> value_type*
    {
        return m_e.raw_data();
    }

    template <class CT>
    inline auto xstrided_view<CT>::raw_data() const noexcept -> const value_type*
    {
        return m_e.raw_data();
    }

    template <class CT>
    inline auto xstrided_view<CT>::raw_data_offset() const noexcept -> size_type
    {
        return m_offset;
    }
    //@}

    /**
     * @name Data
     */
    template <class CT>
    inline auto xstrided_view<CT>::operator()() -> reference
    {
        return m_e();
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class CT>
    template <class... Args>
    inline auto xstrided_view<CT>::operator()(Args... args) -> reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = m_offset + data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_e.data()[index];
    }

    /**
     * Returns the element at the specified position in the xstrided_view. 
     * 
     * @param idx the position in the view
     */
    template <class CT>
    template <class... Args>
    inline auto xstrided_view<CT>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = m_offset + data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_e.data()[index];
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator[](const xindex& index) -> reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the xstrided_view.
     * @param first iterator starting the sequence of indices
     * The number of indices in the squence should be equal to or greater 1.
     */
    template <class CT>
    template <class It>
    inline auto xstrided_view<CT>::element(It first, It last) -> reference
    {
        return m_e.data()[m_offset + element_offset<size_type>(strides(), first, last)];
    }

    template <class CT>
    template <class It>
    inline auto xstrided_view<CT>::element(It first, It last) const -> const_reference
    {
        return m_e.data()[m_offset + element_offset<size_type>(strides(), first, last)];
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the xstrided_view to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT>
    template <class O>
    inline bool xstrided_view<CT>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT>
    template <class O>
    inline bool xstrided_view<CT>::is_trivial_broadcast(const O& str) const noexcept
    {
        return str.size() == strides().size() &&
            std::equal(str.cbegin(), str.cend(), strides().begin());
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT>
    template <class ST>
    inline auto xstrided_view<CT>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset);
    }

    template <class CT>
    template <class ST>
    inline auto xstrided_view<CT>::stepper_end(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset, true);
    }

    template <class CT>
    template <class ST>
    inline auto xstrided_view<CT>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT>
    template <class ST>
    inline auto xstrided_view<CT>::stepper_end(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    /**
     * Construct a strided view from an xexpression, shape, strides and offset.
     *
     * @param e xexpression
     * @param shape the shape of the view
     * @param strides the new strides of the view
     * @param offset the offset of the first element in the underlying container
     *
     * @tparam E type of xexpression
     * @tparam I shape and strides type
     *
     * @return the view
     */
    template <class E, class I>
    inline auto strided_view(E&& e, I&& shape, I&& strides, std::size_t offset = 0) noexcept
    {
        using view_type = xstrided_view<xclosure_t<E>>;
        return view_type(std::forward<E>(e), std::forward<I>(shape), std::forward<I>(strides), offset);
    }

    /****************************
     * transpose implementation *
     ****************************/

    namespace detail
    {
        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::none)
        {
            if (container_size(permutation) != e.dimension())
            {
                throw transpose_error("Permutation does not have the same size as shape");
            }

            // permute stride and shape
            using strides_type = typename std::decay_t<E>::strides_type;
            strides_type temp_strides;
            resize_container(temp_strides, e.strides().size());

            using shape_type = typename std::decay_t<E>::shape_type;
            shape_type temp_shape;
            resize_container(temp_shape, e.shape().size());

            // shape_type temp_backstrides;
            // resize_container(temp_backstrides, backstrides().size());

            for (std::size_t i = 0; i < e.shape().size(); ++i)
            {
                if (std::size_t(permutation[i]) >= e.dimension())
                {
                    throw transpose_error("Permutation contains wrong axis");
                }
                temp_shape[i] = e.shape()[permutation[i]];
                temp_strides[i] = e.strides()[permutation[i]];
                // TODO figure out what to do with backstrides
                // temp_backstrides[i] = backstrides()[permutation[i]];
            }
            using view_type = xstrided_view<xclosure_t<E>>;
            return view_type(std::forward<E>(e), std::move(temp_shape), std::move(temp_strides), 0);
        }

        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::full)
        {
            // check if axis appears twice in permutation
            for (std::size_t i = 0; i < container_size(permutation); ++i)
            {
                for (std::size_t j = i + 1; j < container_size(permutation); ++j)
                {
                    if (permutation[i] == permutation[j])
                    {
                        throw transpose_error("Permutation contains axis more than once");
                    }
                }
            }
            return transpose_impl(std::forward<E>(e), std::forward<S>(permutation), check_policy::none());
        }
    }

    template <class E>
    inline auto transpose(E&& e) noexcept
    {
        using shape_type = typename std::decay_t<E>::shape_type;

        shape_type shape;
        resize_container(shape, e.shape().size());
        std::copy(e.shape().rbegin(), e.shape().rend(), shape.begin());

        shape_type strides;
        resize_container(strides, e.strides().size());
        std::copy(e.strides().rbegin(), e.strides().rend(), strides.begin());

        using view_type = xstrided_view<xclosure_t<E>>;
        return view_type(std::forward<E>(e), std::move(shape), std::move(strides), 0);
    }

    /**
     * Returns a transpose view by permuting the xexpression e with @p permutation.
     * @param permutation the sequence containing permutation
     * @param check_policy the check level (check_policy::full() or check_policy::none())
     * @tparam Tag selects the level of error checking on permutation vector defaults to check_policy::none.
     */
    template <class E, class S, class Tag = check_policy::none>
    inline auto transpose(E&& e, S&& permutation, Tag check_policy = Tag())
    {
        return detail::transpose_impl(std::forward<E>(e), std::forward<S>(permutation), check_policy);
    }

#ifdef X_OLD_CLANG
    template <class E, class I, class Tag = check_policy::none>
    inline auto transpose(E&& e, std::initializer_list<I> permutation, Tag check_policy = Tag())
    {
        std::vector<I> perm(permutation);
        return detail::transpose_impl(std::forward<E>(e), std::move(perm), check_policy);
    }
#else
    template <class E, class I, std::size_t N, class Tag = check_policy::none>
    inline auto transpose(E&& e, const I (&permutation)[N], Tag check_policy = Tag())
    {
        return detail::transpose_impl(std::forward<E>(e), permutation, check_policy);
    }
#endif

    /***************************
     * diagonal implementation *
     ***************************/

    namespace detail
    {
        template <class T, std::size_t N>
        inline std::array<T, N - 1> remove_last(const std::array<T, N>& arr)
        {
            std::array<T, N - 1> temp;
            std::copy(arr.begin(), arr.end() - 1, temp.begin());
            return temp;
        }

        template <class T>
        inline T& remove_last(T& arr)
        {
            arr.pop_back();
            return arr;
        }

        template <class E>
        inline auto diagonal_impl(E&& e, int offset, std::size_t axis_1, std::size_t axis_2, check_policy::none)
        {
            using view_type = xstrided_view<xclosure_t<E>>;
            using shape_type = typename std::decay_t<E>::shape_type;
            using strides_type = typename std::decay_t<E>::strides_type;

            shape_type shape = e.shape();
            strides_type strides = e.strides();

            // the following shape calculation code is an almost verbatim adaptation of numpy:
            // https://github.com/numpy/numpy/blob/2aabeafb97bea4e1bfa29d946fbf31e1104e7ae0/numpy/core/src/multiarray/item_selection.c#L1799

            shape_type ret_shape;
            strides_type ret_strides;
            resize_container(ret_shape, shape.size());
            resize_container(ret_strides, strides.size());

            std::size_t dim_1 = shape[axis_1];
            std::size_t dim_2 = shape[axis_2];
            std::size_t stride_1 = strides[axis_1];
            std::size_t stride_2 = strides[axis_2];
            std::size_t offset_stride = 0;

            std::size_t n_dim = e.dimension();

            if (offset >= 0)
            {
                offset_stride = stride_2;
                dim_2 -= offset;
            }
            else
            {
                offset = -offset;
                offset_stride = stride_1;
                dim_1 -= offset;
            }

            auto diag_size = dim_2 < dim_1 ? dim_2 : dim_1;

            auto data_offset = offset * offset_stride;

            std::size_t i = 0;
            for (std::size_t idim = 0; idim < n_dim; ++idim)
            {
                if (idim != axis_1 && idim != axis_2)
                {
                    ret_shape[i] = shape[idim];
                    ret_strides[i] = strides[idim];
                    ++i;
                }
            }

            ret_shape[n_dim - 2] = diag_size;
            ret_strides[n_dim - 2] = stride_1 + stride_2;

            auto&& final_shape = remove_last(ret_shape);
            auto&& final_strides = remove_last(ret_strides);

            return view_type(std::forward<E>(e), std::move(final_shape), std::move(final_strides), data_offset);
        }

        template <class E>
        inline auto diagonal_impl(E&& e, int offset, std::size_t axis_1, std::size_t axis_2, check_policy::full)
        {
            if (e.dimension() < 2)
            {
                throw std::runtime_error("diag requires an xexpression with at least two dimensions");
            }
            if (axis_1 == axis_2)
            {
                throw std::runtime_error("axis_1 and axis_2 cannot be the same");
            }
            else if (axis_1 < 0 || axis_1 >= e.dimension() || axis_2 < 0 || axis_2 >= e.dimension()) {
                throw std::runtime_error("axis_1 and axis_2 must be in range [0, e.dimension())");
            }

            if ((std::size_t) std::abs(offset) > e.shape()[axis_1] && (std::size_t) std::abs(offset) > e.shape()[axis_2])
            {
                // TODO return empty view instead of throwing error
                throw std::runtime_error("Offset larger than dim_1 and dim_2");
            }
            return diagonal_impl(std::forward<E>(e), offset, axis_1, axis_2, check_policy::none());
        }
    }

    /**
     * @brief Returns a view of the diagonal elements of xexpression e.
     * If arr has more than two dimensions, then the axes specified by 
     * axis_1 and axis_2 are used to determine the 2-D sub-array whose 
     * diagonal is returned. The shape of the resulting array can be 
     * determined by removing axis_1 and axis_2 and appending an index 
     * to the right equal to the size of the resulting diagonals.
     *
     * @param arr the input array
     * @param offset offset of the diagonal from the main diagonal. Can
     *               be positive or negative.
     * @param axis_1 Axis to be used as the first axis of the 2-D sub-arrays 
     *               from which the diagonals should be taken. 
     * @param axis_2 Axis to be used as the second axis of the 2-D sub-arrays 
     *               from which the diagonals should be taken.
     * @param check_policy the check level (check_policy::full() or check_policy::none())
     * @tparam Tag selects the level of error checking on. Default: check_policy::none.
     * @returns view with values of the diagonal
     *
     * \code{.cpp}
     * xt::xarray<double> a = {{1, 2, 3},
     *                         {4, 5, 6}
     *                         {7, 8, 9}};
     * auto b = xt::diagonal(a); // => {1, 5, 9}
     * // as this is a view, you can also easily assign new values to 
     * // the diagonal, e.g.
     * b = 10;
     * // => {{ 10,  2,  3}, 
     * //     {  4, 10,  6},
     * //     {  7,  8, 10}}
     * \endcode
     */
    template <class E, class Tag = check_policy::none>
    inline auto diagonal(E&& e, int offset = 0, std::size_t axis_1 = 0, std::size_t axis_2 = 1, Tag check_policy = Tag()) noexcept
    {
        return detail::diagonal_impl(std::forward<E>(e), offset, axis_1, axis_2, check_policy);
    }

    /**
     * @brief Reverse the order of elements in an xexpression along the given axis.
     * Note: A NumPy/Matlab style `flipud(arr)` is equivalent to `xt::flip(arr, 0)`,
     * `fliplr(arr)` to `xt::flip(arr, 1)`.
     * 
     * @param arr the input xexpression
     * @param axis the axis along which elements should be reversed
     *
     * @return view evaluating to reversed array
     */
    template <class E>
    inline auto flip(E&& e, std::size_t axis)
    {
        using view_type = xstrided_view<xclosure_t<E>>;
        using shape_type = typename std::decay_t<E>::shape_type;
        shape_type strides = e.strides();
        shape_type shape = e.shape();
        std::size_t data_offset = (shape[axis] - 1) * strides[axis];
        strides[axis] = -strides[axis];
        return view_type(std::forward<E>(e), shape, strides, data_offset);
    }

}

#endif
