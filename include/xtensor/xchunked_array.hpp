#ifndef XTENSOR_CHUNKED_ARRAY_HPP
#define XTENSOR_CHUNKED_ARRAY_HPP

#include <vector>
#include <array>

#include "xarray.hpp"
#include "xnoalias.hpp"
#include "xstrided_view.hpp"

namespace xt
{

    /*********************************
     * xchunked_semantic declaration *
     *********************************/

    template <class T, class chunk_storage>
    class xchunked_assigner
    {
    public:

        using temporary_type = T;

        template <class E, class DST>
        void build_and_assign_temporary(const xexpression<E>& e, DST& dst);
    };

    template <class D>
    class xchunked_semantic : public xsemantic_base<D>
    {
    public:

        using base_type = xsemantic_base<D>;
        using derived_type = D;
        using temporary_type = typename base_type::temporary_type;

        template <class E>
        derived_type& assign_xexpression(const xexpression<E>& e);

        template <class E>
        derived_type& computed_assign(const xexpression<E>& e);

        template <class E, class F>
        derived_type& scalar_computed_assign(const E& e, F&& f);

    protected:

        xchunked_semantic() = default;
        ~xchunked_semantic() = default;

        xchunked_semantic(const xchunked_semantic&) = default;
        xchunked_semantic& operator=(const xchunked_semantic&) = default;

        xchunked_semantic(xchunked_semantic&&) = default;
        xchunked_semantic& operator=(xchunked_semantic&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>& e);

    private:

        template <class CS>
        xchunked_assigner<temporary_type, CS> get_assigner(const CS&) const;
    };

    /******************************
     * xchunked_array declaration *
     ******************************/

    template <class chunk_storage>
    class xchunked_array;

    template <class chunk_storage>
    struct xcontainer_inner_types<xchunked_array<chunk_storage>>
    {
        using chunk_type = typename chunk_storage::value_type;
        using const_reference = typename chunk_type::const_reference;
        using reference = typename chunk_type::reference;
        using size_type = std::size_t;
        using storage_type = chunk_type;
        using temporary_type = xchunked_array<chunk_storage>;
    };

    template <class chunk_storage>
    struct xiterable_inner_types<xchunked_array<chunk_storage>>
    {
        using chunk_type = typename chunk_storage::value_type;
        using inner_shape_type = typename chunk_type::shape_type;
        using const_stepper = xindexed_stepper<xchunked_array<chunk_storage>, true>;
        using stepper = xindexed_stepper<xchunked_array<chunk_storage>, false>;
    };

    template <class chunk_storage>
    class xchunked_array: public xaccessible<xchunked_array<chunk_storage>>,
                          public xiterable<xchunked_array<chunk_storage>>,
                          public xchunked_semantic<xchunked_array<chunk_storage>>
    {
    public:

        using chunk_storage_type = chunk_storage;
        using chunk_type = typename chunk_storage::value_type;
        using const_reference = typename chunk_type::const_reference;
        using reference = typename chunk_type::reference;
        using self_type = xchunked_array<chunk_storage>;
        using semantic_base = xchunked_semantic<self_type>;
        using iterable_base = xconst_iterable<self_type>;
        using const_stepper = typename iterable_base::const_stepper;
        using stepper = typename iterable_base::stepper;
        using inner_types = xcontainer_inner_types<self_type>;
        using size_type = typename inner_types::size_type;
        using storage_type = typename inner_types::storage_type;
        using value_type = typename storage_type::value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename chunk_type::shape_type;
        using temporary_type = typename inner_types::temporary_type;
        using bool_load_type = xt::bool_load_type<value_type>;
        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        template <class S>
        xchunked_array(chunk_storage_type&& chunks, S&& shape, S&& chunk_shape, layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT);
        ~xchunked_array() = default;

        xchunked_array(const xchunked_array&) = default;
        xchunked_array& operator=(const xchunked_array&) = default;

        xchunked_array(xchunked_array&&) = default;
        xchunked_array& operator=(xchunked_array&&) = default;

        template <class E>
        xchunked_array(const xexpression<E>&e , chunk_storage_type&& chunks, layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT);

        template <class E, class S>
        xchunked_array(const xexpression<E>& e, chunk_storage_type&& chunks, S&& chunk_shape, layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT);

        template <class E>
        xchunked_array& operator=(const xexpression<E>& e);

        const shape_type& shape() const noexcept;
        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

        template <class... Idxs>
        reference operator()(Idxs... idxs);

        template <class... Idxs>
        const_reference operator()(Idxs... idxs) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& s, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type) const noexcept;

        const shape_type& chunk_shape() const;
        chunk_storage_type& chunks();
        const chunk_storage_type& chunks() const;

    private:

        template <class... Idxs>
        using indexes_type = std::pair<std::array<std::size_t, sizeof...(Idxs)>, std::array<std::size_t, sizeof...(Idxs)>>;

        template <class... Idxs>
        using chunk_indexes_type = std::array<std::pair<std::size_t, std::size_t>, sizeof...(Idxs)>;

        template <std::size_t N>
        using static_indexes_type = std::pair<std::array<std::size_t, N>, std::array<std::size_t, N>>;

        using dynamic_indexes_type = std::pair<std::vector<std::size_t>, std::vector<std::size_t>>;

        template <class S1, class S2>
        void resize(S1&& shape, S2&& chunk_shape, layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT);

        template <class... Idxs>
        indexes_type<Idxs...> get_indexes(Idxs... idxs) const;

        template <class Idx>
        std::pair<std::size_t, std::size_t> get_chunk_indexes_in_dimension(std::size_t dim, Idx idx) const;

        template <std::size_t... dims, class... Idxs>
        chunk_indexes_type<Idxs...> get_chunk_indexes(std::index_sequence<dims...>, Idxs... idxs) const;

        template <class T, std::size_t N>
        static_indexes_type<N> unpack(const std::array<T, N> &arr) const;

        template <class It>
        dynamic_indexes_type get_indexes_dynamic(It first, It last) const;

        shape_type m_shape;
        shape_type m_chunk_shape;
        chunk_storage_type m_chunks;
    };

    template<class E>
    constexpr bool is_chunked(const xexpression<E>& e);

    /**
     * Creates an in-memory chunked array.
     * This function returns an uninitialized ``xchunked_array<xarray<T>>``.
     *
     * @tparam T The type of the elements (e.g. double)
     * @tparam L The layout_type of the array
     *
     * @param shape The shape of the array
     * @param chunk_shape The shape of a chunk
     * @param chunk_memory_layout The layout of each chunk (default: XTENSOR_DEFAULT_LAYOUT)
     *
     * @return returns a ``xchunked_array<xarray<T>>`` with the given shape, chunk shape and memory layout.
     */
    template <class T, layout_type L = XTENSOR_DEFAULT_LAYOUT, class S>
    xchunked_array<xarray<xarray<T>>> chunked_array(S&& shape, S&& chunk_shape, layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT);

    template <class T, layout_type L = XTENSOR_DEFAULT_LAYOUT, class S>
    xchunked_array<xarray<xarray<T>>> chunked_array(std::initializer_list<S> shape, std::initializer_list<S> chunk_shape, layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT);

    /**
     * Creates an in-memory chunked array.
     * This function returns a ``xchunked_array<xarray<T>>`` initialized from an expression.
     *
     * @tparam L The layout_type of the array
     *
     * @param e The expression to initialize the chunked array from
     * @param chunk_shape The shape of a chunk
     * @param chunk_memory_layout The layout of each chunk (default: XTENSOR_DEFAULT_LAYOUT)
     *
     * @return returns a ``xchunked_array<xarray<T>>`` from the given expression, with the given chunk shape and memory layout.
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class E, class S>
    xchunked_array<xarray<xarray<typename E::value_type>>>
    chunked_array(const xexpression<E>& e, S&& chunk_shape, layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT);

    /**
     * Creates an in-memory chunked array.
     * This function returns a ``xchunked_array<xarray<T>>`` initialized from an expression.
     *
     * @tparam L The layout_type of the array
     *
     * @param e The expression to initialize the chunked array from
     * @param chunk_memory_layout The layout of each chunk (default: XTENSOR_DEFAULT_LAYOUT)
     *
     * @return returns a ``xchunked_array<xarray<T>>`` from the given expression, with the expression's chunk shape and the given memory layout.
     */
    template <layout_type L = XTENSOR_DEFAULT_LAYOUT, class E>
    xchunked_array<xarray<xarray<typename E::value_type>>>
    chunked_array(const xexpression<E>&e, layout_type chunk_memory_layout = XTENSOR_DEFAULT_LAYOUT);

    /*******************************
     * chunk_helper implementation *
     *******************************/

    namespace detail
    {
        // Workaround for VS2015
        template <class E>
        using try_chunk_shape = decltype(std::declval<E>().chunk_shape());

        template <class E, template <class> class OP, class = void>
        struct chunk_helper_impl
        {
            using is_chunked = std::false_type;
            static const auto& chunk_shape(const xexpression<E>& e)
            {
                return e.derived_cast().shape();
            }

            template <class S1, class S2>
            static void resize(E& chunks, const S1& container_shape, const S2& chunk_shape, layout_type chunk_memory_layout)
            {
                chunks.resize(container_shape);
                for(auto& c: chunks)
                {
                    c.resize(chunk_shape, chunk_memory_layout);
                }
            }
        };

        template <class E, template <class> class OP>
        struct chunk_helper_impl<E, OP, void_t<OP<E>>>
        {
            using is_chunked = std::true_type;
            static const auto& chunk_shape(const xexpression<E>& e)
            {
                return e.derived_cast().chunk_shape();
            }

            template <class S1, class S2>
            static void resize(E& chunks, const S1& container_shape, const S2& /*chunk_shape*/, layout_type /*chunk_memory_layout*/)
            {
                chunks.resize(container_shape);
            }
        };

        template <class E>
        using chunk_helper = chunk_helper_impl<E, try_chunk_shape>;
    }

    template<class E>
    constexpr bool is_chunked(const xexpression<E>&)
    {
        using return_type = typename detail::chunk_helper<E>::is_chunked;
        return return_type::value;
    }

    template <class T, layout_type L, class S>
    inline xchunked_array<xarray<xarray<T>>> chunked_array(S&& shape, S&& chunk_shape, layout_type chunk_memory_layout)
    {
        using chunk_storage = xarray<xarray<T, L>>;
        return xchunked_array<chunk_storage>(chunk_storage(), std::forward<S>(shape), std::forward<S>(chunk_shape), chunk_memory_layout);
    }

    template <class T, layout_type L, class S>
    xchunked_array<xarray<xarray<T>>> chunked_array(std::initializer_list<S> shape, std::initializer_list<S> chunk_shape, layout_type chunk_memory_layout)
    {
        using sh_type = std::vector<std::size_t>;
        auto sh = xtl::forward_sequence<sh_type, std::initializer_list<S>>(shape);
        auto ch_sh = xtl::forward_sequence<sh_type, std::initializer_list<S>>(chunk_shape);
        return chunked_array<T, L, sh_type>(std::move(sh), std::move(ch_sh), chunk_memory_layout);
    }

    template <layout_type L, class E, class S>
    inline xchunked_array<xarray<xarray<typename E::value_type>>>
    chunked_array(const xexpression<E>& e, S&& chunk_shape, layout_type chunk_memory_layout)
    {
        using chunk_storage = xarray<xarray<typename E::value_type, L>>;
        return xchunked_array<chunk_storage>(e, chunk_storage(), std::forward<S>(chunk_shape), chunk_memory_layout);
    }

    template <layout_type L, class E>
    inline xchunked_array<xarray<xarray<typename E::value_type>>>
    chunked_array(const xexpression<E>& e, layout_type chunk_memory_layout)
    {
        using chunk_storage = xarray<xarray<typename E::value_type, L>>;
        return xchunked_array<chunk_storage>(e, chunk_storage(), chunk_memory_layout);
    }

    /************************************
     * xchunked_semantic implementation *
     ************************************/

    template <class T, class CS>
    template <class E, class DST>
    inline void xchunked_assigner<T, CS>::build_and_assign_temporary(const xexpression<E>& e, DST& dst)
    {
        temporary_type tmp(e, CS(), dst.chunk_shape());
        dst = std::move(tmp);
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::assign_xexpression(const xexpression<E>& e) -> derived_type&
    {
        using shape_type = std::decay_t<decltype(this->derived_cast().shape())>;
        using size_type = typename shape_type::size_type;
        const auto& chunk_shape = this->derived_cast().chunk_shape();
        auto& chunks = this->derived_cast().chunks();
        size_t dimension = this->derived_cast().dimension();
        xstrided_slice_vector sv(chunk_shape.size());  // element slice corresponding to chunk
        std::transform(chunk_shape.begin(), chunk_shape.end(), sv.begin(),
                       [](auto size) { return range(0, size); });
        shape_type ic(dimension);  // index of chunk, initialized to 0...
        size_type ci = 0;
        for (auto& chunk: chunks)
        {
            noalias(chunk) = strided_view(e.derived_cast(), sv);
            bool last_chunk = ci == chunks.size() - 1;
            if (!last_chunk)
            {
                size_type di = dimension - 1;
                while (true)
                {
                    if (ic[di] + 1 == chunks.shape()[di])
                    {
                        ic[di] = 0;
                        sv[di] = range(0, chunk_shape[di]);
                        if (di == 0)
                        {
                            break;
                        }
                        else
                        {
                            di--;
                        }
                    }
                    else
                    {
                        ic[di] += 1;
                        sv[di] = range(ic[di] * chunk_shape[di], (ic[di] + 1) * chunk_shape[di]);
                        break;
                    }
                }
            }
            ++ci;
        }
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::computed_assign(const xexpression<E>& e) -> derived_type&
    {
        D& d = this->derived_cast();
        if (e.derived_cast().dimension() > d.dimension()
            || e.derived_cast().shape() > d.shape())
        {
            return operator=(e);
        }
        else
        {
            return assign_xexpression(e);
        }
    }

    template <class D>
    template <class E, class F>
    inline auto xchunked_semantic<D>::scalar_computed_assign(const E& e, F&& f) -> derived_type&
    {
        for (auto& c: this->derived_cast().chunks())
        {
            c.scalar_computed_assign(e, f);
        }
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::operator=(const xexpression<E>& e) -> derived_type&
    {
        D& d = this->derived_cast();
        get_assigner(d.chunks()).build_and_assign_temporary(e, d);
        return d;
    }

    template <class D>
    template <class CS>
    inline auto xchunked_semantic<D>::get_assigner(const CS&) const -> xchunked_assigner<temporary_type, CS>
    {
        return xchunked_assigner<temporary_type, CS>();
    }

    /*********************************
     * xchunked_array implementation *
     *********************************/

    template <class CS>
    template <class S>
    inline xchunked_array<CS>::xchunked_array(CS&& chunks, S&& shape, S&& chunk_shape, layout_type chunk_memory_layout)
        : m_chunks(std::move(chunks))
    {
        resize(std::forward<S>(shape), std::forward<S>(chunk_shape), chunk_memory_layout);
    }

    template <class CS>
    template <class E>
    inline xchunked_array<CS>::xchunked_array(const xexpression<E>& e, CS&& chunks, layout_type chunk_memory_layout)
        : xchunked_array(e, std::move(chunks), detail::chunk_helper<E>::chunk_shape(e), chunk_memory_layout)
    {
    }

    template <class CS>
    template <class E, class S>
    inline xchunked_array<CS>::xchunked_array(const xexpression<E>& e, CS&& chunks, S&& chunk_shape, layout_type chunk_memory_layout)
        : m_chunks(std::move(chunks))
    {
        resize(e.derived_cast().shape(), std::forward<S>(chunk_shape), chunk_memory_layout);
        semantic_base::assign_xexpression(e);
    }

    template <class CS>
    template <class E>
    inline auto xchunked_array<CS>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class CS>
    inline auto xchunked_array<CS>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class CS>
    inline auto xchunked_array<CS>::layout() const noexcept -> layout_type
    {
        return static_layout;
    }

    template <class CS>
    inline bool xchunked_array<CS>::is_contiguous() const noexcept
    {
        return false;
    }

    template <class CS>
    template <class... Idxs>
    inline auto xchunked_array<CS>::operator()(Idxs... idxs) -> reference
    {
        auto ii = get_indexes(idxs...);
        auto& chunk = m_chunks.element(ii.first.cbegin(), ii.first.cend());
        return chunk.element(ii.second.cbegin(), ii.second.cend());
    }

    template <class CS>
    template <class... Idxs>
    inline auto xchunked_array<CS>::operator()(Idxs... idxs) const -> const_reference
    {
        auto ii = get_indexes(idxs...);
        auto& chunk = m_chunks.element(ii.first.cbegin(), ii.first.cend());
        return chunk.element(ii.second.cbegin(), ii.second.cend());
    }

    template <class CS>
    template <class It>
    inline auto xchunked_array<CS>::element(It first, It last) -> reference
    {
        auto ii = get_indexes_dynamic(first, last);
        auto& chunk = m_chunks.element(ii.first.begin(), ii.first.end());
        return chunk.element(ii.second.begin(), ii.second.end());
    }

    template <class CS>
    template <class It>
    inline auto xchunked_array<CS>::element(It first, It last) const -> const_reference
    {
        auto ii = get_indexes_dynamic(first, last);
        auto& chunk = m_chunks.element(ii.first.begin(), ii.first.end());
        return chunk.element(ii.second.begin(), ii.second.end());
    }

    template <class CS>
    template <class S>
    inline bool xchunked_array<CS>::broadcast_shape(S& s, bool) const
    {
        return xt::broadcast_shape(shape(), s);
    }

    template <class CS>
    template <class S>
    inline bool xchunked_array<CS>::has_linear_assign(const S&) const noexcept
    {
        return false;
    }

    template <class CS>
    template <class S>
    inline auto xchunked_array<CS>::stepper_begin(const S& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset);
    }

    template <class CS>
    template <class S>
    inline auto xchunked_array<CS>::stepper_end(const S& shape, layout_type) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset, true);
    }

    template <class CS>
    template <class S>
    inline auto xchunked_array<CS>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset);
    }

    template <class CS>
    template <class S>
    inline auto xchunked_array<CS>::stepper_end(const S& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset, true);
    }

    template <class CS>
    inline auto xchunked_array<CS>::chunks() -> chunk_storage_type&
    {
        return m_chunks;
    }

    template <class CS>
    inline auto xchunked_array<CS>::chunks() const -> const chunk_storage_type&
    {
        return m_chunks;
    }

    template <class CS>
    inline auto xchunked_array<CS>::chunk_shape() const -> const shape_type&
    {
        return m_chunk_shape;
    }

    template <class CS>
    template <class S1, class S2>
    inline void xchunked_array<CS>::resize(S1&& shape, S2&& chunk_shape, layout_type chunk_memory_layout)
    {
        // compute chunk number in each dimension (shape_of_chunks)
        std::vector<std::size_t> shape_of_chunks(shape.size());
        std::transform
        (
            shape.cbegin(), shape.cend(),
            chunk_shape.cbegin(),
            shape_of_chunks.begin(),
            [](auto s, auto cs)
            {
                std::size_t cn = s / cs;
                if (s % cs > 0)
                    cn += std::size_t(1); // edge_chunk
                return cn;
            }
        );

        detail::chunk_helper<CS>::resize(m_chunks, shape_of_chunks, chunk_shape, chunk_memory_layout);

        m_shape = xtl::forward_sequence<shape_type, S1>(shape);
        m_chunk_shape = xtl::forward_sequence<shape_type, S2>(chunk_shape);
    }

    template <class CS>
    template <class... Idxs>
    inline auto xchunked_array<CS>::get_indexes(Idxs... idxs) const -> indexes_type<Idxs...>
    {
        auto chunk_indexes_packed = get_chunk_indexes(std::make_index_sequence<sizeof...(Idxs)>(), idxs...);
        return unpack(chunk_indexes_packed);
    }

    template <class CS>
    template <class Idx>
    inline std::pair<std::size_t, std::size_t> xchunked_array<CS>::get_chunk_indexes_in_dimension(std::size_t dim, Idx idx) const
    {
        std::size_t index_of_chunk = static_cast<size_t>(idx) / m_chunk_shape[dim];
        std::size_t index_in_chunk = static_cast<size_t>(idx) - index_of_chunk * m_chunk_shape[dim];
        return std::make_pair(index_of_chunk, index_in_chunk);
    }

    template <class CS>
    template <std::size_t... dims, class... Idxs>
    inline auto xchunked_array<CS>::get_chunk_indexes(std::index_sequence<dims...>, Idxs... idxs) const
        -> chunk_indexes_type<Idxs...>
    {
        chunk_indexes_type<Idxs...> chunk_indexes = {{get_chunk_indexes_in_dimension(dims, idxs)...}};
        return chunk_indexes;
    }

    template <class CS>
    template <class T, std::size_t N>
    inline auto xchunked_array<CS>::unpack(const std::array<T, N> &arr) const -> static_indexes_type<N>
    {
        std::array<std::size_t, N> arr0;
        std::array<std::size_t, N> arr1;
        for (std::size_t i = 0; i < N; ++i)
        {
            arr0[i] = std::get<0>(arr[i]);
            arr1[i] = std::get<1>(arr[i]);
        }
        return std::make_pair(arr0, arr1);
    }

    template <class CS>
    template <class It>
    inline auto xchunked_array<CS>::get_indexes_dynamic(It first, It last) const -> dynamic_indexes_type
    {
        auto size = static_cast<std::size_t>(std::distance(first, last));
        std::vector<std::size_t> indexes_of_chunk(size);
        std::vector<std::size_t> indexes_in_chunk(size);
        for (std::size_t dim = 0; dim < size; ++dim)
        {
            auto chunk_index = get_chunk_indexes_in_dimension(dim, *first++);
            indexes_of_chunk[dim] = chunk_index.first;
            indexes_in_chunk[dim] = chunk_index.second;
        }
        return std::make_pair(indexes_of_chunk, indexes_in_chunk);
    }
}

#endif
