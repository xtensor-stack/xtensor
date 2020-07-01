#include <vector>
#include <array>
#include "xarray.hpp"

namespace xt
{
    template <class chunk_type>
    class xchunked_array
    {
    public:

        using const_reference = typename chunk_type::const_reference;

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto chunk_indexes_packed = get_chunk_indexes(std::make_index_sequence<sizeof...(Idxs)>(), idxs...);
            auto chunk_indexes = unpack(chunk_indexes_packed);
            auto indexes_of_chunk(std::get<0>(chunk_indexes));
            auto indexes_in_chunk(std::get<1>(chunk_indexes));
            chunk_type chunk = m_chunks.element(indexes_of_chunk.cbegin(), indexes_of_chunk.cend());
            const_reference val = chunk.element(indexes_in_chunk.cbegin(), indexes_in_chunk.cend());
            int di = 0;
            for (auto index_of_chunk: indexes_of_chunk)
            {
                auto index_in_chunk = indexes_in_chunk[di];
                di++;
            }
            return val;
        }


        xchunked_array(std::vector<size_t> shape, std::vector<size_t> chunks):
            m_shape(shape),
            m_chunk_shape(chunks)
        {
            std::vector<size_t> shape_chunk(shape.size());
            size_t di = 0;
            for (auto s: shape)
            {
                size_t chunk_nb = s / chunks[di];
                if (s % chunks[di] > 0)
                    chunk_nb += 1;  // edge chunk
                shape_chunk[di] = chunk_nb;
                di++;
            }
            for (auto s: chunks)
            m_chunks.resize(shape_chunk);
        }

    private:

        template <class Dim, class Idx>
        std::tuple<size_t, size_t> get_chunk_indexes_in_dimension(Dim dim, Idx idx) const
        {
            size_t index_of_chunk = idx / m_chunk_shape[dim];
            size_t index_in_chunk = idx - index_of_chunk * m_chunk_shape[dim];
            return std::make_tuple(index_of_chunk, index_in_chunk);
        }

        template <size_t... dims, class... Idxs>
        std::array<std::tuple<size_t, size_t>, sizeof...(Idxs)>
        get_chunk_indexes(std::index_sequence<dims...>, Idxs... idxs) const
        {
            std::array<std::tuple<size_t, size_t>, sizeof...(Idxs)> chunk_indexes = {{get_chunk_indexes_in_dimension(dims, idxs)...}};
            return chunk_indexes;
        }

        template <class T, std::size_t N>
        std::tuple<std::array<size_t, N>, std::array<size_t, N>> unpack(std::array<T, N> &arr) const
        {
            std::array<size_t, N> arr0;
            std::array<size_t, N> arr1;
            for (size_t i = 0; i < N; ++i)
            {
                arr0[i] = std::get<0>(arr[i]);
                arr1[i] = std::get<1>(arr[i]);
            }
            return std::make_tuple(arr0, arr1);
        }

        xt::xarray<chunk_type> m_chunks;
        std::vector<size_t> m_shape;
        std::vector<size_t> m_chunk_shape;
    };

}
