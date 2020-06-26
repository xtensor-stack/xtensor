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
            std::cout << "Accessing chunked array at";
            auto chunk_indexes = get_chunk_indexes(std::make_index_sequence<sizeof...(Idxs)>(), idxs...);
            std::cout << std::endl;
            auto indexes_of_chunk(std::get<0>(chunk_indexes));
            auto indexes_in_chunk(std::get<1>(chunk_indexes));
            chunk_type chunk = m_chunks.element(indexes_of_chunk.cbegin(), indexes_of_chunk.cend());
            const_reference val = chunk.element(indexes_in_chunk.cbegin(), indexes_in_chunk.cend());
            int di = 0;
            for (auto index_of_chunk: indexes_of_chunk)
            {
                auto index_in_chunk = indexes_in_chunk[di];
                std::cout << "Dimension " << di << ": chunk " << index_of_chunk << " at " << index_in_chunk << std::endl;
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
            std::cout << "Creating chunked array with shape";
            for (auto s: shape)
            {
                std::cout << " " << s;
                size_t chunk_nb = s / chunks[di];
                if (s % chunks[di] > 0)
                    chunk_nb += 1;  // edge chunk
                shape_chunk[di] = chunk_nb;
                di++;
            }
            std::cout << ", chunk shape";
            for (auto s: chunks)
                std::cout << " " << s;
            std::cout << std::endl;
            m_chunks.resize(shape_chunk);
        }

    private:

        template <class Dim, class Idx>
        size_t get_index_of_chunk_in_dimension(Dim dim, Idx idx) const
        {
            size_t index_of_chunk = idx / m_chunk_shape[dim];
            return index_of_chunk;
        }

        template <class Dim, class Idx>
        size_t get_index_in_chunk_in_dimension(Dim dim, Idx idx) const
        {
            std::cout << " " << dim;
            size_t index_of_chunk = get_index_of_chunk_in_dimension(dim, idx);
            size_t index_in_chunk = idx - index_of_chunk * m_chunk_shape[dim];
            return index_in_chunk;
        }

        template <size_t... dims, class... Idxs>
        std::tuple<std::array<size_t, sizeof...(Idxs)>, std::array<size_t, sizeof...(Idxs)>>
        get_chunk_indexes(std::index_sequence<dims...>, Idxs... idxs) const
        {
            std::array<size_t, sizeof...(Idxs)> indexes_of_chunk = {{get_index_of_chunk_in_dimension(dims, idxs)...}};
            std::array<size_t, sizeof...(Idxs)> indexes_in_chunk = {{get_index_in_chunk_in_dimension(dims, idxs)...}};
            return std::make_tuple(indexes_of_chunk, indexes_in_chunk);
        }

        xt::xarray<chunk_type> m_chunks;
        std::vector<size_t> m_shape;
        std::vector<size_t> m_chunk_shape;
    };

}
