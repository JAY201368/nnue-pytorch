#pragma once

#include "jungle_types.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace jungle
{
    static constexpr std::size_t KiB = 1024;
    static constexpr std::size_t MiB = 1024 * KiB;
    static constexpr std::size_t maxChunkSize = 100 * MiB;

#pragma pack(push, 1)
    struct PackedJungleTrainingDataEntry
    {
        // white 8 + black 8, piece order:
        // ELEPHANT, LION, TIGER, PANTHER, WOLF, DOG, CAT, RAT.
        // Square is 0..62 using rank * 7 + file; 63 means captured.
        std::uint8_t piece_squares[NUM_COLORS * NUM_PIECE_TYPES];
        std::uint8_t side_to_move; // 0 white, 1 black
        std::int16_t score;
        std::uint16_t ply;
        // Result is from side-to-move perspective: -1 loss, 0 draw, 1 win.
        std::int8_t result;
        std::uint8_t flags;
    };
#pragma pack(pop)

    static_assert(sizeof(PackedJungleTrainingDataEntry) == 23);

    [[nodiscard]] inline TrainingDataEntry unpackEntry(const PackedJungleTrainingDataEntry& packed)
    {
        TrainingDataEntry entry;
        for (int color = 0; color < NUM_COLORS; ++color)
        {
            for (int piece = 0; piece < NUM_PIECE_TYPES; ++piece)
            {
                const auto sq = packed.piece_squares[color * NUM_PIECE_TYPES + piece];
                if (sq != CAPTURED_SQUARE && !is_valid_square(sq))
                    throw std::runtime_error("Invalid square in jungle packed entry.");
                entry.pos.piece_squares[color][piece] = sq;
            }
        }
        if (packed.side_to_move > 1)
            throw std::runtime_error("Invalid side_to_move in jungle packed entry.");

        entry.pos.side_to_move = static_cast<Color>(packed.side_to_move);
        entry.score = packed.score;
        entry.ply = packed.ply;
        entry.result = packed.result;
        entry.flags = packed.flags;
        return entry;
    }

    [[nodiscard]] inline PackedJungleTrainingDataEntry packEntry(const TrainingDataEntry& entry)
    {
        PackedJungleTrainingDataEntry packed{};
        for (int color = 0; color < NUM_COLORS; ++color)
        {
            for (int piece = 0; piece < NUM_PIECE_TYPES; ++piece)
            {
                packed.piece_squares[color * NUM_PIECE_TYPES + piece] =
                  entry.pos.piece_squares[color][piece];
            }
        }
        packed.side_to_move = static_cast<std::uint8_t>(entry.pos.side_to_move);
        packed.score = entry.score;
        packed.ply = entry.ply;
        packed.result = static_cast<std::int8_t>(entry.result);
        packed.flags = entry.flags;
        return packed;
    }

    class JungleBinpackFile
    {
    public:
        static constexpr auto openmode = std::ios::in | std::ios::binary;

        explicit JungleBinpackFile(std::string path, std::ios_base::openmode om = openmode) :
            m_path(std::move(path)),
            m_file(m_path, om | std::ios::in | std::ios::binary)
        {
            if (!m_file)
                return;

            const auto cur = m_file.tellg();
            m_file.seekg(0, std::ios_base::end);
            m_sizeBytes = static_cast<std::size_t>(m_file.tellg());
            m_file.seekg(cur, std::ios_base::beg);
        }

        [[nodiscard]] bool hasNextChunk()
        {
            if (!m_file)
                return false;
            m_file.peek();
            return !m_file.eof();
        }

        void seek_to_start()
        {
            m_file.clear();
            m_file.seekg(0, std::ios_base::beg);
        }

        [[nodiscard]] std::vector<unsigned char> readNextChunk()
        {
            const auto size = readChunkHeader();
            if (size % sizeof(PackedJungleTrainingDataEntry) != 0)
                throw std::runtime_error("Jungle binpack chunk is not aligned to fixed record size.");

            std::vector<unsigned char> data(size);
            m_file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
            if (!m_file)
                throw std::runtime_error("Failed to read full jungle binpack chunk.");
            return data;
        }

        [[nodiscard]] std::size_t sizeBytes() const
        {
            return m_sizeBytes;
        }

    private:
        std::string m_path;
        std::fstream m_file;
        std::size_t m_sizeBytes = 0;

        [[nodiscard]] std::uint32_t readChunkHeader()
        {
            unsigned char header[8];
            m_file.read(reinterpret_cast<char*>(header), 8);
            if (!m_file || header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P')
                throw std::runtime_error("Invalid jungle binpack file or chunk.");

            const std::uint32_t size =
              static_cast<std::uint32_t>(header[4])
              | (static_cast<std::uint32_t>(header[5]) << 8)
              | (static_cast<std::uint32_t>(header[6]) << 16)
              | (static_cast<std::uint32_t>(header[7]) << 24);

            if (size > maxChunkSize)
                throw std::runtime_error("Jungle binpack chunk size larger than supported.");

            return size;
        }
    };

    class JungleTrainingDataEntryReader
    {
    public:
        explicit JungleTrainingDataEntryReader(std::string path,
                                               std::ios_base::openmode om = JungleBinpackFile::openmode) :
            m_inputFile(std::move(path), om)
        {
            if (!m_inputFile.hasNextChunk())
                m_isEnd = true;
            else
                m_chunk = m_inputFile.readNextChunk();
        }

        [[nodiscard]] bool hasNext() const
        {
            return !m_isEnd;
        }

        [[nodiscard]] TrainingDataEntry next()
        {
            PackedJungleTrainingDataEntry packed{};
            std::memcpy(&packed, m_chunk.data() + m_offset, sizeof(packed));
            m_offset += sizeof(packed);
            fetchNextChunkIfNeeded();
            return unpackEntry(packed);
        }

    private:
        JungleBinpackFile m_inputFile;
        std::vector<unsigned char> m_chunk;
        std::size_t m_offset = 0;
        bool m_isEnd = false;

        void fetchNextChunkIfNeeded()
        {
            if (m_offset + sizeof(PackedJungleTrainingDataEntry) <= m_chunk.size())
                return;

            if (m_inputFile.hasNextChunk())
            {
                m_chunk = m_inputFile.readNextChunk();
                m_offset = 0;
            }
            else
            {
                m_isEnd = true;
            }
        }
    };
}
