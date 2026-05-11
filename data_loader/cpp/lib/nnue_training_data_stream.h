#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "jungle_binpack.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace training_data
{
    using jungle::TrainingDataEntry;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size())
            return false;
        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;

        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                    break;
                vec.emplace_back(*v);
            }
        }

        virtual void fill_threadsafe(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            std::lock_guard<std::mutex> lock(fill_lock);
            this->fill(vec, n);
        }

        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}

    private:
        std::mutex fill_lock;
    };

    struct JungleBinpackInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "binpack";

        JungleBinpackInputStream(std::vector<std::string> filenames,
                                 bool cyclic,
                                 std::function<bool(const TrainingDataEntry&)> skipPredicate,
                                 int rank = 0,
                                 int world_size = 1) :
            m_filenames(std::move(filenames)),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate)),
            m_rank(rank),
            m_world_size(std::max(1, world_size))
        {
            if (m_filenames.empty())
                m_eof.store(true, std::memory_order_release);
            else
                open_current_file();
        }

        std::optional<TrainingDataEntry> next() override
        {
            for (;;)
            {
                if (!m_reader || !m_reader->hasNext())
                {
                    if (!advance_file())
                    {
                        m_eof.store(true, std::memory_order_release);
                        return std::nullopt;
                    }
                    continue;
                }

                auto entry = m_reader->next();
                const std::uint64_t ordinal = m_record_ordinal++;
                if (ordinal % static_cast<std::uint64_t>(m_world_size)
                    != static_cast<std::uint64_t>(m_rank))
                {
                    continue;
                }
                if (!m_skipPredicate || !m_skipPredicate(entry))
                    return entry;
            }
        }

        bool eof() const override
        {
            return m_eof.load();
        }

    private:
        std::vector<std::string> m_filenames;
        std::size_t m_file_index = 0;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
        int m_rank;
        int m_world_size;
        std::uint64_t m_record_ordinal = 0;
        std::unique_ptr<jungle::JungleTrainingDataEntryReader> m_reader;
        std::atomic<bool> m_eof{false};

        void open_current_file()
        {
            m_reader =
              std::make_unique<jungle::JungleTrainingDataEntryReader>(m_filenames[m_file_index], openmode);
        }

        [[nodiscard]] bool advance_file()
        {
            if (m_filenames.empty())
                return false;

            ++m_file_index;
            if (m_file_index >= m_filenames.size())
            {
                if (!m_cyclic)
                    return false;
                m_file_index = 0;
            }

            open_current_file();
            return m_reader->hasNext();
        }
    };

    inline std::unique_ptr<BasicSfenInputStream>
    open_sfen_input_file_parallel(int concurrency,
                                  const std::vector<std::string>& filenames,
                                  bool cyclic,
                                  std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr,
                                  int rank = 0,
                                  int world_size = 1)
    {
        (void) concurrency;
        if (!filenames.empty() && has_extension(filenames[0], JungleBinpackInputStream::extension))
        {
            return std::make_unique<JungleBinpackInputStream>(
              filenames, cyclic, std::move(skipPredicate), rank, world_size);
        }
        return nullptr;
    }
}

#endif
