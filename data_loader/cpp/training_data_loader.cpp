#include "training_data_loader_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>

#include "lib/rng.h"

using namespace jungle;

// ---------------------------------------------------------
// Jungle feature extractors
// ---------------------------------------------------------

struct JunglePieceSquare
{
    static constexpr std::string_view NAME = "JunglePieceSquare";
    static constexpr int NUM_SQ = NUM_SQUARES;
    static constexpr int NUM_PT = NUM_PIECE_TYPES * NUM_COLORS;
    static constexpr int INPUTS = NUM_SQ * NUM_PT;
    static constexpr int MAX_ACTIVE_FEATURES = 16;

    static int feature_index(Color pov, int sq, PieceType piece_type, Color piece_color)
    {
        const int owner = relative_owner(pov, piece_color);
        return orient_square(pov, sq)
             + NUM_SQUARES * (static_cast<int>(piece_type) + NUM_PIECE_TYPES * owner);
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color pov)
    {
        int j = 0;
        e.pos.forEachPiece([&](PieceOnSquare p) {
            values[j] = 1.0f;
            features[j] = feature_index(pov, p.square, p.type, p.color);
            ++j;
        });
        return {j, INPUTS};
    }
};

struct JunglePieceSquareExtractor: IFeatureExtractor
{
    int inputs() const override { return JunglePieceSquare::INPUTS; }
    int max_active_features() const override { return JunglePieceSquare::MAX_ACTIVE_FEATURES; }
    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int* features,
                                             float* values,
                                             Color color) const override
    {
        return JunglePieceSquare::fill_features_sparse(e, features, values, color);
    }
};

struct JunglePieceTerrain
{
    static constexpr std::string_view NAME = "JunglePieceTerrain";
    static constexpr int NUM_TERRAIN_TAGS = 8;
    static constexpr int INPUTS = NUM_COLORS * NUM_PIECE_TYPES * NUM_TERRAIN_TAGS;
    static constexpr int MAX_ACTIVE_FEATURES = 16 * 3;

    enum TerrainTag
    {
        TERRAIN_LAND = 0,
        TERRAIN_WATER = 1,
        TERRAIN_OWN_TRAP = 2,
        TERRAIN_ENEMY_TRAP = 3,
        TERRAIN_OWN_DEN = 4,
        TERRAIN_ENEMY_DEN = 5,
        TERRAIN_OWN_DEN_ADJACENT = 6,
        TERRAIN_ENEMY_DEN_ADJACENT = 7,
    };

    static constexpr int OWN_DEN = square(3, 0);
    static constexpr int ENEMY_DEN = square(3, 8);

    static bool is_water(int oriented_sq)
    {
        const int file = oriented_sq % BOARD_FILES;
        const int rank = oriented_sq / BOARD_FILES;
        return (file == 1 || file == 2 || file == 4 || file == 5)
            && (rank == 3 || rank == 4 || rank == 5);
    }

    static bool is_own_trap(int oriented_sq)
    {
        return oriented_sq == square(2, 0) || oriented_sq == square(4, 0)
            || oriented_sq == square(3, 1);
    }

    static bool is_enemy_trap(int oriented_sq)
    {
        return oriented_sq == square(2, 8) || oriented_sq == square(4, 8)
            || oriented_sq == square(3, 7);
    }

    template<typename Fn>
    static void for_each_terrain_tag(int oriented_sq, Fn&& fn)
    {
        if (is_water(oriented_sq))
            fn(TERRAIN_WATER);
        else
            fn(TERRAIN_LAND);

        if (is_own_trap(oriented_sq))
        {
            fn(TERRAIN_OWN_TRAP);
            fn(TERRAIN_OWN_DEN_ADJACENT);
        }
        if (is_enemy_trap(oriented_sq))
        {
            fn(TERRAIN_ENEMY_TRAP);
            fn(TERRAIN_ENEMY_DEN_ADJACENT);
        }
        if (oriented_sq == OWN_DEN)
            fn(TERRAIN_OWN_DEN);
        if (oriented_sq == ENEMY_DEN)
            fn(TERRAIN_ENEMY_DEN);
    }

    static std::pair<int, int>
    fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color pov)
    {
        int j = 0;
        e.pos.forEachPiece([&](PieceOnSquare p) {
            const int owner = relative_owner(pov, p.color);
            const int plane = static_cast<int>(p.type) + NUM_PIECE_TYPES * owner;
            const int base = NUM_TERRAIN_TAGS * plane;
            const int oriented_sq = orient_square(pov, p.square);
            for_each_terrain_tag(oriented_sq, [&](int tag) {
                values[j] = 1.0f;
                features[j] = base + tag;
                ++j;
            });
        });
        return {j, INPUTS};
    }
};

struct JunglePieceTerrainExtractor: IFeatureExtractor
{
    int inputs() const override { return JunglePieceTerrain::INPUTS; }
    int max_active_features() const override { return JunglePieceTerrain::MAX_ACTIVE_FEATURES; }
    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int* features,
                                             float* values,
                                             Color color) const override
    {
        return JunglePieceTerrain::fill_features_sparse(e, features, values, color);
    }
};

struct ComposedFeatureExtractor: IFeatureExtractor
{
    std::vector<std::unique_ptr<IFeatureExtractor>> extractors;
    int m_inputs;
    int m_max_active;

    explicit ComposedFeatureExtractor(std::vector<std::unique_ptr<IFeatureExtractor>> exts) :
        extractors(std::move(exts)),
        m_inputs(0),
        m_max_active(0)
    {
        for (auto& e : extractors)
        {
            m_inputs += e->inputs();
            m_max_active += e->max_active_features();
        }
    }

    int inputs() const override { return m_inputs; }
    int max_active_features() const override { return m_max_active; }

    std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e,
                                             int* features,
                                             float* values,
                                             Color color) const override
    {
        int total_written = 0;
        int input_offset = 0;

        for (auto& ext : extractors)
        {
            auto [written, ext_inputs] =
              ext->fill_features_sparse(e, features + total_written, values + total_written, color);
            for (int i = 0; i < written; ++i)
                features[total_written + i] += input_offset;

            input_offset += ext_inputs;
            total_written += written;
        }

        return {total_written, m_inputs};
    }
};

static std::unique_ptr<IFeatureExtractor> make_single_extractor(std::string_view name)
{
    if (name == JunglePieceSquare::NAME)
        return std::make_unique<JunglePieceSquareExtractor>();
    if (name == JunglePieceTerrain::NAME)
        return std::make_unique<JunglePieceTerrainExtractor>();
    return nullptr;
}

std::shared_ptr<IFeatureExtractor> get_feature(std::string_view name)
{
    std::vector<std::unique_ptr<IFeatureExtractor>> components;
    std::size_t start = 0;

    while (start < name.size())
    {
        auto pos = name.find('+', start);
        auto part = name.substr(start, pos == std::string_view::npos ? pos : pos - start);
        auto ext = make_single_extractor(part);

        if (!ext)
        {
            std::cerr << "Unknown feature component: " << part << std::endl;
            return nullptr;
        }

        components.push_back(std::move(ext));
        start = (pos == std::string_view::npos) ? name.size() : pos + 1;
    }

    if (components.empty())
        return nullptr;

    if (components.size() == 1)
        return std::shared_ptr<IFeatureExtractor>(std::move(components[0]));

    return std::make_shared<ComposedFeatureExtractor>(std::move(components));
}

// ---------------------------------------------------------
// Class implementations
// ---------------------------------------------------------

SparseBatch::SparseBatch(const IFeatureExtractor& feature_set,
                         const std::vector<TrainingDataEntry>& entries)
#ifdef NNUE_LOADER_STATISTICS
    :
    entries_copy(entries)
#endif
{
    num_inputs = feature_set.inputs();
    size = entries.size();
    max_active_features = feature_set.max_active_features();
    is_white = new float[size];
    outcome = new float[size];
    score = new float[size];
    white = new int[size * max_active_features];
    black = new int[size * max_active_features];
    white_values = new float[size * max_active_features];
    black_values = new float[size * max_active_features];
    psqt_indices = new int[size];
    layer_stack_indices = new int[size];

    num_active_white_features = 0;
    num_active_black_features = 0;

    for (int i = 0; i < size * max_active_features; ++i)
    {
        white[i] = -1;
        black[i] = -1;
        white_values[i] = 0.0f;
        black_values[i] = 0.0f;
    }

    for (int i = 0; i < size; ++i)
        fill_entry(feature_set, i, entries[i]);
}

SparseBatch::~SparseBatch()
{
    delete[] is_white;
    delete[] outcome;
    delete[] score;
    delete[] white;
    delete[] black;
    delete[] white_values;
    delete[] black_values;
    delete[] psqt_indices;
    delete[] layer_stack_indices;
}

void SparseBatch::fill_entry(const IFeatureExtractor& fs, int i, const TrainingDataEntry& e)
{
    is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
    outcome[i] = (e.result + 1.0f) / 2.0f;
    score[i] = e.score;

    const int piece_count = e.pos.pieceCount();
    const int bucket = std::clamp((piece_count - 1) / 2, 0, 7);
    psqt_indices[i] = bucket;
    layer_stack_indices[i] = bucket;

    fill_features(fs, i, e);
}

void SparseBatch::fill_features(const IFeatureExtractor& fs, int i, const TrainingDataEntry& e)
{
    const int offset = i * max_active_features;
    num_active_white_features +=
      fs.fill_features_sparse(e, white + offset, white_values + offset, Color::White).first;
    num_active_black_features +=
      fs.fill_features_sparse(e, black + offset, black_values + offset, Color::Black).first;
}

int FeaturedBatchStream::calculate_num_reader_threads(int concurrency)
{
    if (worker_thread_ratio >= 1)
        return 1;
    return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FeaturedBatchStream::calculate_num_worker_threads(int concurrency)
{
    if (worker_thread_ratio <= 0)
        return 1;
    return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FeaturedBatchStream::FeaturedBatchStream(
  std::shared_ptr<IFeatureExtractor> feature_set,
  int concurrency,
  const std::vector<std::string>& filenames,
  int batch_size,
  bool cyclic,
  std::function<bool(const TrainingDataEntry&)> skipPredicate,
  int rank,
  int world_size) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames,
             cyclic,
             skipPredicate,
             rank,
             world_size),
    m_feature_set(std::move(feature_set)),
    m_batch_size(batch_size),
    m_concurrency(concurrency),
    m_num_workers(calculate_num_worker_threads(concurrency))
{
    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load())
        {
            entries.clear();
            BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
            if (entries.empty())
                break;

            auto batch = new SparseBatch(*m_feature_set, entries);

            {
                std::unique_lock lock(m_batch_mutex);
                m_batches_not_full.wait(lock, [this]() {
                    return m_batches.size() < m_concurrency + 1 || m_stop_flag.load();
                });
                m_batches.emplace_back(batch);
                lock.unlock();
                m_batches_any.notify_one();
            }
        }
        m_num_workers.fetch_sub(1);
        m_batches_any.notify_one();
    };

    const int num_worker_threads = calculate_num_worker_threads(concurrency);
    for (int i = 0; i < num_worker_threads; ++i)
        m_workers.emplace_back(worker);
}

FeaturedBatchStream::~FeaturedBatchStream()
{
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
    for (auto& batch : m_batches)
        delete batch;
}

SparseBatch* FeaturedBatchStream::next()
{
    std::unique_lock lock(m_batch_mutex);
    m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });
    if (!m_batches.empty())
    {
        auto batch = m_batches.front();
        m_batches.pop_front();
        lock.unlock();
        m_batches_not_full.notify_one();
        return batch;
    }
    return nullptr;
}

Fen::Fen() :
    m_fen(nullptr)
{
}

Fen::Fen(const std::string& fen) :
    m_size(fen.size()),
    m_fen(new char[fen.size() + 1])
{
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
}

Fen& Fen::operator=(const std::string& fen)
{
    if (m_fen != nullptr)
        delete[] m_fen;
    m_size = fen.size();
    m_fen = new char[fen.size() + 1];
    std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    return *this;
}

Fen::~Fen()
{
    delete[] m_fen;
}

FenBatch::FenBatch(const std::vector<TrainingDataEntry>& entries) :
    m_size(entries.size()),
    m_fens(new Fen[entries.size()])
{
    for (int i = 0; i < m_size; ++i)
        m_fens[i] = entries[i].pos.fen();
}

FenBatch::~FenBatch()
{
    delete[] m_fens;
}

int FenBatchStream::calculate_num_reader_threads(int concurrency)
{
    if (worker_thread_ratio >= 1)
        return 1;
    return std::max(1, concurrency - calculate_num_worker_threads(concurrency));
}

int FenBatchStream::calculate_num_worker_threads(int concurrency)
{
    if (worker_thread_ratio <= 0)
        return 1;
    return std::max(1, static_cast<int>(std::floor(concurrency * worker_thread_ratio)));
}

FenBatchStream::FenBatchStream(int concurrency,
                               const std::vector<std::string>& filenames,
                               int batch_size,
                               bool cyclic,
                               std::function<bool(const TrainingDataEntry&)> skipPredicate,
                               int rank,
                               int world_size) :
    BaseType(calculate_num_reader_threads(concurrency),
             filenames,
             cyclic,
             skipPredicate,
             rank,
             world_size),
    m_concurrency(concurrency),
    m_batch_size(batch_size),
    m_num_workers(calculate_num_worker_threads(concurrency))
{
    m_stop_flag.store(false);

    auto worker = [this]() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(m_batch_size);

        while (!m_stop_flag.load())
        {
            entries.clear();
            BaseType::m_stream->fill_threadsafe(entries, m_batch_size);
            if (entries.empty())
                break;

            auto batch = new FenBatch(entries);
            {
                std::unique_lock lock(m_batch_mutex);
                m_batches_not_full.wait(lock, [this]() {
                    return m_batches.size() < m_concurrency + 1 || m_stop_flag.load();
                });
                m_batches.emplace_back(batch);
                lock.unlock();
                m_batches_any.notify_one();
            }
        }
        m_num_workers.fetch_sub(1);
        m_batches_any.notify_one();
    };

    const int num_worker_threads = calculate_num_worker_threads(concurrency);
    for (int i = 0; i < num_worker_threads; ++i)
        m_workers.emplace_back(worker);
}

FenBatchStream::~FenBatchStream()
{
    m_stop_flag.store(true);
    m_batches_not_full.notify_all();
    for (auto& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
    for (auto& batch : m_batches)
        delete batch;
}

FenBatch* FenBatchStream::next()
{
    std::unique_lock lock(m_batch_mutex);
    m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });
    if (!m_batches.empty())
    {
        auto batch = m_batches.front();
        m_batches.pop_front();
        lock.unlock();
        m_batches_not_full.notify_one();
        return batch;
    }
    return nullptr;
}

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config)
{
    if (!config.wld_filtered && config.random_fen_skipping <= 0 && config.early_fen_skipping < 0
        && config.soft_early_fen_skipping <= 0)
    {
        return nullptr;
    }

    double skip_prob = 0.0;
    uint64_t random_skip_threshold = 0;
    if (config.random_fen_skipping > 0)
    {
        skip_prob = double(config.random_fen_skipping) / (config.random_fen_skipping + 1);
        random_skip_threshold = static_cast<uint64_t>(skip_prob * static_cast<double>(~0ULL));
    }

    std::array<double, 17> target_pc_weights_lut{};
    double target_pc_weights_total = 0.0;
    auto desired_piece_count_weights = [&config](int pc) -> double {
        double x = static_cast<double>(pc);
        double y[5] = {config.pc_y0, config.pc_y1, config.pc_y2, config.pc_y3, config.pc_y4};
        if (x <= 0)
            return y[0];
        if (x >= 16)
            return y[2];
        const int i = static_cast<int>(x / 8.0);
        const double x0 = i * 8.0;
        const double t = (x - x0) / 8.0;
        return std::max(0.0, y[i] + t * (y[i + 1] - y[i]));
    };
    for (int i = 0; i <= 16; ++i)
    {
        target_pc_weights_lut[i] = desired_piece_count_weights(i);
        target_pc_weights_total += target_pc_weights_lut[i];
    }
    if (target_pc_weights_total <= 0.0)
    {
        target_pc_weights_lut.fill(1.0);
        target_pc_weights_total = static_cast<double>(target_pc_weights_lut.size());
    }

    std::vector<double> early_ply_accept_prob;
    if (config.soft_early_fen_skipping > 0)
    {
        const size_t lut_size = static_cast<size_t>(config.soft_early_fen_skipping) + 1;
        early_ply_accept_prob.resize(lut_size);
        for (size_t i = 0; i < lut_size; ++i)
        {
            const double ply = static_cast<double>(i);
            const double denom = std::max(1.0, static_cast<double>(config.soft_early_fen_skipping));
            early_ply_accept_prob[i] = std::clamp(ply / denom, 0.0, 1.0);
        }
    }

    return [config, random_skip_threshold, target_pc_weights_lut, target_pc_weights_total,
            early_ply_accept_prob = std::move(early_ply_accept_prob)](const TrainingDataEntry& e) {
        static constexpr int VALUE_NONE = 32002;
        if (e.score == VALUE_NONE)
            return true;
        if (e.ply <= config.early_fen_skipping)
            return true;

        auto& prng = rng::get_thread_local_rng();
        if (config.random_fen_skipping && (prng() < random_skip_threshold))
            return true;

        if (config.wld_filtered)
        {
            uint64_t wld_skip_threshold =
              static_cast<uint64_t>((1.0 - e.score_result_prob()) * static_cast<double>(~0ULL));
            if (prng() < wld_skip_threshold)
                return true;
        }

        if (config.soft_early_fen_skipping > 0 && e.ply < config.soft_early_fen_skipping)
        {
            uint64_t ply_reject_threshold = static_cast<uint64_t>(
              (1.0 - early_ply_accept_prob[e.ply]) * static_cast<double>(~0ULL));
            if (prng() < ply_reject_threshold)
                return true;
        }

        const int pc = e.pos.pieceCount();
        if (pc < 0 || pc > 16)
            return true;

        static thread_local double alpha = 1.0;
        static thread_local double pc_history_all[17] = {0};
        static thread_local uint64_t step_count = 0;
        static thread_local double pc_history_all_total = 0;

        pc_history_all[pc] += 1.0;
        pc_history_all_total += 1.0;
        step_count++;

        if (step_count == 100 || step_count == 500 || step_count == 1000
            || (step_count > 1000 && step_count % 5000 == 0))
        {
            double min_ratio = std::numeric_limits<double>::infinity();
            bool found_valid = false;
            for (int i = 0; i <= 16; ++i)
            {
                if (target_pc_weights_lut[i] > 0.0 && pc_history_all[i] > 0.0)
                {
                    const double current_ratio =
                      (pc_history_all_total * target_pc_weights_lut[i])
                      / (target_pc_weights_total * pc_history_all[i]);
                    min_ratio = std::min(min_ratio, current_ratio);
                    found_valid = true;
                }
            }
            if (found_valid && min_ratio > 0.0)
                alpha = 0.025 / min_ratio;
        }

        double accept_prob = 0.0;
        if (target_pc_weights_lut[pc] > 0.0 && pc_history_all[pc] > 0.0)
        {
            const double current_ratio = (pc_history_all_total * target_pc_weights_lut[pc])
                                       / (target_pc_weights_total * pc_history_all[pc]);
            accept_prob = alpha * current_ratio;
        }
        accept_prob = std::clamp(accept_prob, 0.0, 1.0);

        uint64_t reject_threshold =
          static_cast<uint64_t>((1.0 - accept_prob) * static_cast<double>(~0ULL));
        return prng() < reject_threshold;
    };
}
