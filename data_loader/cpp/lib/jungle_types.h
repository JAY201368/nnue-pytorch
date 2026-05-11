#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace jungle
{
    static constexpr int BOARD_FILES = 7;
    static constexpr int BOARD_RANKS = 9;
    static constexpr int NUM_SQUARES = BOARD_FILES * BOARD_RANKS;
    static constexpr int NUM_PIECE_TYPES = 8;
    static constexpr int NUM_COLORS = 2;
    static constexpr std::uint8_t CAPTURED_SQUARE = 63;

    enum class Color : std::uint8_t
    {
        White = 0,
        Black = 1,
    };

    enum class PieceType : std::uint8_t
    {
        Elephant = 0,
        Lion = 1,
        Tiger = 2,
        Panther = 3,
        Wolf = 4,
        Dog = 5,
        Cat = 6,
        Rat = 7,
    };

    struct PieceOnSquare
    {
        Color color;
        PieceType type;
        int square;
    };

    [[nodiscard]] constexpr int square(int file, int rank)
    {
        return rank * BOARD_FILES + file;
    }

    [[nodiscard]] constexpr bool is_valid_square(int sq)
    {
        return sq >= 0 && sq < NUM_SQUARES;
    }

    [[nodiscard]] constexpr int orient_square(Color pov, int sq)
    {
        return pov == Color::White ? sq : NUM_SQUARES - 1 - sq;
    }

    [[nodiscard]] constexpr int relative_owner(Color pov, Color piece_color)
    {
        return pov == piece_color ? 0 : 1;
    }

    [[nodiscard]] inline PieceType piece_type_from_fen_char(char ch)
    {
        switch (ch)
        {
            case 'E':
            case 'e':
                return PieceType::Elephant;
            case 'L':
            case 'l':
                return PieceType::Lion;
            case 'T':
            case 't':
                return PieceType::Tiger;
            case 'P':
            case 'p':
                return PieceType::Panther;
            case 'W':
            case 'w':
                return PieceType::Wolf;
            case 'D':
            case 'd':
                return PieceType::Dog;
            case 'C':
            case 'c':
                return PieceType::Cat;
            case 'R':
            case 'r':
                return PieceType::Rat;
            default:
                throw std::runtime_error("Invalid jungle piece character in FEN.");
        }
    }

    [[nodiscard]] inline char piece_type_to_fen_char(PieceType type, Color color)
    {
        static constexpr char white_chars[NUM_PIECE_TYPES] = {'E', 'L', 'T', 'P', 'W', 'D', 'C', 'R'};
        static constexpr char black_chars[NUM_PIECE_TYPES] = {'e', 'l', 't', 'p', 'w', 'd', 'c', 'r'};
        const int idx = static_cast<int>(type);
        return color == Color::White ? white_chars[idx] : black_chars[idx];
    }

    struct Position
    {
        std::array<std::array<std::uint8_t, NUM_PIECE_TYPES>, NUM_COLORS> piece_squares{};
        Color side_to_move = Color::White;

        Position()
        {
            for (auto& side : piece_squares)
                side.fill(CAPTURED_SQUARE);
        }

        [[nodiscard]] Color sideToMove() const
        {
            return side_to_move;
        }

        [[nodiscard]] int pieceCount() const
        {
            int count = 0;
            for (const auto& side : piece_squares)
                for (std::uint8_t sq : side)
                    if (is_valid_square(sq))
                        ++count;
            return count;
        }

        template<typename Fn>
        void forEachPiece(Fn&& fn) const
        {
            for (int color_idx = 0; color_idx < NUM_COLORS; ++color_idx)
            {
                for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt)
                {
                    const int sq = piece_squares[color_idx][pt];
                    if (is_valid_square(sq))
                    {
                        fn(PieceOnSquare{
                          static_cast<Color>(color_idx),
                          static_cast<PieceType>(pt),
                          sq});
                    }
                }
            }
        }

        [[nodiscard]] std::string fen() const
        {
            char board[NUM_SQUARES];
            std::fill(std::begin(board), std::end(board), '1');
            forEachPiece([&](PieceOnSquare p) {
                board[p.square] = piece_type_to_fen_char(p.type, p.color);
            });

            std::string out;
            for (int rank = BOARD_RANKS - 1; rank >= 0; --rank)
            {
                int empty_run = 0;
                for (int file = 0; file < BOARD_FILES; ++file)
                {
                    const char ch = board[square(file, rank)];
                    if (ch == '1')
                    {
                        ++empty_run;
                    }
                    else
                    {
                        if (empty_run > 0)
                        {
                            out += static_cast<char>('0' + empty_run);
                            empty_run = 0;
                        }
                        out += ch;
                    }
                }
                if (empty_run > 0)
                    out += static_cast<char>('0' + empty_run);
                if (rank != 0)
                    out += '/';
            }
            out += side_to_move == Color::White ? " w" : " b";
            return out;
        }

        [[nodiscard]] static Position fromFen(std::string_view fen)
        {
            Position pos;
            const std::size_t space = fen.find(' ');
            const std::string_view board = fen.substr(0, space);
            int rank = BOARD_RANKS - 1;
            int file = 0;

            for (char ch : board)
            {
                if (ch == '/')
                {
                    if (file != BOARD_FILES)
                        throw std::runtime_error("Invalid jungle FEN rank width.");
                    --rank;
                    file = 0;
                    continue;
                }
                if (ch >= '1' && ch <= '7')
                {
                    file += ch - '0';
                    continue;
                }
                if (rank < 0 || file >= BOARD_FILES)
                    throw std::runtime_error("Invalid jungle FEN board.");

                const Color color = (ch >= 'a' && ch <= 'z') ? Color::Black : Color::White;
                const PieceType type = piece_type_from_fen_char(ch);
                pos.piece_squares[static_cast<int>(color)][static_cast<int>(type)] =
                  static_cast<std::uint8_t>(square(file, rank));
                ++file;
            }

            if (rank != 0 || file != BOARD_FILES)
                throw std::runtime_error("Invalid jungle FEN board shape.");

            if (space != std::string_view::npos && space + 1 < fen.size())
                pos.side_to_move = fen[space + 1] == 'b' ? Color::Black : Color::White;

            return pos;
        }
    };

    struct TrainingDataEntry
    {
        Position pos;
        std::int16_t score = 0;
        std::uint16_t ply = 0;
        std::int16_t result = 0;
        std::uint8_t flags = 0;

        [[nodiscard]] double score_result_prob() const
        {
            const double x = std::clamp(static_cast<double>(score) / 600.0, -8.0, 8.0);
            const double win = 1.0 / (1.0 + std::exp(-x));
            if (result > 0)
                return win;
            if (result < 0)
                return 1.0 - win;
            return 1.0 - std::abs(win - 0.5) * 2.0;
        }
    };
}
