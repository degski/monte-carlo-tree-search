// Petter Strandmark 2013
// petter.strandmark@gmail.com

#include <algorithm>
#include <iostream>

#include <mcts.h>

#include "multi_array.hpp"

template<std::size_t NumRows = 6, std::size_t NumCols = 7>
class ConnectFourState {
    public:
    using Move  = int;
    using Moves = sax::compact_vector<Move, std::int64_t, NumCols, NumCols>;
    using Board = ma::MatrixCM<char, NumRows, NumCols>;

    using ZobristHash     = std::uint64_t;
    using ZobristHashKeys = ma::Cube<ZobristHash, 2, NumRows, NumCols, 1>; // 2 players, 1 based.

    ConnectFourState ( ) noexcept : player_to_move ( 1 ), board ( player_markers[ 0 ] ), last_col ( -1 ), last_row ( -1 ) {}

    // Returns the hash of the board xor'ed with the player's hash.
    // For outside consumption.
    ZobristHash zobrist ( ) const noexcept {
        // Order of hashes (it is inverted) is not relevant, as long it's the same every time.
        return m_zobrist_hash ^ m_zobrist_player_keys[ player_to_move ];
    }

    void do_hash_move ( Move move ) {
        attest ( 0 <= move && move < NumCols );
        attest ( board ( 0, move ) == player_markers[ 0 ] );

        int row = NumRows - 1;
        while ( board ( row, move ) != player_markers[ 0 ] )
            row--;

        board ( row, move ) = player_markers[ player_to_move ];
        last_col            = move;
        last_row            = row;

        player_to_move = 3 - player_to_move;
    }

    void do_move ( Move move ) {
        attest ( 0 <= move && move < NumCols );
        attest ( board ( 0, move ) == player_markers[ 0 ] );

        int row = NumRows - 1;
        while ( board ( row, move ) != player_markers[ 0 ] )
            row--;

        m_zobrist_hash ^=
            m_zobrist_keys.at ( player_to_move, row, move ); // player_to_move is here the player who is makeing a move.
        board ( row, move ) = player_markers[ player_to_move ];
        last_col            = move;
        last_row            = row;

        player_to_move = 3 - player_to_move;
    }

    template<typename RandomEngine>
    void do_random_move ( RandomEngine * engine ) {
        dattest ( has_moves ( ) );
        sax::uniform_int_distribution<Move> moves ( 0, NumCols - 1 );

        while ( true ) {
            auto move = moves ( *engine );
            if ( board ( 0, move ) == player_markers[ 0 ] ) {
                do_move ( move );
                return;
            }
        }
    }

    bool has_moves ( ) const {
        char winner = get_winner ( );
        if ( winner != player_markers[ 0 ] )
            return false;
        for ( int col = 0; col < NumCols; ++col )
            if ( board ( 0, col ) == player_markers[ 0 ] )
                return true;
        return false;
    }

    [[nodiscard]] Moves get_moves ( ) const {
        Moves moves;
        if ( get_winner ( ) != player_markers[ 0 ] )
            return moves;
        // moves.reserve ( NumCols ); no need to reserve, first allocation will be max.
        for ( int col = 0; col < NumCols; ++col )
            if ( board ( 0, col ) == player_markers[ 0 ] )
                moves.push_back ( col );
        return moves;
    }

    [[nodiscard]] char get_winner ( ) const noexcept {
        if ( last_col < 0 )
            return player_markers[ 0 ];
        // We only need to check around the last piece played.
        auto piece = board ( last_row, last_col );
        // X X X X
        int left = 0, right = 0;
        for ( int col = last_col - 1; col >= 0 && board ( last_row, col ) == piece; --col )
            left++;
        for ( int col = last_col + 1; col < NumCols && board ( last_row, col ) == piece; ++col )
            right++;
        if ( left + 1 + right >= 4 )
            return piece;
        // X
        // X
        // X
        // X
        int up = 0, down = 0;
        for ( int row = last_row - 1; row >= 0 && board ( row, last_col ) == piece; --row )
            up++;
        for ( int row = last_row + 1; row < NumRows && board ( row, last_col ) == piece; ++row )
            down++;
        if ( up + 1 + down >= 4 )
            return piece;
        // X
        //  X
        //   X
        //    X
        up   = 0;
        down = 0;
        for ( int row = last_row - 1, col = last_col - 1; row >= 0 && col >= 0 && board ( row, col ) == piece; --row, --col )
            up++;
        for ( int row = last_row + 1, col = last_col + 1; row < NumRows && col < NumCols && board ( row, col ) == piece;
              ++row, ++col )
            down++;
        if ( up + 1 + down >= 4 )
            return piece;
        //    X
        //   X
        //  X
        // X
        up   = 0;
        down = 0;
        for ( int row = last_row + 1, col = last_col - 1; row < NumRows && col >= 0 && board ( row, col ) == piece; ++row, --col )
            up++;
        for ( int row = last_row - 1, col = last_col + 1; row >= 0 && col < NumCols && board ( row, col ) == piece; --row, ++col )
            down++;
        if ( up + 1 + down >= 4 )
            return piece;
        return player_markers[ 0 ];
    }

    double get_result ( int current_player_to_move ) const {
        dattest ( not has_moves ( ) );

        auto winner = get_winner ( );
        if ( winner == player_markers[ 0 ] ) {
            return 0.5;
        }

        if ( winner == player_markers[ current_player_to_move ] ) {
            return 0.0;
        }
        else {
            return 1.0;
        }
    }

    int player_to_move;

    private:
    template<typename Stream>
    void print ( Stream & out ) const noexcept {
        out << endl;
        out << " ";
        for ( int col = 0; col < NumCols - 1; ++col )
            out << col << ' ';
        out << NumCols - 1 << endl;
        for ( int row = 0; row < NumRows; ++row ) {
            out << "|";
            for ( int col = 0; col < NumCols - 1; ++col )
                out << board ( row, col ) << ' ';
            out << board ( row, NumCols - 1 ) << "|" << endl;
        }
        out << "+";
        for ( int col = 0; col < NumCols - 1; ++col )
            out << "--";
        out << "-+" << endl;
        out << player_markers[ player_to_move ] << " to move " << endl << std::hex << zobrist ( ) << std::dec << endl << endl;
    }

    ZobristHash m_zobrist_hash = m_zobrist_player_keys[ 0 ]; // Hash of the current m_board, irrespective of who played last.
    Board board;
    int last_col, last_row;

    public:
    template<typename Stream>
    [[maybe_unused]] friend Stream & operator<< ( Stream & out, const ConnectFourState & state ) {
        state.print ( out );
        return out;
    }

    static constexpr Move const no_move             = -1;
    static constexpr int max_no_moves               = NumCols;
    static constexpr char const player_markers[ 3 ] = { '.', 'X', 'O' };
    static constexpr ZobristHashKeys m_zobrist_keys = {

        0xa1a656cb9731c5d5ull, 0xc3dce6ad6465ea7aull, 0x9e2556e2bbec18d3ull, 0x900670630f4f76afull, 0xda8071005889fa3cull,
        0xd1efb50aec8b61a9ull, 0x73203d10cf4db8b8ull, 0x6ab7fd70679d877full, 0x3a56cdae74f9d816ull, 0xb3b48dc62bacaf9bull,
        0x27760b12660e6c3bull, 0xd9ac7fb482854702ull, 0xd35e698b064e4f93ull, 0x7b379503f68242bdull, 0xdad6afcb4409d282ull,
        0xf04b592c8e1183feull, 0x6dbb4f77e63f5267ull, 0x970b0ae4e9e7d347ull, 0xd19027f157c2845aull, 0x82a53746e2d25fa5ull,
        0xe2097dbb17c142f7ull, 0x5eba98d936a14c91ull, 0x963286f60ab69777ull, 0x96e9eb899e5e615bull, 0xecd8957747d0bef8ull,
        0x961b3fb52b112218ull, 0x44c776ac7af4cc2dull, 0xfa2708e399719ac4ull, 0xe34b58c2f6acac45ull, 0x7f6d2cb0416a63caull,
        0x287ecf88477a3e7dull, 0xe57d268150b95703ull, 0xf9cc76357617493cull, 0xe956f77acaa2f112ull, 0x9a9441286a0a70e7ull,
        0x5b5a62ba1d8dfd33ull, 0xb3d1b947205bf8f4ull, 0x4aabdee7fb6aa20bull, 0xa810d257d77576afull, 0x6a1789922b7af41aull,
        0x315833a0f0b5ceebull, 0x481a32e97fbd47d8ull, 0x11e80a41d2022fdcull, 0xfab59400ba6c780cull, 0xfce9f47e1dc3037dull,
        0xf5f404421f6c78b2ull, 0x274ef7151bd8503eull, 0x1d5268cdadd43ad3ull, 0x59ed9dc04b81a0c1ull, 0x3c10ea92d1a6d79dull,
        0x595d9292d07ee51dull, 0x1a62a32bb174ee71ull, 0x417fd9b9b0bc7a47ull, 0x3e266eca431347d6ull, 0x74a093aeceb1fd60ull,
        0x7720a5e78ae8d571ull, 0x9645ae72f6f57362ull, 0xcc7279ab05731ef7ull, 0xf5a0574bc2385c6full, 0xb254ccf017ebc43bull,
        0x34184cd5945aff3eull, 0x4c5ede78a68fd1a5ull, 0x49adf513d838ce5dull, 0x44940842e2c75c16ull, 0x7aacd877d0831e19ull,
        0x9d8d5e4f7c511acdull, 0xac2f78583e0e9692ull, 0x03e2da677110440cull, 0x07d2a6b527f4ef05ull, 0x91a680f12222cf16ull,
        0x08617f45641626d0ull, 0xb2df85147e2a11cbull, 0x6bf333747f7f10a4ull, 0xc6f2a33e3a94b2c1ull, 0xf5358b1cb75e528full,
        0x904af33725c150b5ull, 0xd75d6d3f202f964bull, 0x8d58eeece3979331ull, 0xb58f905351a0d8f1ull, 0x38ad67581ffcbdfbull,
        0xcd5f48e9ac464398ull, 0xfcc2df3237564c0cull, 0x1ea8202ddf77efdeull, 0x000617fafba044adull
    };
    static constexpr ZobristHash m_zobrist_player_keys[ 3 ] = {

        0x41fec34015a1bef2ull, 0x8b80677c9c144514ull, 0xf6242292160d5bb7ull
    };

    // Spare hash keys...
    // 0xe028283c7b3c8bc3ull, 0x0fce58188743146dull, 0x5c0d56eb69eac805ull
};
