// Petter Strandmark 2013
// petter.strandmark@gmail.com

#include <algorithm>
#include <iostream>

#include <mcts.h>

#include "multi_array.hpp"

template<int NumRows = 6, int NumCols = 7>
class ConnectFourState {
    public:
    using Move                = int;
    using Moves               = sax::compact_vector<Move, std::int64_t, NumCols, NumCols>;
    static const Move no_move = -1;
    // using Board               = ma::MatrixChar<char, 6, 7>;

    using ZobristHash     = std::uint64_t;
    using ZobristHashKeys = ma::Cube<ZobristHash, 2, NumRows, NumCols, 1>; // 2 players, 1 based...

    static const char player_markers[ 3 ];

    ConnectFourState ( ) : player_to_move ( 1 ), last_col ( -1 ), last_row ( -1 ) {
        board.resize ( NumRows, std::vector<char> ( NumCols, player_markers[ 0 ] ) );
    }

    void do_move ( Move move ) {
        attest ( 0 <= move && move < NumCols );
        attest ( board[ 0 ][ move ] == player_markers[ 0 ] );

        int row = NumRows - 1;
        while ( board[ row ][ move ] != player_markers[ 0 ] )
            row--;
        board[ row ][ move ] = player_markers[ player_to_move ];
        last_col             = move;
        last_row             = row;

        player_to_move = 3 - player_to_move;
    }

    template<typename RandomEngine>
    void do_random_move ( RandomEngine * engine ) {
        dattest ( has_moves ( ) );
        sax::uniform_int_distribution<Move> moves ( 0, NumCols - 1 );

        while ( true ) {
            auto move = moves ( *engine );
            if ( board[ 0 ][ move ] == player_markers[ 0 ] ) {
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
            if ( board[ 0 ][ col ] == player_markers[ 0 ] )
                return true;
        return false;
    }

    [[nodiscard]] Moves get_moves ( ) const {
        Moves moves;
        if ( get_winner ( ) != player_markers[ 0 ] )
            return moves;
        // moves.reserve ( NumCols ); no need to reserve, first allocation will be max.
        for ( int col = 0; col < NumCols; ++col )
            if ( board[ 0 ][ col ] == player_markers[ 0 ] )
                moves.push_back ( col );
        return moves;
    }

    [[nodiscard]] char get_winner ( ) const noexcept {
        if ( last_col < 0 )
            return player_markers[ 0 ];
        // We only need to check around the last piece played.
        auto piece = board[ last_row ][ last_col ];
        // X X X X
        int left = 0, right = 0;
        for ( int col = last_col - 1; col >= 0 && board[ last_row ][ col ] == piece; --col )
            left++;
        for ( int col = last_col + 1; col < NumCols && board[ last_row ][ col ] == piece; ++col )
            right++;
        if ( left + 1 + right >= 4 )
            return piece;
        // X
        // X
        // X
        // X
        int up = 0, down = 0;
        for ( int row = last_row - 1; row >= 0 && board[ row ][ last_col ] == piece; --row )
            up++;
        for ( int row = last_row + 1; row < NumRows && board[ row ][ last_col ] == piece; ++row )
            down++;
        if ( up + 1 + down >= 4 )
            return piece;
        // X
        //  X
        //   X
        //    X
        up   = 0;
        down = 0;
        for ( int row = last_row - 1, col = last_col - 1; row >= 0 && col >= 0 && board[ row ][ col ] == piece; --row, --col )
            up++;
        for ( int row = last_row + 1, col = last_col + 1; row < NumRows && col < NumCols && board[ row ][ col ] == piece;
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
        for ( int row = last_row + 1, col = last_col - 1; row < NumRows && col >= 0 && board[ row ][ col ] == piece; ++row, --col )
            up++;
        for ( int row = last_row - 1, col = last_col + 1; row >= 0 && col < NumCols && board[ row ][ col ] == piece; --row, ++col )
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

    void print ( ostream & out ) const noexcept {
        out << endl;
        out << " ";
        for ( int col = 0; col < NumCols - 1; ++col ) {
            out << col << ' ';
        }
        out << NumCols - 1 << endl;
        for ( int row = 0; row < NumRows; ++row ) {
            out << "|";
            for ( int col = 0; col < NumCols - 1; ++col ) {
                out << board[ row ][ col ] << ' ';
            }
            out << board[ row ][ NumCols - 1 ] << "|" << endl;
        }
        out << "+";
        for ( int col = 0; col < NumCols - 1; ++col ) {
            out << "--";
        }
        out << "-+" << endl;
        out << player_markers[ player_to_move ] << " to move " << endl << endl;
    }

    int player_to_move;

    private:
    vector<vector<char>> board;
    int last_col, last_row;
};

template<int NumRow, int NumCols>
ostream & operator<< ( ostream & out, const ConnectFourState<NumRow, NumCols> & state ) {
    state.print ( out );
    return out;
}

template<int NumRow, int NumCols>
const char ConnectFourState<NumRow, NumCols>::player_markers[ 3 ] = { '.', 'X', 'O' };
