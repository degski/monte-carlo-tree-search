// Petter Strandmark 2013
// petter.strandmark@gmail.com

#include <iostream>
using namespace std;

#include <mcts.h>

#include "connect_four.h"

void main_program ( ) {
    using namespace std;

    using State = ConnectFourState<6, 7>;

    bool human_player = true;

    Mcts::ComputeOptions player1_options, player2_options;
    player1_options.max_iterations = 100'000;
    player1_options.verbose        = true;
    player2_options.max_iterations = 10'000;
    player2_options.verbose        = true;

    State state;
    while ( state.has_moves ( ) ) {
        cout << endl << "State: " << state << endl;

        State::Move move = State::no_move;
        if ( state.player_to_move == 1 ) {
            move = Mcts::compute_move ( state, player1_options );
            state.do_move ( move );
        }
        else {
            if ( human_player ) {
                while ( true ) {
                    cout << "Input your move: ";
                    move = State::no_move;
                    cin >> move;
                    try {
                        state.do_move ( move );
                        break;
                    }
                    catch ( std::exception & ) {
                        cout << "Invalid move." << endl;
                    }
                }
            }
            else {
                move = Mcts::compute_move ( state, player2_options );
                state.do_move ( move );
            }
        }
    }

    cout << endl << "Final state: " << state << endl;

    if ( state.get_result ( 2 ) == 1.0 ) {
        cout << "Player 1 wins!" << endl;
    }
    else if ( state.get_result ( 1 ) == 1.0 ) {
        cout << "Player 2 wins!" << endl;
    }
    else {
        cout << "Nobody wins!" << endl;
    }
}

int main ( ) {
    try {
        main_program ( );
    }
    catch ( std::runtime_error & error ) {
        std::cerr << "ERROR: " << error.what ( ) << std::endl;
        return 1;
    }
}
