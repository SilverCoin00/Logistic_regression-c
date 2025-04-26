Code structure:
     P_&_N -> Sklearn   -> |
         |     |           |    ->   Core.h   ->   Execute
         V     V           |          ^  ^
         Dataset        -> |          |  |
                                      V  V
                               Logistic_regression
                                     Weights
