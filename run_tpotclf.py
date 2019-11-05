from tpot import TPOTClassifier
import helpers
from functools import partial
import numpy as np
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    args = helpers.args()

    method = TPOTClassifier(
        max_time_mins=args.runtime,
        random_state=0,
        scoring="f1_weighted",
        n_jobs=args.cores,
        verbosity=2,
    )

    train_x, train_y, test_x, test_y = helpers.train_test_split(args.dataset, k=args.fold, seed=args.seed)

    testing_frequency = 1
    total_test_runs = 3

    tpot = TPOTClassifier(
        max_time_mins=testing_frequency,
        warm_start=True
    )

    scores = []

    for _ in range(total_test_runs):
        tpot.fit(train_x, train_y)
        score = tpot.score(test_x, test_y)
        scores.append(score)

    print(scores)

    #fn = partial(helpers.run_and_time_estimator, method)
    #scores = helpers.main(args, fn)
    #print("TPOT", scores)
