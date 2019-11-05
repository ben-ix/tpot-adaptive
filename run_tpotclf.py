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

    data_x, data_y = helpers.read_data(args.dataset)
    testing_frequency = 1
    total_test_runs = 3

    tpot = TPOTClassifier(
        max_time_mins=testing_frequency,
        warm_start=True
    )

    scores = []

    for _ in range(total_test_runs):
        tpot.fit(data_x, data_y)
        score = tpot.score(data_x, data_y)
        scores.append(score)

    print(scores)

    #fn = partial(helpers.run_and_time_estimator, method)
    #scores = helpers.main(args, fn)
    #print("TPOT", scores)
