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

    fn = partial(helpers.run_and_time_estimator, method)
    scores = helpers.main(args, fn)
    print("TPOT", scores)
