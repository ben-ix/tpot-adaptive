from tpot import TPOTClassifier
import helpers
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    args = helpers.args()

    train_x, train_y, test_x, test_y = helpers.train_test_split(args.dataset, k=args.fold, seed=args.seed)

    max_train_time = args.runtime
    total_test_runs = 10

    testing_frequency = max_train_time / total_test_runs

    tpot = TPOTClassifier(
        max_time_mins=testing_frequency,
        warm_start=True,
        scoring="f1_weighted",
        n_jobs=args.cores,
        random_state=0,
    )

    test_scores = []
    fitness_progression = []
    complexity = []
    pareto_front_size = []

    for _ in range(total_test_runs):
        tpot.fit(train_x, train_y)
        test_score = tpot.score(test_x, test_y)
        fitness = tpot._optimized_pipeline_score
        test_scores.append(test_score)
        fitness_progression.append(fitness)
        complexity.append(tpot._complexity)
        pareto_front_size.append(len(tpot._pareto_front))

    print("Mutation rate path:", tpot._param_dict["mutpb_rates"])
    print("Population size path:", tpot._logbook.select("nevals"))
    print("Frontier size path:", pareto_front_size)
    print("Training curve:", fitness_progression)
    print("Testing curve:", test_scores)
    print("Complexity curve:", complexity)


