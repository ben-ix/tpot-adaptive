# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import random
import itertools
from deap import tools, gp
from inspect import isclass
from .operator_utils import set_sample_weight
from sklearn.utils import indexable
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._split import check_cv
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone, is_classifier
from collections import defaultdict
import warnings
from stopit import threading_timeoutable, TimeoutException
from statistics import mean


def pick_two_individuals_eligible_for_crossover(population):
    """Pick two individuals from the population which can do crossover, that is, they share a primitive.

    Parameters
    ----------
    population: array of individuals

    Returns
    ----------
    tuple: (individual, individual)
        Two individuals which are not the same, but share at least one primitive.
        Alternatively, if no such pair exists in the population, (None, None) is returned instead.
    """
    primitives_by_ind = [set([node.name for node in ind if isinstance(node, gp.Primitive)])
                         for ind in population]

    pop_as_str = [str(ind) for ind in population]

    eligible_pairs = [(i, i+1+j) for i, ind1_prims in enumerate(primitives_by_ind)
                                 for j, ind2_prims in enumerate(primitives_by_ind[i+1:])
                                 if not ind1_prims.isdisjoint(ind2_prims) and
                                    pop_as_str[i] != pop_as_str[i+1+j]]

    # Pairs are eligible in both orders, this ensures that both orders are considered
    eligible_pairs += [(j, i) for (i, j) in eligible_pairs]

    random.shuffle(eligible_pairs)

    for idx1, idx2 in eligible_pairs:
        yield population[idx1], population[idx2]


def mutate_random_individual(population, toolbox):
    """Picks a random individual from the population, and performs mutation on a copy of it.

    Parameters
    ----------
    population: array of individuals

    Returns
    ----------
    individual: individual
        An individual which is a mutated copy of one of the individuals in population,
        the returned individual does not have fitness.values
    """
    indices = list(range(len(population)))
    random.shuffle(indices)

    for idx in indices:
        ind = population[idx]
        ind, = toolbox.mutate(ind)

        if ind:
            del ind.fitness.values
            return ind

    # Couldnt mutate. Return a copy
    print("Couldnt mutate")
    return toolbox.clone(population[0])


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    offspring = []

    for _ in range(lambda_):
        op_choice = np.random.random()
        if op_choice < cxpb:  # Apply crossover

            generated_offspring = False

            for ind1, ind2 in pick_two_individuals_eligible_for_crossover(population):
                ind1, _ = toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                generated_offspring = True
                break

            if not generated_offspring:
                ind1 = mutate_random_individual(population, toolbox)

            offspring.append(ind1)

        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = mutate_random_individual(population, toolbox)
            offspring.append(ind)
        else:  # Apply reproduction
            idx = np.random.randint(0, len(population))
            offspring.append(toolbox.clone(population[idx]))

    return offspring

def initialize_stats_dict(individual):
    '''
    Initializes the stats dict for individual
    The statistics initialized are:
        'generation': generation in which the individual was evaluated. Initialized as: 0
        'mutation_count': number of mutation operations applied to the individual and its predecessor cumulatively. Initialized as: 0
        'crossover_count': number of crossover operations applied to the individual and its predecessor cumulatively. Initialized as: 0
        'predecessor': string representation of the individual. Initialized as: ('ROOT',)

    Parameters
    ----------
    individual: deap individual

    Returns
    -------
    object
    '''
    individual.statistics['generation'] = 0
    individual.statistics['mutation_count'] = 0
    individual.statistics['crossover_count'] = 0
    individual.statistics['predecessor'] = 'ROOT',


def adaptiveEa(population, logbook, toolbox, param_dict, stats=None, verbose=0,
               per_generation_function=None):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param verbose: Whether or not to log the statistics.
    :param per_generation_function: if supplied, call this function before each generation
                            used by tpot to save best pipeline before each new generation
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """

    # Initialize statistics dict for the individuals in the population, to keep track of mutation/crossover operations and predecessor relations
    for ind in population:
        initialize_stats_dict(ind)

    population[:] = toolbox.evaluate(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), **record)

    print(logbook.stream)

    best_fitness_last_gen = param_dict['best_individual_fitness']

    # Begin the generational process
    for gen in range(1, 999999999):  # TODO: Update this condition
        # after each population save a periodic pipeline
        if per_generation_function is not None:
            per_generation_function(gen)

        previous_sizes = param_dict["previous_sizes"]

        current_pop_size = previous_sizes[-1]

        # We're always adding the value 2 before in the fibonacci sequence, as the final element is the current_pop_size
        offspring_size = previous_sizes[-2]

        # Use the most recent rate
        mutpb = param_dict["mutpb_rates"][-1]

        # Vary the population
        offspring = varOr(population, toolbox, offspring_size, cxpb=1-mutpb, mutpb=mutpb)

        # Update generation statistic for all individuals which have invalid 'generation' stats
        # This hold for individuals that have been altered in the varOr function
        for ind in population:
            if ind.statistics['generation'] == 'INVALID':
                ind.statistics['generation'] = gen

        offspring = toolbox.evaluate(offspring)

        # Compute improvement over previous gen
        best_fitness_this_gen = param_dict['best_individual_fitness']
        comparison_fitnesses = list(zip(best_fitness_this_gen, best_fitness_last_gen))

        # The number of objectives we improved
        improvement = sum([1 * (x > y) for (x, y) in comparison_fitnesses])

        if improvement == 2:        # If we improved on both objectives
            # Shrink population if we can
            if len(previous_sizes) > 2:
                previous_sizes.pop()  # Remove current size
                next_population_size = previous_sizes[-1]  # Retreat to previous pop size. Be careful not to go to zero
            else:
                # If we cant, then the minimum size is 1
                next_population_size = 1
        elif improvement == 1:          # If we improved on only one objective
            # Better, so keep population size
            next_population_size = current_pop_size
        else:                   # No progress. So increase population size, as done in fibonacci sequence
            next_population_size = previous_sizes[-2] + previous_sizes[-1]
            previous_sizes.append(next_population_size)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, next_population_size)

        # If we didnt improve, adjust mutation rate
        if not improvement:
            # Adjust crossover and mutation rates
            fitness_std = logbook.chapters["fitness"].select("std")[-1]

            # The maximum obeserved standard deviation
            max_std = max(logbook.chapters["fitness"].select("std"))

            # Fitness is proportional to the standard deviation. A low standard deviation means similar individuals,
            # so we should have a high mutation rate.
            mutpb = 1 - (fitness_std / max_std)

        param_dict["mutpb_rates"].append(mutpb)

        # after each population save a periodic pipeline
        if per_generation_function is not None:
            per_generation_function(gen)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)

        print(logbook.stream)

        best_fitness_last_gen = best_fitness_this_gen

    return population, param_dict


def cxOnePoint(ind1, ind2, clone):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    common_types = []
    for idx, node in enumerate(ind2[1:], 1):
        if node.ret in types1 and node.ret not in types2:
            common_types.append(node.ret)
        types2[node.ret].append(idx)

    random.shuffle(common_types)

    for type_ in common_types:

        random.shuffle(types1[type_])
        random.shuffle(types2[type_])

        for index1 in types1[type_]:
            for index2 in types2[type_]:
                slice1 = ind1.searchSubtree(index1)
                slice2 = ind2.searchSubtree(index2)

                ind1_copy, ind2_copy = clone(ind1), clone(ind2)
                ind1_copy[slice1], ind2_copy[slice2] = ind2_copy[slice2], ind1_copy[slice1]

                yield ind1_copy, ind1_copy

def mutInsert(individual, pset, clone):
    """

    Modification of deap.gp.mutInsert that uses a generator
    to produce offspring.

    :param individual: The normal or typed tree to be mutated.
    :returns: A generator of tuples of one tree.
    """
    indices = list(range(len(individual)))
    random.shuffle(indices)

    for index in indices:
        node = individual[index]
        slice_ = individual.searchSubtree(index)

        # As we want to keep the current node as children of the new one,
        # it must accept the return value of the current node
        primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

        random.shuffle(primitives)

        for new_node in primitives:
            positions = [i for i, a in enumerate(new_node.args) if a == node.ret]
            random.shuffle(positions)

            for position in positions:
                new_subtree_possibilities = [[None]] * len(new_node.args)

                for i, arg_type in enumerate(new_node.args):
                    if i != position:
                        terminals = [term() if isclass(term) else term for term in pset.terminals[arg_type]]
                        random.shuffle(terminals)
                        new_subtree_possibilities[i] = terminals

                new_subtrees = itertools.product(*new_subtree_possibilities)

                for new_subtree in new_subtrees:
                    new_subtree = list(new_subtree)

                    new_subtree[position:position + 1] = individual[slice_]
                    new_subtree.insert(0, new_node)
                    individual_clone = clone(individual)
                    individual_clone[slice_] = new_subtree

                    yield individual_clone,

def mutShrink(individual, clone):
    """
    Modification of deap.gp.mutShrink which uses generators

    :param individual: The tree to be shrunk.
    :returns: A generator of tuples of one tree.
    """
    # We don't want to "shrink" the root
    if len(individual) < 3 or individual.height <= 1:
        yield individual,

    iprims = []
    for i, node in enumerate(individual[1:], 1):
        if isinstance(node, gp.Primitive) and node.ret in node.args:
            iprims.append((i, node))

    random.shuffle(iprims)

    for index, prim in iprims:
        arg_indices = [i for i, type_ in enumerate(prim.args) if type_ == prim.ret]
        random.shuffle(arg_indices)

        for arg_idx in arg_indices:
            rindex = index + 1
            for _ in range(arg_idx + 1):
                rslice = individual.searchSubtree(rindex)
                subtree = individual[rslice]
                rindex += len(subtree)

            slice_ = individual.searchSubtree(index)
            individual_copy = clone(individual)
            individual_copy[slice_] = subtree

            yield individual_copy,

# point mutation function
def mutNodeReplacement(individual, pset, clone):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive no matter if it has the same number of arguments from the :attr:`pset`
    attribute of the individual.
    Parameters
    ----------
    individual: DEAP individual
        A list of pipeline operators and model parameters that can be
        compiled by DEAP into a callable function

    Returns
    -------
    individual: DEAP individual
        Returns the individual with one of point mutation applied to it

    """

    indices = list(range(len(individual)))
    np.random.shuffle(indices)

    for index in indices:
        node = individual[index]
        slice_ = individual.searchSubtree(index)

        if node.arity == 0:  # Terminal
            terminals = list(pset.terminals[node.ret])
            random.shuffle(terminals)

            for term in terminals:
                if isclass(term):
                    term = term()

                individual_clone = clone(individual)
                individual_clone[index] = term

                yield individual_clone,
        else:   # Primitive
            # find next primitive if any
            rindex = None
            if index + 1 < len(individual):
                for i, tmpnode in enumerate(individual[index + 1:], index + 1):
                    if isinstance(tmpnode, gp.Primitive) and tmpnode.ret in tmpnode.args:
                        rindex = i
                        break

            # pset.primitives[node.ret] can get a list of the type of node
            # for example: if op.root is True then the node.ret is Output_DF object
            # based on the function _setup_pset. Then primitives is the list of classifor or regressor
            primitives = list(pset.primitives[node.ret])
            random.shuffle(primitives)

            if rindex:
                for new_node in primitives:
                    rnode = individual[rindex]
                    rslice = individual.searchSubtree(rindex)
                    # find position for passing return values to next operator
                    positions = [i for i, a in enumerate(new_node.args) if a == rnode.ret]
                    random.shuffle(positions)

                    for position in positions:
                        new_subtree_possibilities = [[None]] * len(new_node.args)

                        for i, arg_type in enumerate(new_node.args):
                            if i != position:
                                terminals = [term() if isclass(term) else term for term in pset.terminals[arg_type]]
                                random.shuffle(terminals)
                                new_subtree_possibilities[i] = terminals

                        # Note: we dont shuffle here as would involve converting tierator to list. Instead
                        # we shuffle the terminals above.
                        new_subtrees = itertools.product(*new_subtree_possibilities)

                        for new_subtree in new_subtrees:
                            new_subtree = list(new_subtree)

                            new_subtree[position:position + 1] = individual[rslice]

                            # combine with primitives
                            new_subtree.insert(0, new_node)
                            individual_clone = clone(individual)
                            individual_clone[slice_] = new_subtree

                            yield individual_clone,
            else:
                for new_node in primitives:
                    new_subtree_possibilities = [[None]] * len(new_node.args)

                    for i, arg_type in enumerate(new_node.args):
                        terminals = [term() if isclass(term) else term for term in pset.terminals[arg_type]]
                        random.shuffle(terminals)
                        new_subtree_possibilities[i] = terminals

                    new_subtrees = itertools.product(*new_subtree_possibilities)

                    for new_subtree in new_subtrees:
                        new_subtree = list(new_subtree)

                        # combine with primitives
                        new_subtree.insert(0, new_node)
                        individual_clone = clone(individual)
                        individual_clone[slice_] = new_subtree

                        yield individual_clone,

@threading_timeoutable(default="Timeout")
def _wrapped_cross_val_score(sklearn_pipeline, features, target,
                             cv, scoring_function, sample_weight=None,
                             groups=None, use_dask=False):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    sklearn_pipeline : pipeline object implementing 'fit'
        The object to use to fit the data.
    features : array-like of shape at least 2D
        The data to fit.
    target : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    cv: int or cross-validation generator
        If CV is a number, then it is the number of folds to evaluate each
        pipeline over in k-fold cross-validation during the TPOT optimization
         process. If it is an object then it is an object to be used as a
         cross-validation generator.
    scoring_function : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    sample_weight : array-like, optional
        List of sample weights to balance (or un-balanace) the dataset target as needed
    groups: array-like {n_samples, }, optional
        Group labels for the samples used while splitting the dataset into train/test set
    use_dask : bool, default False
        Whether to use dask
    """
    sample_weight_dict = set_sample_weight(sklearn_pipeline.steps, sample_weight)

    features, target, groups = indexable(features, target, groups)

    cv = check_cv(cv, target, classifier=is_classifier(sklearn_pipeline))
    cv_iter = list(cv.split(features, target, groups))
    scorer = check_scoring(sklearn_pipeline, scoring=scoring_function)

    if use_dask:
        try:
            import dask_ml.model_selection  # noqa
            import dask  # noqa
            from dask.delayed import Delayed
        except ImportError:
            msg = "'use_dask' requires the optional dask and dask-ml depedencies."
            raise ImportError(msg)

        dsk, keys, n_splits = dask_ml.model_selection._search.build_graph(
            estimator=sklearn_pipeline,
            cv=cv,
            scorer=scorer,
            candidate_params=[{}],
            X=features,
            y=target,
            groups=groups,
            fit_params=sample_weight_dict,
            refit=False,
            error_score=float('-inf'),
        )

        cv_results = Delayed(keys[0], dsk)
        scores = [cv_results['split{}_test_score'.format(i)]
                  for i in range(n_splits)]
        CV_score = dask.delayed(np.array)(scores)[:, 0]
        return dask.delayed(np.nanmean)(CV_score)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scores = [_fit_and_score(estimator=clone(sklearn_pipeline),
                                         X=features,
                                         y=target,
                                         scorer=scorer,
                                         train=train,
                                         test=test,
                                         verbose=0,
                                         parameters=None,
                                         fit_params=sample_weight_dict)
                                    for train, test in cv_iter]
                CV_score = np.array(scores)[:, 0]
                return np.nanmean(CV_score)
        except TimeoutException:
            return "Timeout"
        except Exception as e:
            return -float('inf')