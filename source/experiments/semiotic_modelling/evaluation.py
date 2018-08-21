import datetime
from typing import Tuple, Iterable, Generator, Iterator

from dateutil.tz import tzutc
from matplotlib import pyplot

from source.experiments.semiotic_modelling.sequence_generation import ExchangeRateSequence, TrigonometricSequence
from source.experiments.semiotic_modelling.methods import MovingAverage, Regression, RationalSemioticModel
from source.experiments.semiotic_modelling.modelling import EXAMPLE, TIME
from source.experiments.semiotic_modelling.visualization import ComparativeEvaluation, QualitativeEvaluationSingleSequence, \
    QualitativeEvaluationMultiSequence
from source.tools.timer import Timer


def fix(_level: int) -> int:
    # sizes = [100, 50, 20, 10, 1, 0]
    # sizes = [10, 5, 1, 0]
    sizes = [3, 1, 0]
    if _level < len(sizes):
        return sizes[_level]
    return -1


def join_sequences(sequences: Iterable[Iterator[Tuple[TIME, EXAMPLE]]]) -> Generator[Tuple[TIME, Tuple[EXAMPLE, ...]], None, None]:
    while True:
        time_set = set()
        examples = []
        for each_sequence in sequences:
            each_time, each_example = next(each_sequence)
            time_set.add(each_time)
            examples.append(each_example)
        assert len(time_set) == 1
        this_time, = time_set
        yield this_time, tuple(examples)


def multiple_sequences():
    symbols = "EOS", "SNT", "QTUM", "BNT"                                                       # type: Tuple[str, ...]
    # factories = tuple(ExchangeRateSequence((_x, ), (_x, ), start_timestamp=1501113780, end_timestamp=1501250240) for _x in symbols)
    factories = tuple(ExchangeRateSequence((_x, ), (_x, ), start_timestamp=1501113780, end_timestamp=1529712000) for _x in symbols)
    sequences = join_sequences([_x.get_generator() for _x in factories])

    no_parallel_examples = len(symbols)
    input_dimension, output_dimension = 1, 1
    drag = 500

    alpha, sigma, trace_length = 0, .8, 1
    semiotic_model = RationalSemioticModel(
        input_dimension, output_dimension,
        no_parallel_examples,
        alpha, sigma, drag, trace_length, fix_level_size_at=fix)

    analysis = QualitativeEvaluationMultiSequence(output_dimension, no_parallel_examples)

    iterations = 0
    for time_step, examples in sequences:
        input_values, target_values = zip(*examples)

        # test
        semiotic_output = semiotic_model.predict(input_values)

        # train
        semiotic_model.fit(input_values, target_values)

        # log data
        certainty = semiotic_model.get_certainty(input_values, target_values)
        structure = semiotic_model.get_structure()
        states = semiotic_model.get_states()
        analysis.log_multisequence(time_step, target_values, semiotic_output, certainty)

        if Timer.time_passed(2000):
            msg = "At iteration {:d} time {:s} structure {:s}"
            print(msg.format(iterations, str(datetime.datetime.fromtimestamp(time_step, tz=tzutc())), str(semiotic_model.get_structure())))

        iterations += 1

    analysis.plot()


def single_sequence():
    # instantiate data source
    #"""
    factory = TrigonometricSequence(50000)
    """
    symbols = "EOS", "SNT", "QTUM", "BNT"                                                       # type: Tuple[str, ...]
    # factory = ExchangeRateSequence(symbols[:1], symbols[:1], start_timestamp=1501113780, end_timestamp=1501250240)
    factory = ExchangeRateSequence(symbols[:1], symbols[:1], start_timestamp=1501113780, end_timestamp=1532508240)
    # """
    sequence = factory.get_generator()                                     # type: Iterable[Tuple[TIME, EXAMPLE]]

    # instantiate predictors
    no_parallel_examples = 1
    input_dimension, output_dimension = factory.get_dimensions()
    drag = 200

    # instantiate semiotic model separately for future reference
    alpha, sigma, trace_length = 0, .8, 1
    semiotic_model = RationalSemioticModel(
        input_dimension, output_dimension,
        no_parallel_examples,
        alpha, sigma, drag, trace_length, fix_level_size_at=fix, differentiate=False)
    predictors = [
        MovingAverage(output_dimension, no_parallel_examples, drag),
        Regression(input_dimension, output_dimension, no_parallel_examples, drag),
    ]

    # instantiate visualization
    method_names = [each_predictor.__class__.__name__ for each_predictor in predictors] + ["semiotic model"]
    comparison = ComparativeEvaluation(output_dimension, method_names)
    analysis = QualitativeEvaluationSingleSequence(output_dimension, ["semiotic model"])

    # iterate over sequence
    iterations = 0
    for time_step, each_example in sequence:
        each_input, each_target = each_example
        input_values = each_input,
        target_values = each_target,

        # test
        all_outputs = []
        for each_predictor in predictors:
            output_values = each_predictor.predict(input_values)
            all_outputs.append(output_values[0])
        semiotic_output = semiotic_model.predict(input_values)
        all_outputs.append(semiotic_output[0])

        # train
        for each_predictor in predictors:
            each_predictor.fit(input_values, target_values)
        semiotic_model.fit(input_values, target_values)

        # log data
        comparison.log_predictors(time_step, all_outputs, each_target)
        certainty = semiotic_model.get_certainty(input_values, target_values)
        structure = semiotic_model.get_structure()
        states = semiotic_model.get_states()
        analysis.log_semiotic_model(time_step, input_values[0], target_values[0], semiotic_output[0], certainty[0], structure, states[0])

        if Timer.time_passed(2000):
            msg = "At iteration {:d} time {:s} structure {:s}"
            print(msg.format(iterations, str(datetime.datetime.fromtimestamp(time_step, tz=tzutc())), str(semiotic_model.get_structure())))

        iterations += 1

    # save results
    #ce.save("")
    # save each predictor
    for each_predictor in predictors:
        #each_predictor.save("")
        pass

    # visualize results
    comparison.plot()
    # visualize semiotic model
    analysis.plot(plot_segments=True)


def main():
    # TODO: replace fix model at with ability to pregenerate model
    # investigate content generation by parallel examples (problematic)
    single_sequence()
    # multiple_sequences()


if __name__ == "__main__":
    main()
