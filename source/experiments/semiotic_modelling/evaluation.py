import datetime
from typing import Tuple, Iterable, Sequence

from dateutil.tz import tzutc

from source.experiments.semiotic_modelling.sequence_generation import ExchangeRateSequence, TrigonometricSequence
from source.experiments.semiotic_modelling.methods import MovingAverage, Regression, RationalSemioticModel
from source.experiments.semiotic_modelling.modelling import EXAMPLE
from source.experiments.semiotic_modelling.visualization import ComparativeEvaluation, QualitativeEvaluationSingleSequence
from source.tools.timer import Timer


def multiple_sequences():
    pass


def single_sequence():
    # instantiate data source
    """
    factory = TrigonometricSequence(100000)
    """
    symbols = "EOS", "SNT", "QTUM", "BNT"                                                       # type: Tuple[str, ...]
    factory = ExchangeRateSequence(symbols[:], symbols[:2], start_timestamp=1501113780, end_timestamp=1502508240)
    # factory = ExchangeRateSequence(symbols[:], symbols[:2], start_timestamp=1501113780, end_timestamp=1532508240)
    # """
    sequence = factory.get_generator()                                                            # type: Iterable[Sequence[EXAMPLE]]

    # instantiate predictors
    no_parallel_examples = 1
    input_dimension, output_dimension = factory.get_dimensions()
    drag = 500

    # instantiate semiotic model separately for future reference
    def fix(_level: int) -> int:
        # sizes = [100, 50, 20, 10, 1, 0]
        sizes = [10, 5, 3, 1, 0]
        # sizes = [1, 0]
        if _level < len(sizes):
            return sizes[_level]
        return -1
    alpha, sigma, trace_length = 0, .8, 1
    semiotic_model = RationalSemioticModel(
        input_dimension, output_dimension,
        no_parallel_examples,
        alpha, sigma, drag, trace_length, fix_level_size_at=fix)

    predictors = [
        MovingAverage(output_dimension, no_parallel_examples, drag),
        Regression(input_dimension, output_dimension, no_parallel_examples, drag),
    ]

    # instantiate comparison visualization
    method_names = [each_predictor.__class__.__name__ for each_predictor in predictors] + ["semiotic model"]
    comparison = ComparativeEvaluation(output_dimension, method_names)
    analysis = QualitativeEvaluationSingleSequence(output_dimension, ["semiotic model"])

    iterations = 0
    # iterate over sequence
    for time_step, each_example in sequence:
        each_input, each_target = each_example
        input_values = each_input,
        target_values = each_target,

        all_outputs = []
        for each_predictor in predictors:
            # test
            output_values = each_predictor.predict(input_values)
            all_outputs.append(output_values[0])

            # train
            each_predictor.fit(input_values, target_values)

        semiotic_output = semiotic_model.predict(input_values)
        all_outputs.append(semiotic_output[0])
        semiotic_model.fit(input_values, target_values)

        # log data
        comparison.log_predictors(time_step, all_outputs, each_target)
        certainty = semiotic_model.get_certainty(input_values, target_values)
        structure = semiotic_model.get_structure()
        states = semiotic_model.get_states()
        analysis.log_semiotic_model(time_step, target_values[0], semiotic_output[0], certainty[0], structure, states[0])

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
    analysis.plot(plot_segments=False)


def main():
    single_sequence()


if __name__ == "__main__":
    main()
