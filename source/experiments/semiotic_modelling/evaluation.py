import datetime
from typing import Tuple, Iterable, Sequence

from dateutil.tz import tzutc

from source.experiments.semiotic_modelling.sequence_generation import MultipleExchangeRateSequence, TrigonometricSequence
from source.experiments.semiotic_modelling.methods import MovingAverage, Regression, RationalSemioticModel
from source.experiments.semiotic_modelling.modelling import EXAMPLE
from source.experiments.semiotic_modelling.visualization import ComparativeEvaluation, QualitativeEvaluation
from source.tools.timer import Timer


def multiple_sequences():
    pass


def single_sequence():
    # instantiate data source
    factory = TrigonometricSequence(1000)
    symbols = "EOS", "SNT", "QTUM", "BNT"                                                       # type: Tuple[str, ...]

    # self.start_ts = 1501113780
    # self.end_ts = 1532508240

    #factory = SingularExchangeRateGeneratorFactory(symbols[:], symbols[:2], length=1000)
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
        semiotic_model
    ]

    # instantiate comparison visualization
    comparison = ComparativeEvaluation([each_predictor.__class__.__name__ for each_predictor in predictors])
    analysis = QualitativeEvaluation(no_parallel_examples)

    iterations = 0
    # iterate over sequence
    for time_step, examples in sequence:
        each_input, each_target = examples
        input_value = each_input,
        target_value = each_target,

        all_outputs = []
        for each_predictor in predictors:
            # test
            output_values = each_predictor.predict(input_value)
            all_outputs.append(output_values[0])

            # train
            each_predictor.fit(input_value, target_value)

        # log data
        comparison.log(time_step, all_outputs, each_target)
        analysis.log(time_step, input_value, target_value, semiotic_model)
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
    single_sequence()


if __name__ == "__main__":
    main()
