import datetime
from typing import Tuple, Iterable, Sequence

from dateutil.tz import tzutc

from source.experiments.semiotic_modelling.data_generators import SingularExchangeRateGeneratorFactory, SingularTrigonometryGeneratorFactory
from source.experiments.semiotic_modelling.methods import MovingAverage, Regression, RationalSemioticModel
from source.experiments.semiotic_modelling.modelling import EXAMPLE
from source.experiments.semiotic_modelling.visualization import ComparativeEvaluation, QualitativeEvaluation
from source.tools.timer import Timer


def multiple_sequences():
    pass


def single_sequence():
    # instantiate data source
    factory = SingularTrigonometryGeneratorFactory(100000)
    symbols = "EOS", "SNT", "QTUM", "BNT"                                                       # type: Tuple[str, ...]
    #factory = SingularExchangeRateGeneratorFactory(symbols[:], symbols[:2], length=1000)
    source = factory.get_generator()                                                            # type: Iterable[Sequence[EXAMPLE]]

    # instantiate predictors
    no_examples = len(factory.output_definition)
    input_dimension = len(factory.input_definition)
    output_dimension = 1
    drag = 500

    # instantiate semiotic model separately for future reference
    def fix(_level: int) -> int:
        # sizes = [100, 50, 20, 10, 1, 0]
        sizes = [10, 5, 1, 0]
        # sizes = [1, 0]
        if _level < len(sizes):
            return sizes[_level]
        return -1
    alpha, sigma, trace_length = 0, .8, 1
    semiotic_model = RationalSemioticModel(input_dimension, output_dimension, no_examples, alpha, sigma, drag, trace_length, fix_level_size_at=fix)

    predictors = [
        MovingAverage(output_dimension, no_examples, drag),
        Regression(input_dimension, output_dimension, no_examples, drag),
        semiotic_model
    ]

    # instantiate comparison visualization
    comparison = ComparativeEvaluation([each_predictor.__class__.__name__ for each_predictor in predictors])
    analysis = QualitativeEvaluation(no_examples)

    iterations = 0
    # iterate over sequence
    for time_step, examples in source:
        input_values, target_value = tuple(_x for _x in zip(*examples))
        target_values = target_value,
        all_outputs = []
        for each_predictor in predictors:
            # test
            output_values = each_predictor.predict(input_values)
            all_outputs.append(output_values[0])

            # train
            each_predictor.fit(input_values, target_values)

        # log data
        comparison.log(time_step, all_outputs, target_value)
        analysis.log(time_step, input_values, target_values, semiotic_model)
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
