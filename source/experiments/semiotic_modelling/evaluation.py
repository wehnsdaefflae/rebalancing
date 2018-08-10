from typing import Tuple, Iterable, Sequence

from source.experiments.semiotic_modelling.data_generators import ExchangeRateGeneratorFactory, TrigonometryGeneratorFactory
from source.experiments.semiotic_modelling.methods import MovingAverage, Regression, RationalSemioticModel
from source.experiments.semiotic_modelling.modelling import EXAMPLE
from source.experiments.semiotic_modelling.visualization import ComparativeEvaluation, QualitativeEvaluation
from source.tools.timer import Timer


def evaluate():
    # instantiate data source
    factory = TrigonometryGeneratorFactory(50000)
    symbols = "EOS", "SNT", "QTUM" # , "BNT"                                                       # type: Tuple[str, ...]
    # factory = ExchangeRateGeneratorFactory(symbols[:1], symbols[:1], length=279625)
    source = factory.get_generator()                                                            # type: Iterable[Sequence[EXAMPLE]]

    # instantiate predictors
    no_examples = len(factory.output_definition)
    input_dimension = len(factory.input_definition)
    drag = 500

    # instantiate semiotic model separately for future reference
    def fix(_level: int) -> int:
        # sizes = [100, 50, 20, 10, 1, 0]
        sizes = [10, 5, 1, 0]
        # sizes = [1, 0]
        if _level < len(sizes):
            return sizes[_level]
        return -1
    alpha, sigma, trace_length = 0, .5, 1
    semiotic_model = RationalSemioticModel(no_examples, drag, input_dimension, alpha, sigma, trace_length, fix_level_size_at=fix)

    predictors = [
        MovingAverage(no_examples, drag),
        Regression(no_examples, drag, input_dimension),
        semiotic_model
    ]

    # instantiate comparison visualization
    comparison = ComparativeEvaluation([each_predictor.__class__.__name__ for each_predictor in predictors])
    analysis = QualitativeEvaluation(input_dimension)

    iterations = 0
    # iterate over sequence
    for time_step, examples in source:
        input_values, target_values = tuple(_x for _x in zip(*examples))
        all_outputs = []
        for each_predictor in predictors:
            # test
            output_values = each_predictor.predict(input_values)
            all_outputs.append(output_values)

            # train
            each_predictor.fit(input_values, target_values)

        # log data
        comparison.log(time_step, all_outputs, target_values)
        analysis.log(time_step, input_values, target_values, semiotic_model)
        if Timer.time_passed(2000):
            print("At iteration {:d} time stamp {:s} structure {:s}".format(iterations, str(time_step), str(semiotic_model.get_structure())))

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
    evaluate()


if __name__ == "__main__":
    main()
