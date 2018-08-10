from numpy.linalg import LinAlgError
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from sklearn.metrics import mean_squared_error

from source.experiments.semiotic_modelling.data_generators import ExchangeRateGeneratorFactory


def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


symbols = "EOS", "SNT", "QTUM", "BNT"                                                       # type: Tuple[str, ...]
factory = ExchangeRateGeneratorFactory(symbols[:1], symbols[:1])
g = factory.get_generator()
h = 100

X = [each_example[0][0][0] for _, each_example in g][:1000]

predictions = []
targets = []
model = ARIMA(X[:900], order=(5, 1, 0))
try:
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=100)
    predictions = output[0]
    targets = X[900:]

except LinAlgError as e:
    pass

error = mean_squared_error(targets, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(X[:900] + targets, label="target")
pyplot.plot(range(900, 1000), predictions, label="output")
pyplot.legend()
pyplot.show()
