import numpy

from scipy.io import wavfile

from deprecated.experiments.dtw.my_dtw import get_table, get_path, get_fitted_sequences, plot_series

fs_o, input_sequence = wavfile.read("C:/Users/Mark/Daten/Audio/test.wav")
fs_t, target_sequence = wavfile.read("C:/Users/Mark/Daten/Audio/baum.wav")

"""
pyplot.clf()
pyplot.close()
pyplot.plot(input_sequence)
pyplot.show()
"""
a, b = [_x for _x in input_sequence], [_x for _x in target_sequence]
t = get_table(a, b,
              normalized=True, derivative=False, overlap=False, diag_factor=1.,
              distance=lambda _x, _y: (_x - _y) ** 2, w=0)
p = get_path(t, overlap=False)

fit_a, fit_b = get_fitted_sequences(a, b, p)

wavfile.write("C:/Users/Mark/Daten/Audio/test_fit.wav", fs_o, numpy.array(fit_a))
wavfile.write("C:/Users/Mark/Daten/Audio/baum_fit.wav", fs_t, numpy.array(fit_b))


plot_series(a, b, p, a_label="test", b_label="baum", file_path=None)


print()