import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import seaborn as sns

class GeSbS_Graphics:
    def __init__(self, dpi):
        self.dpi = dpi

    def dB(self, x, y_input = None, y_output = None, x_limit = None, y_limit = None, alpha = 0.4, linewidth = 1, color_input = '#5e2129', color_output = "#191970"):
        sns.set(style="darkgrid")

        try:
            fft_input = fft.ifft(y_input)
            maximum = np.max( fft_input*np.conj(fft_input) )
        except:

            try:
                maximum = np.max(y_output*np.conj(y_output))
            except:
                return 'Datos insuficientes o invalidos'

        try:
            output_if = len(y_output)
        except:
            output_if = 0

        try:
            input_if = len(y_input)
        except:
            input_if = 0

        fig, ax = plt.subplots(dpi=self.dpi)

        if output_if:
            self.output_dB = 10*np.log10( (y_output*np.conj(y_output)) / maximum)
            self.output_dB = np.real(self.output_dB)

            ax.plot(x, self.output_dB, label="Output", color = color_output)
            ax.fill(x, self.output_dB, facecolor = color_output, linewidth = linewidth, alpha = alpha)

        if input_if:
            self.input_dB = 10*np.log10(( fft_input*np.conj(fft_input) )/ maximum)
            self.input_dB = np.real(self.input_dB)

            ax.plot(x, self.input_dB, label="Input", color = color_input)
            ax.fill(x, self.input_dB, facecolor = color_input, linewidth = linewidth, alpha = alpha)

        try:
            plt.ylim(y_limit)
            plt.xlim(x_limit)
            plt.show
        except:
            return 'Datos insuficientes o invalidos'
