import numpy as np
import numpy.fft as fft

class GeSbS:
    def __init__(self, N: float,T_window: float):

        self.N = N                                                                                                  # Muestreo
        self.T_window   = T_window                                                                                  # Ventana temporal

        # Ventana temporal derivados
        self.t_step     = T_window / N                                                                              # Paso temporal
        self.t_vector   = np.arange(-N/2,N/2,1) * self.t_step                                                       # Vector de tiempo

        # Ventana frecuencial
        self.w_window   = 2*np.pi / T_window                                                                        # Ventana frecuencial
        self.w_step     = self.w_window / N                                                                         # Paso frecuencias
        self.w_vector   = (np.pi/T_window) * np.concatenate((np.arange(-(N/2)+1,1,1), np.arange(0,(N/2),1)))        # Vector de frecuencias

    def Gaussian(self, power: float, lambda_0: float, FWHM: float, width: float):

        # instance declaration
        self.power         = power                                                                 # input power
        self.lambda_0      = lambda_0                                                              # central wavelength
        self.FWHM          = FWHM                                                                  # Full Width Half Maximum
        self.width         = width                                                                 # spectrum width
        self.c             = 3 * (10 **8)

        # equation values
        self.lambda_i      = lambda_0 - (FWHM/2)                                                   # initial wavelength
        self.lambda_f      = lambda_0 + (FWHM/2)                                                   # final wavelength
        self.lambda_vector = np.linspace(self.lambda_i,self.lambda_f,self.N)                       # lambda vector
        self.pulse_aux     = np.float128(-( (self.lambda_vector - lambda_0) / (width*2) ) ** 2)    # Gaussian exponential argument
        self.pulse         = np.exp(self.pulse_aux)                                                # Gaussian equation

        # Fourier
        self.fourier_pulse = fft.fft(self.pulse)
        self.input_field   = self.fourier_pulse / np.max(self.fourier_pulse)

        return self.input_field

    def Output_field(self,input_field, device_length: float, betas, A_eff: float, n2: float, gamma: float, step_number = 1000, losses = 0, alpha = 0):

        # instance declaration
        self.input_field         = input_field
        self.device_length       = device_length
        self.betas               = betas
        self.A_eff               = A_eff
        self.n2                  = n2
        self.gamma               = gamma
        self.losses              = losses
        self.alpha               = alpha
        self.step_number         = step_number

        # step size
        self.dz                  =  self.device_length / self.step_number

        # Non linear operator
        self.nonlinear_operator  = 1j*self.gamma*self.dz*(self.power)

        # dispersion operator
        self.dispersion_operator = np.exp(1j*(0.5*self.betas[0]*(self.w_vector**2)-self.w_vector)*self.dz)*np.exp(1j*(1/6)*self.betas[1]*(self.w_vector**3)*self.dz)*np.exp(-self.alpha/2*self.dz) # phase factor of the pump wave

        # Main loop
        Initial_HalfStep = input_field*np.exp((abs(input_field)**2)*self.nonlinear_operator/2) # 1/2 nonlinear

        for i in range(1,step_number+1):

            dispersion_HalfStep = fft.ifft(Initial_HalfStep) * self.dispersion_operator  # dispersion
            field               = fft.fft(dispersion_HalfStep)

            Initial_HalfStep        = field * np.exp((abs(field)**2)*self.nonlinear_operator/2)

        field_ssfm  = Initial_HalfStep * np.exp((abs(Initial_HalfStep)**2)*self.nonlinear_operator/2)
        self.final_Field = fft.ifft(field_ssfm)

        return self.final_Field
