import numpy as np
import math
import numpy as np
import scipy
from lmfit import minimize, Parameters


class CubicFit:
	def __init__ (self, fit_data, thickness, radius, tension, poisson, c=1):
		self.fit_data = fit_data
		self.t  = float(thickness)
		self.R  = float(radius)
		self.T  = float(tension)
		self.nu = float(poisson)
		self.c  = 1

	def force_func (self, x, Elin, Ecub, x_shift, y_shift):
		# I will include a constant shift in both x and y;
		# that is not really included in the model
		# but should account for the fact that we pick (0,0) arbitratily;
		# these shifts should be small!
		linear_coefficient = ( 4*np.pi*Elin*self.t**3 ) / ( 3*(1-self.nu**2)*self.R**2 ) + np.pi*self.T
		cubic_coefficient  = self.c*Ecub*self.t / self.R**2
		force = linear_coefficient * (x-x_shift) + cubic_coefficient * (x-x_shift)**3 + y_shift
		return force
	
	def residual_function (self, pars):
		x_data = self.fit_data[0]
		y_data = self.fit_data[1]
		y_calc = self.force_func(x_data, pars['Elin'], pars['Ecub'], pars['x_shift'], pars['y_shift'])
		return y_data - y_calc	
	
	def perform_fit (self, guess, vary):
		self.params = Parameters()
		self.params.add('Elin', value=guess[0], vary=vary[0])
		self.params.add('Ecub', value=guess[1], vary=vary[1])
		self.params.add('x_shift', value=guess[2], vary=vary[2])
		self.params.add('y_shift', value=guess[3], vary=vary[3])

		fit_output = minimize(self.residual_function, self.params, method='leastsq')
		popt = np.array([fit_output.params['Elin'].value, fit_output.params['Ecub'].value, fit_output.params['x_shift'].value, fit_output.params['y_shift'].value])
		
		x_fit = np.linspace(np.min(self.fit_data[0]), np.max(self.fit_data[0]), 500)
		y_fit = self.force_func(x_fit, *popt)

		return popt, np.array([x_fit, y_fit])