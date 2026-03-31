import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator, interp2d, RectBivariateSpline
import math
import numpy as np
import os
import re
import sys
import platform
from numpy.polynomial import Polynomial
import cv2
import pandas as pd


class AFMForceMapData:
	def __init__ (self):
		"""
		AFMForceMapData is able to load a series of AFM tip deflection vs z-stage position files where the tip is pushed into different spots on a suspended membrane;
		it can then convert this data set to a 2D map of the compliance of the membrane;
		it can also project this data onto 1D where the compliance is stored as a function of the distance of where the membrane is poked to its center;
			-> this last part is partcularly helpful if the membrane is a circular drumhead
		"""
		self.operating_system = platform.system()
		if self.operating_system in ['windows', 'Windows']:
			self.separator = '\\'
		elif self.operating_system in ['mac', 'Mac', 'Darwin', 'darwin']:
			self.separator = '/'
		else:
			print('operating system not any of the possible options')
			print('current operating system is: ', self.operating_system)
		
		self.x_files, self.y_files, self.x_index, self.y_index = None, None, None, None
		self.approach_data, self.retract_data = [], []
		self.approach_fit_data, self.retract_fit_data = [], []
		self.approach_fit, self.retract_fit = [], []
		self.raw_compliance_array = None
		self.processed_compliance_array = None

	
	def moving_average(self, x, y, n):
		"""
		create a moving average
		x, y: data
		n: 2n data points evenly split around each point will be used for average
		"""
		ret = np.cumsum(y, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		return x[n-1:], ret[n - 1:] / n
	
	def bin_x (self, x, y, dx):
		"""
		at every point in x, average data in a bin dx/2 in both directions
		"""
		size = np.zeros(len(x))
		x_ave = np.zeros(len(x))
		y_ave = np.zeros(len(y))
		for ii, x0 in enumerate(x):
			mask = (x>=x0-dx/2)&(x<=x0+dx/2)
			x_ave[ii] = np.mean(x[mask])
			y_ave[ii] = np.mean(y[mask])
			size[ii] = len(x[mask])
		print(np.mean(size))
		return x_ave, y_ave
	
	def median_filter (self, data, threshold=10, window=100):
		"""
		function to detect outliers with a rolling median average;
		returns masks for good and bad points
		"""
		median     = pd.DataFrame(data).rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill').to_numpy().flatten()
		difference = np.abs((data - median)/median)
		good_bool  = difference <= threshold
		bad_bool   = np.invert(good_bool)
		return bad_bool, good_bool


	def get_filenames (self, folder, x_name, y_name):
		"""
		find the filenames of the relevant AFM files - not only delection and z-sensor pos (which are the two we need) are saved but also other data;
		- folder: directory of where the AFM data is stored
		- x_name: should be a str of the identifying filename of the x-data (for us this is z-sensor pos)
		- y_name: should be a str of the identifying filename of the y-data (for us this is tip deflection)
		"""
		x_files = np.array([], dtype=str)
		y_files = np.array([], dtype=str)
		x_index = []
		y_index = []

		for file_name in os.listdir(folder):
			# the files are organized by the pixel position of where the AFM tip poked the membrane
			# think of it as indices for row and column of the data grid
			# index is an array of [row, column]
			index = np.array(re.findall(r'\d+', file_name), dtype=int)
			if file_name.find(x_name)>-1:
				x_files = np.append(x_files, folder+self.separator+file_name)
				x_index.append(index)
			elif file_name.find(y_name)>-1:
				y_files = np.append(y_files, folder+self.separator+file_name)
				y_index.append(index)
		
		# sort both x and y data so that it is first sorted by the row and then by the column
		# mostly sorting x and y data the same way!
		x_index = np.array(x_index)
		sorted_indices = np.lexsort((x_index[:, 1], x_index[:, 0]))
		self.x_files, self.x_index = x_files[sorted_indices], x_index[sorted_indices]
		y_index = np.array(y_index)
		sorted_indices = np.lexsort((y_index[:, 1], y_index[:, 0]))
		self.y_files, self.y_index = y_files[sorted_indices], y_index[sorted_indices]
		return 1
	
	def load_data (self, x_files=None, y_files=None, x_index=None, y_index=None):
		"""
		load raw data
		- x_files: array of filenames for x-data
		- y_files: array of filenames for y-data
		- x_index: array specifying the pixel position of where membrane was poked corresponding to element in x_files
		- y_index: array specifying the pixel position of where membrane was poked corresponding to element in y_files
		"""
		self.approach_data = []
		self.retract_data = []
		if x_files is None:
			x_files = self.x_files
		if y_files is None:
			y_files = self.y_files
		if x_index is None:
			x_index = self.x_index
		if y_index is None:
			y_index = self.y_index

		for xx, x_ind in enumerate(x_index):
			x_data = np.loadtxt(x_files[xx])
			yy = np.where((y_index == x_ind).all(axis=1)) # make sure to import y_data at the same pixel
			y_data = np.loadtxt(y_files[yy][0])

			# sometimes there are huge steps at the beginning/end of these curves, so remove the first/last few points
 			# otherwise the "ceiling=..." line a few lines below might throw an error
			x_data = x_data[5:-5]
			y_data = y_data[5:-5]

			# split approach and retract curves
			y_max_idx = np.argmax(y_data) # this is where the y data is maximum;
								      # there should only be one large maximum, separating approach (to the left) and retraction (to the right)
			# separate approach and retract curves
			y_approach = y_data[:y_max_idx]
			x_approach = x_data[:y_max_idx]
			y_retract  = y_data[y_max_idx:]
			x_retract  = x_data[y_max_idx:]

			self.approach_data.append([x_approach, y_approach])
			self.retract_data.append([x_retract, y_retract])
		return 1
	

	def prepare_individual_data_for_fit (self, individual_approach_data, individual_retract_data):
		"""
		takes data of one pixel (separated for approach and retract of AFM tip) and selects parts of it which will be used to fit compliance
		- individual_approach_data: 2 by N array where 0/1 element is x/y data for one pixel during approach of AFM tip
		- individual_retract_data: 2 by N array where 0/1 element is x/y data for one pixel during retraction of AFM tip
		"""
		x_approach = individual_approach_data[0]
		y_approach = individual_approach_data[1]
		x_retract = individual_retract_data[0]
		y_retract = individual_retract_data[1]

		# find minimum in y_data marking beggining/end of contact of tip with sample
		y_approach_min_idx = np.argmin(y_approach)
		# only taking small set of approach data, hopefully this is the actually linear part
		y_approach = y_approach[y_approach_min_idx+40:y_approach_min_idx+100]
		x_approach = x_approach[y_approach_min_idx+40:y_approach_min_idx+100]

		y_retract_min_idx = np.argmin(y_retract)
		y_retract = y_retract[:y_retract_min_idx]
		x_retract = x_retract[:y_retract_min_idx]
		# this pretty much forces the retract data to not go further down that the approach data
		# I don't know why this is a thing but it was a thing in the old code
		# hopefully this i the most linear part?
		temp_idx = max(np.argwhere(y_retract>min(y_approach)).flatten())
		x_retract, y_retract = x_retract[:temp_idx], y_retract[:temp_idx]
		x_retract, y_retract   = x_retract[-70:], y_retract[-70:]

		return x_approach, y_approach, x_retract, y_retract
    

	def fit_map_compliance (self, x_index, approach_data, retract_data, k_tip=3, fit_type='linear'):
		"""
		fits compliance to complete AFM data set;
		- x_index: array specifying the pixel position of where membrane was poked corresponding to element in x_files
		- approach_data: Mx2xN array of all approach data: M iterates thorugh all pixels, 2 is for x, y data, N is for number of points within each data set
		- extract_data: Mx2xN array of all retract data: M iterates thorugh all pixels, 2 is for x, y data, N is for number of points within each data set
		- k_tip: spring constant of AFM tip
		- fit_type: only "linear" allowed for now
		"""
		self.approach_fit_data, self.retract_fit_data = [], []
		self.approach_fit, self.retract_fit = [], []
		dimension = int(np.sqrt(len(x_index)))
		# store compliance results in 2D array where each element corresponds to a pixel where the data was taken
		compliance_array=np.zeros((dimension, dimension))
		for xx, x_ind in enumerate(x_index):
			x_approach, y_approach, x_retract, y_retract = self.prepare_individual_data_for_fit(approach_data[xx], retract_data[xx])
			if fit_type=='linear':
				# fit line to approach data
				fit_approach = Polynomial.fit(x_approach, y_approach, deg=1)
				fit_approach = fit_approach.convert().coef
				self.approach_fit_data.append([x_approach,y_approach])
				y_approach_fitvals = fit_approach[1]*x_approach + fit_approach[0]
				self.approach_fit.append([x_approach, y_approach_fitvals])

				# fit line to retract data
				fit_retract = Polynomial.fit(x_retract, y_retract, deg=1)
				fit_retract = fit_retract.convert().coef
				self.retract_fit_data.append([x_retract,y_retract])
				y_retract_fitvals = fit_retract[1]*x_retract + fit_retract[0]
				self.retract_fit.append([x_retract, y_retract_fitvals])

				# use average between apprach and retract
				slope_ave = (fit_approach[1]+fit_retract[1])/2
				compliance_array[x_ind[0], x_ind[1]] = (1/slope_ave - 1)/k_tip

			else:
				print('need to specify an allowed fit type for the approach and retract curves')
				print('allowed values are: "linear"')
				sys.exit()

		self.raw_compliance_array = compliance_array
		return compliance_array
	
	def post_process_compliance_array (self, compliance_array, threshold_compliance=6):
		"""
		smooth 2D array of compliance map
		"""
		# both for loops pretty much run some moving average type smoothing;
		# however this is very manual and not super motivated (e.g. why are diagonals not included?)
		# this is only done because it's what was done previously
		# a decent Gaussian average should be much more motivated and better
		# OR even better would be to use raw data until the very end (e.g. to get radial compliance data) and then only average/smooth that final data
		for ii, row in enumerate(compliance_array):
			for jj, element in enumerate(row):
				if (np.abs(element)>threshold_compliance) or (element==0):
					if ii>0 and jj>0 and ii<np.shape(compliance_array)[0]-1 and jj<np.shape(compliance_array)[1]-1:
						compliance_array[ii,jj]=(compliance_array[ii+1,jj]+compliance_array[ii,jj+1]+compliance_array[ii-1,jj]+compliance_array[ii,jj-1])/4
					else:
						compliance_array[ii,jj] = 0

		for ii, row in enumerate(compliance_array):
			for jj, element in enumerate(row):
				if ii<np.shape(compliance_array)[0]-1 and jj<np.shape(compliance_array)[1]-1 and np.abs(element-(compliance_array[ii+1,jj]+compliance_array[ii,jj+1]+compliance_array[ii-1,jj]+compliance_array[ii,jj-1])/4)>(0.002):
					compliance_array[ii,jj]=(compliance_array[ii+1,jj]+compliance_array[ii,jj+1]+compliance_array[ii-1,jj]+compliance_array[ii,jj-1])/4
		self.processed_compliance_array = compliance_array
		return self.processed_compliance_array

	
	def find_circle (self, compliance_array):
		"""
		uses open-cv to find the center of a circle - helpful if suspended membrane is circular drumhead;
		input is 2D array of compliance map;
		"""
		blurred_image = compliance_array
		# Convert the 2D array to an 8 bit grayscale image
		# blurred_image = np.array(blurred_image, dtype=np.uint8)

		# Apply a Gaussian blur to reduce noise and improve circle detection
		blurred_image = cv2.GaussianBlur(blurred_image, (7, 7), 1.5)
		# blurred_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
		blurred_image = np.array(blurred_image, dtype=np.uint8)

		# threshold to binary
		blurred_image = cv2.threshold(blurred_image,.1,255,cv2.THRESH_BINARY)[1]

		# Use the Hough Circle Transform to detect circles
		circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.5, minDist=50,
                            param1=300,param2=.9,minRadius=5,maxRadius=15)
		return circles
	

	def find_circle_manual (self, compliance_array, circle_center_guess):
		"""
		finding the center of a circle of a 2D map based on initial guess;
		necessary to know if we want to project 2D map on plot of compliance vs distance from center
		- compliance_array: 2D array of compliance map
		- circle_center_guess: np.array([x_guess, y_guess]), initial guess of where the center of the circle is
		"""
		# pretty much just takes line cuts of the data close to the center_guess and finds the maximum of each line cut;
		# then averages over all maxima - that's the new circle center
		x0, y0 = np.array(circle_center_guess).astype(int)
		scan_size = 5
		meaner_x, meaner_y = 0, 0
		# plt.figure()
		size=0
		for l in range (x0-scan_size,x0+scan_size):
			if l>=0 and l<np.shape(compliance_array)[0]:
				meaner_x=meaner_x+np.argmax(compliance_array[l,x0-scan_size:x0+scan_size])+x0-scan_size		
				size+=1
			# plt.plot(range(x0-scan_size,x0+scan_size), compliance_array[l,x0-scan_size:x0+scan_size])

		meaner_xf=float(meaner_x)/size
		meaner_xf=int(math.floor(meaner_xf+.5))

		# plt.figure()
		size=0
		for l in range (y0-scan_size,scan_size+y0):
			if l>=0 and l<np.shape(compliance_array)[1]:
				meaner_y=meaner_y+np.argmax(compliance_array[y0-scan_size:y0+scan_size,l])+y0-scan_size
				size+=1
			# plt.plot(range(y0-scan_size,y0+scan_size), compliance_array[y0-scan_size:y0+scan_size,l])

		meaner_yf=float(meaner_y)/size
		meaner_yf=int(math.floor(meaner_yf+.5))
		return meaner_xf, meaner_yf
			

	def create_radial_plot_data (self, compliance_array, circle_center, scan_window_size, radius, zero_compl=0):
		"""
		project 2D compliance map onto 1D data as a function of distance from the circle center rather than absolute pixel position;
		no averaging happens here: every pixel is just assigned a distance from circle center based on pixel position
		- compliance_array: 2D array of compliance map
		- circle_center: center of the drumhead in pixels
		- scan_window_size: range of the entire AFM map in um
		- radius: radius of the drumhead in pixels
		- zero_compl: shift final compliance data such that the value given by zero_compl is at zero
		"""
		# x, y = np.indices(np.shape(compliance_array))/len(compliance_array)*scan_window_size/radius
		scan     = np.linspace(0,scan_window_size/radius,len(compliance_array))
		x = np.tile(scan, (len(compliance_array), 1))
		y = x.T
		x_shifted, y_shifted = x-scan[circle_center[0]], y-scan[circle_center[1]]
		distance = np.sqrt(x_shifted**2 + y_shifted**2) # distance of every pixel from the circle center
		distance   = distance.flatten()
		compliance = compliance_array.flatten()-zero_compl
		compliance, distance = compliance[np.argsort(distance)], distance[np.argsort(distance)]

		# distance, compliance = np.zeros(int(np.shape(compliance_array)[0]**2)), np.zeros(int(np.shape(compliance_array)[1]**2))
		# scan     = np.linspace(0,scan_window_size/radius,len(compliance_array))

		# origin_x, origin_y = scan[circle_center[0]], scan[circle_center[1]]
		# counter=0
		# for i in range(0,np.shape(compliance_array)[0]):
		# 	for j in range(0,np.shape(compliance_array)[1]):
		# 		distance[counter]   = np.sqrt((scan[i]-origin_x)**2+(scan[j]-origin_y)**2)
		# 		compliance[counter] = compliance_array[j,i]-zero_compl
		# 		counter=counter+1
		# compliance, distance = compliance[np.argsort(distance)], distance[np.argsort(distance)]		
		
		return distance, compliance

	def create_radial_fit_data (self, compliance_array, circle_center, radius, scan_window_size, r_divs=20, theta_divs=80, calib_boundary=1.2):
		"""
		project 2D compliance map onto 1D data as a function of distance from the circle center rather than absolute pixel position;
		this is done by changing coordinates from x-y to radial coordinates and then averaging over the angle and certain radius values;
		I am not sure if this is the best way of doing things; I still feel like the best approach is to take raw (un-processed) compliance map and
		create radial data with self.create_radial_plot_data and then take that data set and remove outliers and run averaging
		- compliance_array: 2D array of compliance map
		- circle_center: center of the drumhead in pixels
		- scan_window_size: range of the entire AFM map in um
		- radius: radius of the drumhead in pixels
		"""
		scan_x = np.linspace(0, scan_window_size/radius, np.shape(compliance_array)[0])
		scan_y = np.linspace(0, scan_window_size/radius, np.shape(compliance_array)[1])

		origin_x = scan_x[circle_center[0]]
		origin_y = scan_y[circle_center[1]]

		radial_profile = np.zeros([r_divs*theta_divs,2])
		radial_avg     = np.zeros(r_divs)

		radial_dist    = np.linspace(0,calib_boundary,r_divs)

		indices_radial = [k for (k,val) in enumerate(radial_dist) if val>1.1] #trying this 20230607
		indices_useful = [k for (k,val) in enumerate(radial_dist) if val<1]

		ftemp = RectBivariateSpline(scan_x, scan_y, compliance_array.T)
		f = lambda xnew, ynew: ftemp(xnew, ynew).T
		# f = interp2d(scan_x,scan_y,compliance_array,kind='cubic')
		counter, counter_r = 0, 0
		for r in radial_dist:
			for theta in np.linspace(0,1.9375*np.pi,theta_divs):
			
				xnew = origin_x+r*np.cos(theta)        #taking a line cut along the theta direction and defining the x value
				ynew = origin_y+r*np.sin(theta)			#taking a linecut along the theta direction and defining the y value

				radial_profile[counter,0]=r
				radial_profile[counter,1:]=f(xnew,ynew)      #interpolate to get the radial profile 

				counter=counter+1

			radial_avg[counter_r]=np.mean(radial_profile[counter-theta_divs:counter-1,1])
			 #I use the average value at each radial linecut to fit the numerical results. radial_avg is the array that stores those average values
			counter_r=counter_r+1

		zero_compl=np.mean(radial_avg[indices_radial])
		radial_avg=radial_avg-zero_compl*np.ones(r_divs)
		radial_profile[:,1]=radial_profile[:,1]-zero_compl*np.ones(r_divs*theta_divs)
		return radial_dist[indices_useful[1:]],radial_avg[indices_useful[1:]], zero_compl