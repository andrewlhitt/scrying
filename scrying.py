import numpy as np 
from dataclasses import dataclass
from bisect import bisect_left
from skimage.segmentation import find_boundaries
import warnings 

nucleation_rate_modes = ['constant']
orientation_modes = ['random','updown']
center_modes = ['anywhere','central']
growth_rate_modes = ['constant','physical']
snapshot_modes = ['none','time','area']
get_image_options = ['final','snapshot','time']
import_nucleation_data_position_orders = ['xy','yx']
angle_units = ['degrees','radians','revolutions']
angle_chiralities = ['cw','ccw']
zero_directions = ['right','left','up', 'down']
import_nucleation_data_final_column_types = ['time','size']
get_grain_boundary_modes = ['all','internal','external']

class Simulator: 
	""" 
	An object that handles the full simulation process.

	Attributes
	----------
	width : int, default 128 
		the horizontal dimension of produced images 
	height : int, default 128  
		the vertical dimension of produced images
	crystal_sides : int, default 3 
		the number of sides of the crystal (overriden by `shape_array`, if used) 
	shape_array : numpy.array, optional 
		an array describing the growth vectors from the center of a crystal 
		can be produced via scrying.get_shape_array(...) for arbitrary, convex polygons 
	maximum_time : int, default 100 
		the maximum number of time steps the simulator will run before terminating 
	nucleation_rate : float, default 1.0 
		the expected number of nucleation events per time step (as a Poisson parameter)
	nucleation_rate_mode : {'constant'}, optional
		a setting that describes how the nucleation rate varies during the simulation 
	maximum_crystals : int, default 5 
		the maximum number of crystals before nucleation events are prevented
	center_mode : {'anywhere', 'central'}, optional
		a setting that describes where new crystals are allowed to nucleate. 
		'anywhere' allows a nucleation to occur at any non-crystallized pixel. 
		'central' limits nucleation to occur within a central region defined by `nucleation_region_percent` 
	nucleation_region_percent : float, optional
		only used when `center_mode` is central. 
		crystals can only nucleate within `nucleation_region_percent` to (1-`nucleation_region_percent`) along each dimension
	orientation_mode : {'random', 'updown'}
		a setting that describes how an orientation is assigned to a new crystal, relative to a base orientation determined at the start of the simulation. 
		'random' means that the crystal will be randomly oriented relative to the base orientation
		'updown' means that the crystal will be randomly oriented either along the base orientation or against the base orientation. 
	growth_rate : float, default 1.0 
		the increase in each crystal's size each time step 
	growth_rate_mode : {'constant', 'physical'}
		a setting that describes how the growth rate varies during simulation
		'constant' gives uniform growth over time
		'physical' gives decreasing growth over time (proportional to 1/size)
	periodic_boundary : bool , default False 
		whether opposite edges of the simulation are considered connected 
		False (default) is more "physically" accurate
		True produces simulated images that are translationally invariant 
	snapshot_mode : {'none', 'time', 'area'}
		defines the conditions under which an early stage "snapshot" is taken 
		'none' means no snapshot will be taken 
		'time' means a snapshot will be taken at the `snapshot_time` time step of the simulation
		'area' means that a snapshot will be taken when `snapshot_area`% of the image is crystallized 
	snapshot_time : int, optional 
		see `snapshot_mode` 
	snapshot_area : float, optional
		see `snapshot_mode` 
	end_after_snapshot : bool, default False 
		whether the simulation is terminated after the snapshot is taken 
	stop_nucleation_after_snapshot : bool, default False 
		whether nucleation events are prevented after the snapshot is taken 
	save_evolution: bool, default True 
		whether the image at every time step is saved for later access via get_image_evolution(...) and get_image(time=...)
	random_seed: int or float, optional
		the seed used by the random number generator 
	orientation_precision : int, optional
		the number of digits to which angle values are truncated.


	"""
	__arg_list = ['width','height','crystal_sides','shape_array','maximum_time',
					'nucleation_rate','nucleation_rate_mode','maximum_crystals','center_mode','nucleation_region_percent',
					'orientation_mode',
					'growth_rate','growth_rate_mode','periodic_boundary',
					'snapshot_mode','snapshot_time','snapshot_area','end_after_snapshot','stop_nucleation_after_snapshot',
					'save_evolution','random_seed','orientation_precision']

	def __init__(self, 
		width: int = 128, height: int = 128, crystal_sides: int = 3, shape_array: np.array = None, maximum_time: int = 100, 
		nucleation_rate: float = 1.0, nucleation_rate_mode: str = 'constant', maximum_crystals: int = 5, center_mode: str = 'anywhere', nucleation_region_percent: float = 0.0,
		orientation_mode: str = 'random', 
		growth_rate: float = 1.0, growth_rate_mode: str = 'constant', periodic_boundary: bool = False, 
		snapshot_mode: str = 'none', snapshot_time: int = 0, snapshot_area: float = 0.25, end_after_snapshot: bool = False, stop_nucleation_after_snapshot: bool = False,
		save_evolution: bool = True, random_seed: int = None, orientation_precision: int = 5): 

		# Initializing the random number generator
		self._RNG = np.random.default_rng(random_seed)

		# default parameters
		self.width = width
		self.height = height
		self.crystal_sides = crystal_sides
		self.shape_array = shape_array 
		if self.shape_array is not None: 
			self.crystal_sides = self.shape_array.shape[0] 
		self.maximum_time = maximum_time 
		
		# nucleation parameters
		self.nucleation_rate = nucleation_rate
		self.nucleation_rate_mode = nucleation_rate_mode.lower() 
		self.maximum_crystals = maximum_crystals
		self.center_mode = center_mode.lower()
		self.orientation_mode = orientation_mode.lower()
		self.orientation_precision = orientation_precision
		self.nucleation_region_percent = nucleation_region_percent

		# growth parameters
		self.growth_rate = growth_rate
		self.growth_rate_mode = growth_rate_mode.lower()
		self.periodic_boundary = periodic_boundary 

		# snapshot parameters 
		self.snapshot_mode = snapshot_mode.lower()
		self.snapshot_time = snapshot_time
		self.snapshot_area = snapshot_area
		self.end_after_snapshot = end_after_snapshot
		self.stop_nucleation_after_snapshot = stop_nucleation_after_snapshot

		# data parameters
		self.save_evolution = save_evolution 
		self._imported_data = None 

	def change_settings(self, **kwargs): 
		"""
		A generalized method for modifying the parameters of the simulator. 

		Parameters
		----------
		**kwargs : dict 
			Arguments for changes to simulator settings. 
			See `Simulator` for a list of accepted arguments. 

		"""
		for key in kwargs: 
			if key in self.__arg_list: 
				if key == 'random_seed':
					self._RNG = np.random.default_rng(kwargs.get(key))
				elif key == 'shape_array':
					self.shape_array = kwargs.get(key)
					self.crystal_sides = self.shape_array.shape[0] 
				else:
					self.__setattr__(key,kwargs.get(key))
			else: 
				raise AttributeError(f'Unrecognized parameter \'{key}\' in change_simulator_settings(...)')
		
	def run_simulation(self, use_imported_data: bool = False): 
		"""
		Using the existing configuration of the simulator, perform a single crystal growth simulation.

		Parameters
		----------
		use_imported_data : bool , default False 
			whether to use nucleation data that has been imported with import_nucleation_data(...)
	
		"""

		if use_imported_data and (self._imported_data is None): 
			raise ValueError(f'No nucleation data has been imported. Use the import_nucleation_data(...) function before calling run_simulation(...).')
			# return 

		if use_imported_data and (self._imported_data[self._imported_data[:,3] >= self.maximum_time].shape[0] > 0): 
			warnings.warn(f'Simulation will conclude before the nucleation of all crystals in the imported data. Consider increasing maximum_time to at least {int(np.max([self._imported_data[:,3]]))}') 

		#if (self.snapshot_mode == 'time') and (self.maximum_time < self.snapshot_time): 
		#	warnings.warn(f'The snapshot time exceeds maximum simulation time.')

		self._reset_simulation_parameters() 
		while (self._current_time < self.maximum_time): 

			if use_imported_data: self._nucleate_crystals_from_data()
			else: self._nucleate_crystals()
			self._grow_crystals() 
			self._take_snapshot()

			if self.save_evolution: self._image_over_time.append(self._image.copy())

			# Criteria for ending the simulation early
			if self._get_area_coverage() == 1.0: 
				if (not self._snapshot_taken) and (self.snapshot_mode != 'none'): 
					warnings.warn('Snapshot criteria were not met during simulation.')
				break 
			if (self._snapshot_taken and self._current_crystals == 0 and self.stop_nucleation_after_snapshot): 
				warnings.warn('Snapshot was taken with no crystals present, preventing the simulation from proceeding.')
				break
			if (self._snapshot_taken and self.end_after_snapshot): break				

			self._current_time += 1 
		else: 
			if (not self._snapshot_taken) and (self.snapshot_mode != 'none'): 
					warnings.warn('Snapshot criteria were not met during simulation.')
			warnings.warn('Simulation concluded before reaching 100% area coverage. Consider increasing maximum_time.')

		self._conclude_simulation()

	def import_nucleation_data(self, nucleation_data: np.array, 
							   position_order: str = 'xy',
							   angle_unit: str = 'revolutions', chirality: str = 'cw', zero_direction = 'right', 
							   final_column: str = 'time', autoconfigure_snapshot: bool = False,
							   has_header_row: bool = False, has_index_column: bool = True): 

		"""
		Loads user-specified nucleation data into the simulator. 

		Parameters
		----------
		nucleation_data : numpy.array 
			the nucleation data, formatted as columns of [position1, position2, angle, time or size]
		position_order : {'xy', 'yx'}
			whether the first two columns are horizontal-then-vertical or the reverse 
		angle_unit : {'revolutions', 'degrees', 'radians'}
			the unit of the orientations from `nucleation_data` 
		chirality : {'cw', 'ccw'}
			the direction of increasing angles 
		zero_direction : {'right', 'left', 'up', 'down'}
			the direction from which the angle difference is measured 
			'right' corresponds to the positive x-axis in a normal Cartesian plane 
		final_column : {'time', 'size'}
			the type of data encoded in the last column. 
			'time' means that the nucleation time of the crystal is recorded
			'size' means that the current size of the crystal is recorded
		autoconfigure_snapshot : bool, default False 
			whether to update the snapshot settings using the provided data 
		has_header_row : bool, default False 
			whether the imported data has a header row or not 
		has_index_column: bool, default True
			whether the array has an index column 
		
		Notes 
		-----
		When `final_column` is set to "size", an approximate nucleation time will be calculated for each crystal assuming a constant growth rate. 

		"""
		self._verify_string_variable_setting('position_order',position_order,import_nucleation_data_position_orders)
		self._verify_string_variable_setting('angle_unit',angle_unit,angle_units)
		self._verify_string_variable_setting('chirality',chirality,angle_chiralities)
		self._verify_string_variable_setting('zero_direction',zero_direction,zero_directions)
		self._verify_string_variable_setting('final_column',final_column,import_nucleation_data_final_column_types)

		if has_header_row: first_row = 1 
		else: first_row = 0

		if has_index_column: first_column = 1
		else: first_column = 0

		data = nucleation_data[first_row:,first_column:].astype('float')
		

		if position_order == 'xy': data[:,[0,1]] = data[:,[1,0]].astype('int')
		elif position_order == 'yx': data[:,[0,1]] = data[:,[0,1]].astype('int')

		if angle_unit == 'degrees': full_rotation = 360 
		elif angle_unit == 'radians': full_rotation = 2*np.pi
		elif angle_unit == 'revolutions': full_rotation = 1

		if chirality == 'cw': angle_multiplier = 1 
		elif chirality == 'ccw': angle_multiplier = -1 

		if zero_direction == 'right': angle_addend = 0. 
		elif zero_direction == 'down': angle_addend = 0.25
		elif zero_direction == 'left': angle_addend = 0.5
		elif zero_direction == 'up': angle_addend = 0.75

		data[:,2] = angle_addend + (angle_multiplier*data[:,2]/full_rotation).astype('float')

		# if data reports crystal size, calculates an approximate nucleation time for each crystal
		# this assumes globally uniform growth rate 
		if final_column == 'size': 
			final_time = np.max(data[:,3])
			data[:,3] = np.round((final_time - data[:,3])).astype('int')
		else:
			final_time = np.max(data[:,3])

		# if autoconfigure is enabled, sets up the snapshot to capture the approximate image 
		# that would have been used for the data measurement 
		if autoconfigure_snapshot: 
			self.snapshot_time = final_time - 1 
			self.snapshot_mode = 'time'

		self._imported_data = data 

	def export_nucleation_data(self,position_order: str = 'xy',
							   angle_unit: str = 'revolutions', chirality: str = 'cw', zero_direction: str = 'right') -> np.array:
		"""
		Produces an array describing the nucleation events from a completed simulation. 
		
		Parameters
		----------
		position_order : {'xy', 'yx'}
			which position data (horizontal or vertical) is reported first
		angle_unit : {'revolutions', 'degrees', 'radians'}
			the unit of the orientations reported
		chirality : {'cw', 'ccw'}
			the direction of increasing angles 
		zero_direction : {'right', 'left', 'up', 'down'}
			the direction from which the angle difference is measured 
			'right' corresponds to the positive x-axis in a normal Cartesian plane 

		Returns 
		-------
		data : numpy.array
			An array summarizing the nucleation information from the simulation
			Row format is [index, center_x, center_y, orientation, nucleation_time]
		
		"""
		self._verify_string_variable_setting('position_order',position_order,import_nucleation_data_position_orders)
		self._verify_string_variable_setting('angle_unit',angle_unit,angle_units)
		self._verify_string_variable_setting('chirality',chirality,angle_chiralities)
		self._verify_string_variable_setting('zero_direction',zero_direction,zero_directions)

		data = np.zeros((self._current_crystals,5))
		for i in range(1,self._current_crystals+1):
			data[i-1,0] = i # index/crystal label
			data[i-1,1],data[i-1,2] = self._crystals[i].center
			data[i-1,3] = self._crystals[i].orientation
			data[i-1,4] = self._crystals[i].nucleation_time

		if position_order == 'xy': data[:,[1,2]] = data[:,[2,1]].astype('int')
		elif position_order == 'yx': data[:,[1,2]] = data[:,[1,2]].astype('int')

		if angle_unit == 'degrees': full_rotation = 360 
		elif angle_unit == 'radians': full_rotation = 2*np.pi
		elif angle_unit == 'revolutions': full_rotation = 1

		if chirality == 'cw': angle_multiplier = 1 
		elif chirality == 'ccw': angle_multiplier = -1 

		if zero_direction == 'right': angle_addend = 0. 
		elif zero_direction == 'down': angle_addend = 0.25
		elif zero_direction == 'left': angle_addend = 0.5
		elif zero_direction == 'up': angle_addend = 0.75

		data[:,3] = ((angle_addend + angle_multiplier*data[:,3])*full_rotation)%full_rotation
		
		return data

	def get_image(self, mode: str ='final', time: int = None) -> np.array: 
		"""
		Retrieves a single "frame" from the most recently completed simulation. 
		
		Parameters
		----------
		mode : {'final', 'snapshot', 'time'}
			which image should be retrieved
			'final' produces the last frame of the simulation
			'snapshot' produces the snapshot taken when specific conditions were met. (see `snapshot_mode` in Simulator(...))
			'time' produces the frame taken at the time step specified by `time` 
		time : int, optional 
			see `mode` 

		Returns 
		-------
		image : numpy.array
			A 2D array (y,x) showing the state of the simulation at a single time step.
		
		"""
		if mode not in get_image_options:
			raise AttributeError(f'\'{mode}\' is not a recognized option for get_image. Accepted options are: {[option for option in get_image_options]}')
		
		image = None

		if (mode == 'time') or (time is not None): 
			if self.save_evolution: image = self._image_evolution_array[time,:,:].copy()
			else: raise ValueError(f'Time-series data has not been saved. Image at time {time} cannot be retrieved.') 
		elif mode == 'final': 
			image = self._image.copy()
		elif mode == 'snapshot': 
			image = self._snapshot.copy()
		return image

	def get_image_evolution(self) -> np.array: 
		"""
		Retrieves the time-series data of the most recently completed simulation. 
		
		Returns 
		-------
		numpy.array
			A 3D array (t,y,x) containing the state of the simulation are every time step. 
		
		"""
		if self.save_evolution: return self._image_evolution_array.copy()
		else: 
			raise ValueError(f'WARNING: Time-series data has not been saved.')

	def get_grain_structure(self, image: np.array, maximum_misorientation: float = 0, symmetry: int = 1) -> np.array:
		"""
		Produces an image where different crystals of similar orientations are given the same label.
			
		Parameters
		----------
		image : numpy.array 
			The image of simulated crystals from get_image(...). 
		maximum_misorientation : float, default 0 
			The largest difference in orientation (in revolutions) between two crystals sharing a label. 
		symmetry : int, default 1 
			The rotational symmetry of the crystal structure.  

		Returns 
		-------
		image_groups : numpy.array
			An array where similarly oriented crystals share the same label. 

		Notes
		----- 
		The algorithm used to merge groups is an adaptation of complete-linkage clustering for a periodic, 1D system. 
		The heuristic for group distance is farthest neighbor (e.g. maximum of next group - minimum of current group)

		When permitting non-zero misorientation, 
		(1) angles are split into groups where adjacent angles have a difference less than the misorientation
		(2) within these groups, angles are separated and clustered using complete-linkage clustering. 

		"""

		symmetry_angle = round(1./symmetry,self.orientation_precision)
		
		crystal_to_orientation = dict({0:-1.})
		orientations = set()
		for i in range(1,len(self._crystals)): 
			crystal_orientation = round((self._crystals[i].orientation)%(symmetry_angle),self.orientation_precision)
			crystal_to_orientation[i] = crystal_orientation
			orientations.add(crystal_orientation)

		sorted_orientations = sorted(list(orientations))

		if maximum_misorientation > 0: 

			# add the smallest angle + symmetry to check for adjacency across the modular boundary 
			sorted_orientations.append(round(sorted_orientations[0] + symmetry_angle,self.orientation_precision))
			
			# do an initial pass to attempt to break up the full set
			orientation_diffs = np.diff(sorted_orientations)
			split_indices = [i+1 for i in range(orientation_diffs.shape[0]-1) if orientation_diffs[i] > maximum_misorientation]
			initial_groups = [sorted_orientations[i:j] for i, j in zip([None]+split_indices, split_indices+[-1])]
			
			# if first and last groups are sufficiently similar, merge into one group
			if (orientation_diffs[-1] < maximum_misorientation) and len(initial_groups) > 1:
				initial_groups[0] = [orientation-symmetry_angle for orientation in initial_groups[-1]] + initial_groups[0]
				del initial_groups[-1]
			
			final_groups = list()
			
			for initial_group in initial_groups:
				group_range = np.max(initial_group) - np.min(initial_group)

				if group_range > maximum_misorientation:
					# do complete-linkage agglomeration if necessary
					clinked_groups = self._iteratively_clink_group(initial_group,maximum_misorientation)
					final_groups.extend(clinked_groups)
				else: 
					final_groups.append(initial_group)

			orientations_to_group = dict({-1.: 0})
			for i in range(len(final_groups)): 
				for j in range(len(final_groups[i])): orientations_to_group[(final_groups[i][j]%symmetry_angle)] = i+1

		else: 
			orientations_to_group = dict({-1.: 0})
			for i in range(len(orientations)): orientations_to_group[sorted_orientations[i]] = i+1

		# DSM on stackoverflow: https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
		image_orientations = np.vectorize(crystal_to_orientation.get)(image)
		image_groups = np.vectorize(orientations_to_group.get)(image_orientations)

		return image_groups

	def get_grain_boundaries(self, image: np.array, mode: str ='all') -> np.array: 
		"""
		Produces an image where pixels are identified as boundary points or not. 

		Parameters
		----------
		image : numpy.array 
			The image of simulated crystals, usually from get_grain_structure(...).
		mode : {'all', 'internal', 'external'}
			Which boundaries will be identified 
			'internal' means that only boundaries between crystals are shown.
			'external' means only boundaries between a crystal and the background are shown. 
			'all' means both types of boundaries are shown. 

		Returns 
		-------
		boundary_img : numpy.array
			An array where boundary pixels are labeled 1 (all other pixels labeled 0).  

		Notes
		----- 
		Boundaries produced via this function will be 2 pixels wide. 

		"""
		self._verify_string_variable_setting('mode',mode,get_grain_boundary_modes)
		if mode == 'external':
			boundary_img = find_boundaries((image != 0),mode='thick')
		elif mode == 'internal': 
			external_boundaries = find_boundaries((image != 0),mode='thick')
			all_boundaries = find_boundaries(image,mode='thick')
			boundary_img = all_boundaries.astype('int') - external_boundaries.astype('int')
		elif mode == 'all':
			 boundary_img = find_boundaries(image,mode='thick')

		return boundary_img 

	def get_grain_boundary_length(self, image: np.array, mode: str ='all') -> int:
		"""
		Calculates the approximate length of the grain boundaries present in an image. 

		Parameters
		----------
		image : numpy.array 
			The image of simulated crystals, usually from get_grain_structure(...).
		mode : {'all', 'internal', 'external'}
			see get_grain_boundaries(...)

		Returns 
		-------
		int
			The approximate length of the grain boundary, in pixels. 

		Notes
		----- 
		The sum is divided by two because get_grain_boundaries(...) produces boundaries 2 pixels wide.

		"""
		boundary_image = self.get_grain_boundaries(image,mode)
		return int(np.sum(boundary_image)/2)

	# Main loop functions 
	def _reset_simulation_parameters(self):
		self._current_time = 0 
		self._image = np.zeros((self.height,self.width),dtype=int)
		self._crystals = list([None]) # this allows the list index to match the image label
		self._current_crystals = 0 
		self._primary_orientation = round(self._RNG.random(),self.orientation_precision)
		self._total_crystal_area = 0 
		self.snapshot = np.zeros((self.height,self.width),dtype=int)
		self._snapshot_taken = False 
		self.snapshot_taken_at_time = 0
		if self.save_evolution: self._image_over_time = list()
 

	def _nucleate_crystals(self):
		if self._snapshot_taken and self.stop_nucleation_after_snapshot: return 
		new_crystals = self._RNG.poisson(self._get_current_nucleation_rate())
		for new_crystal in range(new_crystals):
			if self._current_crystals >= self.maximum_crystals: return 
			if self._get_area_coverage() == 1.0: return 

			new_center = self._assign_new_center() 
			if new_center is None: return 

			new_orientation = self._assign_new_orientation()
			self._nucleate_new_crystal(new_center,new_orientation)

	def _nucleate_crystals_from_data(self):
		new_crystals = self._imported_data[self._imported_data[:,3]==self._current_time]
		for new_crystal in new_crystals: 
			new_center = int(new_crystal[0]),int(new_crystal[1])
			new_orientation = new_crystal[2]
			self._nucleate_new_crystal(new_center,new_orientation)	

	def _grow_crystals(self):
		if self._get_area_coverage() == 1.0: return 
		if self._current_crystals == 0: return 
		finished_updating_points = np.zeros((self._current_crystals+1))
		finished_updating_points[0] = 1

		for i in range(1,self._current_crystals+1):	
			self._crystals[i].radii.append(self._crystals[i].radii[-1])
			self._crystals[i].areas.append(self._crystals[i].areas[-1])
			self._crystals[i].radii[-1] += self._get_growth_rate(i)

		while not np.all(finished_updating_points):
			for i in range(1,self._current_crystals+1):
				if finished_updating_points[i]: continue
				candidate_points_copy = self._crystals[i].candidate_points.copy()
				for candidate_point,size in candidate_points_copy.items(): self._check_candidate_point(candidate_point,size,i)
				if self._crystals[i].candidate_points == candidate_points_copy: finished_updating_points[i] = 1


	def _take_snapshot(self): 
		self._verify_string_variable_setting('snapshot_mode',self.snapshot_mode,snapshot_modes)

		if self.snapshot_mode == 'none': return 
		if self._snapshot_taken: return 
		if (self.snapshot_mode == 'area' and self._get_area_coverage() >= self.snapshot_area) or (self.snapshot_mode == 'time' and self._current_time >= self.snapshot_time):
		   self._snapshot_taken = True
		   self.snapshot_taken_at_time = self._current_time
		   self._snapshot = self._image.copy()

	def _conclude_simulation(self):
		if self.save_evolution:
			self._image_evolution_array = np.rollaxis(np.dstack(self._image_over_time),-1)
		
	# Main loop nested functions  		
	def _get_current_nucleation_rate(self) -> float: 
		self._verify_string_variable_setting('nucleation_rate_mode',self.nucleation_rate_mode,nucleation_rate_modes)

		if self.nucleation_rate_mode == 'constant': return self.nucleation_rate
		else: return self.nucleation_rate

	# This produces an appropriate center for a new crystal (y,x)
	# It will return None if there is no available pixel for a center.
	def _assign_new_center(self) -> tuple: 
		self._verify_string_variable_setting('center_mode',self.center_mode,center_modes)

		if self.center_mode == 'anywhere': 
			if 0 not in self._image: return None 
			new_center = self._RNG.choice(list(zip(*np.where(self._image==0))))

		elif self.center_mode == 'central': 
			x_min = int(self.nucleation_region_percent * self.width)
			x_max = int((1-self.nucleation_region_percent) * self.width)
			y_min = int(self.nucleation_region_percent * self.height)
			y_max = int((1-self.nucleation_region_percent) * self.height)
			if 0 not in self._image[y_min:y_max,x_min:x_max]: return None 
			new_center = map(sum,zip(self._RNG.choice(list(zip(*np.where(self._image[y_min:y_max,x_min:x_max]==0)))),(y_min,x_min)))
				
		return tuple(new_center)

	# This produces an appropriate orientation (in revolutions).
	def _assign_new_orientation(self) -> float: 
		self._verify_string_variable_setting('orientation_mode',self.orientation_mode,orientation_modes)

		if self.orientation_mode == 'random': 
			orientation_modifier = self._RNG.random()
		elif self.orientation_mode == 'updown': 
			orientation_modifier = self._RNG.choice([0,1/(2*self.crystal_sides)])

		new_orientation = round((self._primary_orientation + orientation_modifier)%1,self.orientation_precision)
		return new_orientation

	def _nucleate_new_crystal(self, center: tuple, orientation: float): 

		if (center[0] > self.height) or (center[1] > self.width):
			warnings.warn(f'A new crystal at {center} would have nucleated outside the image bounds.')
			
			return 
		if self._image[center]: 
			warnings.warn(f'A new crystal at {center} would have nucleated inside existing crystal {self._image[center]}')
			return 

		else: 
			self._current_crystals += 1 
			self._image[center] = self._current_crystals
			gd, gv = self._generate_growth_vectors(orientation)
			new_crystal = _Crystal(center=center,
								  orientation=orientation,
								  sides = self.crystal_sides,
								  growth_directions=gd,
								  growth_vectors=gv,
								  nucleation_time = self._current_time,
								  areas=list([0]*(self._current_time)+[1]),
								  radii=list([0.]*(self._current_time)+[0.5]),
								  candidate_points = dict())

			self._crystals.append(new_crystal)
			self._total_crystal_area += 1

			for point in self._get_adjacent_points((0,0),self._current_crystals):
				self._crystals[self._current_crystals].candidate_points[point] = self._calculate_minimum_size(point,self._current_crystals)

	# Produces a list of primary growth directions and a list of tuple "unit vectors"
	def _generate_growth_vectors(self, orientation: float = 0) -> list:
		# Presumes a regular polygon
		if self.shape_array is None: 
			shape_array = np.empty((self.crystal_sides,2))
			shape_array[:,0] = (np.array(range(self.crystal_sides))/self.crystal_sides)
			shape_array[:,1] = 1
		else: shape_array = self.shape_array.copy()
		shape_array[:,0] = (shape_array[:,0] + orientation)%1
		shape_array = shape_array[shape_array[:,0].argsort()]
		return (shape_array[:,0],[(shape_array[i,1]*np.sin(shape_array[i,0]*2*np.pi),shape_array[i,1]*np.cos(shape_array[i,0]*2*np.pi)) for i in range(shape_array.shape[0])])

	def _get_adjacent_points(self, relative_point: tuple, crystal: int) -> list:
		(y,x) = relative_point
		adjacent_points = [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
		unoccupied_points = []
		for (ay,ax) in adjacent_points:

			(ty,tx) = self._get_absolute_point((ay,ax),crystal)
			if self._image[ty,tx]: continue
			unoccupied_points.append((ay,ax))

		return unoccupied_points

	def _calculate_minimum_size(self, relative_point: tuple, crystal: int) -> float: 
		if relative_point in self._crystals[crystal].candidate_points: 
			return self._crystals[crystal].candidate_points[relative_point]
		
		(y,x) = relative_point
		angle_to_point = np.arctan2(y,x)%(2*np.pi)/(2*np.pi)
		orientation_after = bisect_left(self._crystals[crystal].growth_directions,angle_to_point)%self._crystals[crystal].sides
		orientation_before = (orientation_after-1)%self._crystals[crystal].sides

		projection_after = np.dot(self._crystals[crystal].growth_vectors[orientation_after],relative_point)
		projection_before = np.dot(self._crystals[crystal].growth_vectors[orientation_before],relative_point)
		
		return max(projection_before,projection_after)


	def _get_growth_rate(self, crystal: int) -> float: 
		self._verify_string_variable_setting('growth_rate_mode',self.growth_rate_mode,growth_rate_modes)

		if len(self._crystals[crystal].candidate_points) == 0: return 0

		if self.growth_rate_mode == 'constant': 
			growth_rate = self.growth_rate
		if self.growth_rate_mode == 'physical': 
			growth_rate = self.growth_rate/self._crystals[crystal].radii[-1]
		# elif self.growth_rate_mode == 'physical': return self.growth_rate/len(self._crystals[crystal].candidate_points)
		
		return growth_rate

	def _check_candidate_point(self, candidate_point: tuple, size: float, crystal: int):
		if size < self._crystals[crystal].radii[-1]:
			(ay,ax) = self._get_absolute_point(candidate_point,crystal)
			if self._image[(ay,ax)]: 
				del self._crystals[crystal].candidate_points[candidate_point]
				return
			self._image[(ay,ax)] = crystal
			self._total_crystal_area += 1
			self._crystals[crystal].areas[-1] += 1
			del self._crystals[crystal].candidate_points[candidate_point]
			for point in self._get_adjacent_points(candidate_point,crystal): 
				self._crystals[crystal].candidate_points[point] = self._calculate_minimum_size(point,crystal)

	# Support functions
	def _get_absolute_point(self, relative_point: tuple, crystal: int) -> tuple:
		(y,x) = relative_point
		(y_center,x_center) = self._crystals[crystal].center
		if self.periodic_boundary:
			ty = (y+y_center)%self.height
			tx = (x+x_center)%self.width
		else:
			ty = max(0,y+y_center) if (y+y_center) < 0 else min(self.height-1,y+y_center)
			tx = max(0,x+x_center) if (x+x_center) < 0 else min(self.width-1,x+x_center)
		return (ty,tx)

	def _get_area_coverage(self) -> float: 
		return (self._total_crystal_area/(self.height*self.width))

	# Verifies user entries to various string parameters, 
	#  and reconfigures the simulator to a default setting if necessary
	def _verify_string_variable_setting(self, variable: str, value: str, options_list: list):
		if value not in options_list: 
			raise ValueError(f'\'{value}\' is not a recognized option for {variable} (options: {[option for option in options_list]}).')

	# A periodic 1D implementation of complete-linkage clustering 
	# Assumes an ordered list of angles  
	def _iteratively_clink_group(self, group: list, maximum_misorientation: float) -> list:
		""" 
		Performs farthest neighbor / complete-linkage clustering on a group of numbers.

		Parameters
		----------
		group : list of num
			the numbers to be clustered, ordered from least to greatest and known to not have a spacing larger than `maximum_misorientation` between adjacent elements
		maximum_misorientation : float 
			the largest permitted difference between two numbers in the same group

		Returns 
		-------
		clinked_groups : list of list of num 
			list of lists of clustered angles

		Notes
		-----
		This algorithm loosely follows from: https://en.wikipedia.org/wiki/Complete-linkage_clustering 
		Modifications were made due to the 1-D nature of the problem. 
		
		(1) assign each number to a group by itself 
		(2) calculate the differences between adjacent numbers 
		(3) link the two closest numbers together
		(4) calculate the farthest neighbor difference (maximum of next - minimum of current) between adjacent groups
		(4a) if the smallest of these differences exceeds maximum_misorientation, stop
		(4b) otherwise, merge the two groups with the smallest differnece and go to (3)
		 
		""" 
		group_diffs = np.diff(group)
		clinked_groups = list([[e] for e in group])
		finished_clinking = False

		while not finished_clinking: 
			if len(clinked_groups) == 1: break
			# find most similar pair  
			merge_index = np.argmin(group_diffs)
			
			# combine the two groups 
			clinked_groups[merge_index].extend(clinked_groups[merge_index+1])
			del clinked_groups[merge_index+1]
			
			# compare the minimum of each group with the maximum from the next group 
			group_diffs = [clinked_groups[i+1][-1] - clinked_groups[i][0] for i in range(0,len(clinked_groups)-1)]

			# check if no possible merges
			if np.min(group_diffs) > maximum_misorientation: finished_clinking = True

		return clinked_groups


def get_shape_array(points: list) -> np.array: 
	""" 
	Produces a Simulator-compatible "shape array" from a polygon (defined as a list of vertices).

	Parameters
	----------
	points : list of tuple of num
		the vertices of a polygon, expressed as (y,x) coordinates

	Returns 
	-------
	shape_array : numpy.array 
		an array describing the orientations (in revolutions) and normalized distances of each side of the polygon

	Notes
	-----
	This function only works for convex polygons. 

	"""
	looped_points = list(points).copy()
	looped_points.append(points[0])
	edge_vectors = [(np.subtract(looped_points[i+1],looped_points[i])) for i in range(len(points))] 
	edge_vectors.append(edge_vectors[0]) # to allow for looping around the first corner 

	vector_lengths = np.linalg.norm(edge_vectors,axis=1)

	if np.any(vector_lengths == 0): 
		raise ValueError('One of the sides of the specified polygon has a length of 0.')

	unit_vectors = edge_vectors/vector_lengths.reshape(-1,1) 

	interior_angles = np.array([np.arccos(np.dot(unit_vectors[i+1,:],-unit_vectors[i,:]))/(2*np.pi) for i in range(len(points))])
	central_angles = 1/2-interior_angles
	if np.sum(central_angles) > 1: 
		raise ValueError('The polygon specified by these input points is not convex.')

	cumulative_angles = np.cumsum(central_angles)-central_angles[0]

	center = np.mean(np.array(points),axis=0)

	distances = [np.linalg.norm((np.cross(edge_vectors[i],looped_points[i]-center)))/np.linalg.norm(edge_vectors[i]) for i in range(len(points))] 
	coeffs_polygon = distances/np.max(distances)

	# construct shape array
	shape_array = np.empty((len(points),2))
	shape_array[:,0] = cumulative_angles
	shape_array[:,1] = coeffs_polygon
	return shape_array

@dataclass
class _Crystal: 
	""" 
	A dataclass that stores all information of a given crystal. 

	Attributes
	----------
	center : tuple of int 
		the geometric center of the crystal in the form (y,x)
	orientation : float
		an angle in revolutions (clockwise, from positive x direction) 
	sides : int
		the number of sides of the crystal 
	growth_directions : list of float  
		an ordered list of the growth directions of the crystal
	growth_vectors : list of tuple of float
		an list of "unit" growth vectors of the crystal, corresponding to each growth_direction
	nucleation_time : int
		the time step at which the crystal nucleated
	radii : list of float 
		the radius (in pixels) of the crystal at each time step 
	areas : list of int 
		the area (in pixels) of the crystal at each time step 
	candidate_points : dict of tuple to float 
		a mapping of relative points to a calculated minimum size to be added to the crystal
	"""
	center: tuple 			
	orientation: float 		
	sides: int 				
	growth_directions: list 
	growth_vectors: list 	
	nucleation_time: int 	
	radii: list 			
	areas: list 			
	candidate_points: dict