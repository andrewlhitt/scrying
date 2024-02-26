import numpy as np 
from dataclasses import dataclass
from bisect import bisect_left

class Simulator: 
	__arg_list = ['width','height','crystal_sides','shape_array','maximum_time',
					'nucleation_rate','nucleation_rate_mode','maximum_crystals','center_mode','orientation_mode','nucleation_region_percent',
					'growth_rate','growth_rate_mode','periodic_boundary',
					'snapshot_mode','snapshot_time','snapshot_coverage','end_after_snapshot',
					'save_evolution', 'random_seed']

	__nucleation_rate_modes = ['constant']
	__orientation_modes = ['random','updown']
	__center_modes = ['anywhere','central']
	__growth_rate_modes = ['constant','physical']
	__snapshot_modes = ['none','time','area']
	__get_image_options = ['final','snapshot','time']
	__import_nucleation_data_position_orders = ['xy','yx']
	__angle_units = ['degrees','radians','revolutions']
	__angle_chiralities = ['cw','ccw']
	__zero_directions = ['right','left','up', 'down']
	__import_nucleation_data_final_column_types = ['time','size']

	def __init__(self, 
		width: int = 128, height: int = 128, crystal_sides: int = 3, shape_array: np.array = None, maximum_time: int = 100, 
		nucleation_rate: float = 1.0, nucleation_rate_mode: str = 'constant', maximum_crystals: int = 5, center_mode: str = 'anywhere', orientation_mode: str = 'random', nucleation_region_percent: float = 0.0,
		growth_rate: float = 1.0, growth_rate_mode: str = 'constant', periodic_boundary: bool = False, 
		snapshot_mode: str = 'none', snapshot_time: int = 0, snapshot_coverage: float = 0.25, end_after_snapshot: bool = False, stop_nucleation_after_snapshot: bool = False,
		save_evolution: bool = False, random_seed: int = None): 

		# Initializing the random number generator
		self._RNG = np.random.default_rng(random_seed)

		# default parameters
		self.width = width 
		self.height = height 
		self.crystal_sides = crystal_sides
		self.shape_array = shape_array 
		if self.shape_array is not None: 
			self.crystal_sides = self.shape_array.shape[0] 
		# else: self.shape_array = self._generate_shape_array()# np.array() # TODO: replace this with a generated one
		self.maximum_time = maximum_time 
		
		# nucleation parameters
		self.nucleation_rate = nucleation_rate
		self.nucleation_rate_mode = nucleation_rate_mode.lower() 
		self.maximum_crystals = maximum_crystals
		self.center_mode = center_mode.lower()
		self.orientation_mode = orientation_mode.lower()
		self.nucleation_region_percent = nucleation_region_percent

		# growth parameters
		self.growth_rate = growth_rate
		self.growth_rate_mode = growth_rate_mode.lower()
		self.periodic_boundary = periodic_boundary 

		# snapshot parameters 
		self.snapshot_mode = snapshot_mode.lower()
		self.snapshot_time = snapshot_time
		self.snapshot_coverage = snapshot_coverage
		self.end_after_snapshot = end_after_snapshot
		self.stop_nucleation_after_snapshot = stop_nucleation_after_snapshot

		# data parameters
		self.save_evolution = save_evolution 
		self._imported_data = None 

	def change_simulator_settings(self, **kwargs): 
		for key in kwargs: 
			if key in self.__arg_list: 
				self.__setattr__(key,kwargs.get(key))
			else: 
				print(f'WARNING: Unrecognized keyword argument \'{key}\' with value {kwargs.get(key)}')
		
	def run_simulation(self, use_imported_data: bool = False): 

		if use_imported_data and (self._imported_data is None): 
			print(f'WARNING: No nucleation data has been imported. Use the import_nucleation_data(...) function before calling the simulation.')
			return 

		if use_imported_data and (self._imported_data[self._imported_data[:,3] >= self.maximum_time].shape[0] > 0): 
			print(f'WARNING: Simulation will conclude before the nucleation of all crystals in the imported data. Consider increasing maximum_time to at least {int(np.max([self._imported_data[:,3]]))}') 

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
					print('WARNING: Snapshot criteria were not met during simulation.')
				break 
			if (self._snapshot_taken and self._current_crystals == 0 and self.stop_nucleation_after_snapshot): 
				print('WARNING: Snapshot was taken with no crystals present , preventing the simulation from proceeding.')
				break
			if (self._snapshot_taken and self.end_after_snapshot): break				

			self._current_time += 1 
		else: 
			print('WARNING: Simulation concluded before reaching 100% area coverage. Consider increasing maximum_time.')

		self._conclude_simulation()

	# Main loop functions 

	def _reset_simulation_parameters(self):
		self._current_time = 0 
		self._image = np.zeros((self.height,self.width),dtype=int)
		self._crystals = list([None]) # this allows the list index to match the image label
		self._current_crystals = 0 
		self._primary_orientation = self._RNG.random()
		self._orientations = set()
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
		# if self._current_time in self._imported_data[:,3]:
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
		self._verify_string_variable_setting('snapshot_mode',self.snapshot_mode,self.__snapshot_modes,'none')

		if self.snapshot_mode == 'none': return 
		if self._snapshot_taken: return 
		if (self.snapshot_mode == 'area' and self._get_area_coverage() >= self.snapshot_coverage) or (self.snapshot_mode == 'time' and self._current_time >= self.snapshot_time):
		   self._snapshot_taken = True
		   self.snapshot_taken_at_time = self._current_time
		   self._snapshot = self._image.copy()

	def _conclude_simulation(self):
		if self.save_evolution:
			self._image_evolution_array = np.rollaxis(np.dstack(self._image_over_time),-1)
		
	# Main loop nested functions  		
	def _get_current_nucleation_rate(self) -> float: 
		self._verify_string_variable_setting('nucleation_rate_mode',self.nucleation_rate_mode,self.__nucleation_rate_modes,'constant')

		if self.nucleation_rate_mode == 'constant': return self.nucleation_rate
		else: return self.nucleation_rate

	# This produces an appropriate center for a new crystal (y,x)
	# It will return None if there is no available pixel for a center.
	def _assign_new_center(self) -> tuple: 
		self._verify_string_variable_setting('center_mode',self.center_mode,self.__center_modes,'anywhere')

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
		self._verify_string_variable_setting('orientation_mode',self.orientation_mode,self.__orientation_modes,'random')

		if self.orientation_mode == 'random': 
			orientation_modifier = self._RNG.random()
		elif self.orientation_mode == 'updown': 
			orientation_modifier = self._RNG.choice([0,1/(2*self.crystal_sides)])

		new_orientation = (self._primary_orientation + orientation_modifier)%1
		self._orientations.add(new_orientation)
		return new_orientation

	def _nucleate_new_crystal(self, center: tuple, orientation: float): 

		if (center[0] > self.height) or (center[1] > self.width):
			print(f'WARNING: A new crystal at {center} would have nucleated outside the image bounds.')
			
			return 
		if self._image[center]: 
			print(f'WARNING: A new crystal at {center} would have nucleated inside existing crystal {self._image[center]}')
			return 

		else: 
			self._current_crystals += 1 
			self._image[center] = self._current_crystals
			gd, gv = self._generate_growth_vectors(orientation)
			new_crystal = Crystal(center=center,
								  orientation=orientation,
								  sides = self.crystal_sides,
								  growth_directions=gd,
								  growth_vectors=gv,
								  nucleation_time = self._current_time,
								  areas=list([0]*(self._current_time)+[1]),
								  radii=list([0.]*(self._current_time)+[0.5]),
								  candidate_points = dict())

			self._crystals.append(new_crystal)
			self._orientations.add(orientation)	
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
		self._verify_string_variable_setting('growth_rate_mode',self.growth_rate_mode,self.__growth_rate_modes,'constant')

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
	def _verify_string_variable_setting(self, variable: str, value: str, options_list: list, default_setting: str):
		if value not in options_list: 
			print(f'WARNING: \'{value}\' is not a recognized option for {variable} (options: {[option for option in options_list]}). Simulator has been set to default mode (\'{default_setting}\').')
			setattr(self,variable,default_setting)

	def import_nucleation_data(self, nucleation_data: np.array, 
							   position_order: str = 'xy',
							   angle_unit: str = 'revolutions', chirality: str = 'cw', zero_direction = 'right', 
							   final_column: str = 'time', autoconfigure_snapshot: bool = False,
							   has_header_row: bool = False, has_index_column: bool = True): 

		self._verify_string_variable_setting('position_order',position_order,self.__import_nucleation_data_position_orders,'xy')
		self._verify_string_variable_setting('angle_unit',angle_unit,self.__angle_units,'revolutions')
		self._verify_string_variable_setting('chirality',chirality,self.__angle_chiralities,'cw')
		self._verify_string_variable_setting('zero_direction',zero_direction,self.__zero_directions,'right')
		self._verify_string_variable_setting('final_column',final_column,self.__import_nucleation_data_final_column_types,'time')

		if has_header_row: first_row = 1 
		else: first_row = 0

		if has_index_column: first_column = 1
		else: first_column = 0

		data = nucleation_data[first_row:,first_column:]
		

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

		data[:,2] = angle_addend + (angle_multiplier*data[:,2]/full_rotation)
		
		# if data reports crystal size, calculates an approximate nucleation time for each crystal
		# this assumes globally uniform growth rate 
		if final_column == 'size': 
			final_time = np.max(data[:,3])
			data[:,3] = np.round((final_time - data[:,3])).astype('int')

			# if autoconfigure is enabled, sets up the snapshot to capture the approximate image 
			# that would have been used for the data measurement 
			if autoconfigure_snapshot: 
				self.snapshot_time = final_time 
				self.snapshot_mode = 'time'

		self._imported_data = data 



	# Data/Analysis functions
	def export_nucleation_data(self,position_order: str = 'xy',
							   angle_unit: str = 'revolutions', chirality: str = 'cw', zero_direction: str = 'right') -> np.array:

		self._verify_string_variable_setting('position_order',position_order,self.__import_nucleation_data_position_orders,'xy')
		self._verify_string_variable_setting('angle_unit',angle_unit,self.__angle_units,'degrees')
		self._verify_string_variable_setting('chirality',chirality,self.__angle_chiralities,'ccw')
		self._verify_string_variable_setting('zero_direction',zero_direction,self.__zero_directions,'right')

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
		if mode not in self.__get_image_options:
			print(f'WARNING: \'{mode}\' is not a recognized option for get_image. Accepted options are: {[option for option in self.__get_image_options]}')
		
		image = None

		if (mode == 'time') or (time is not None): 
			if self.save_evolution: image = self._image_evolution_array[time,:,:]
			else: print(f'WARNING: Time-series data has not been saved. Image at time {time} cannot be retrieved.') 
		elif mode == 'final': 
			image = self._image
		elif mode == 'snapshot': 
			image = self._snapshot
		return image

	def get_image_evolution(self) -> np.array: 
		if self.save_evolution: return self._image_evolution_array
		else: 
			print(f'WARNING: Time-series data has not been saved.')
			return None

	def get_grain_structure(self, maximum_misorientation: float = 0, symmetry: int = 1):
		# in progress (as of Feb 26, 2024) 
		pass 
		


@dataclass
class Crystal: 
	center: tuple 			# (y,x) in pixels
	orientation: float 		# clockwise angle in revolutions from positive X
	sides: int 				# number of sides 
	growth_directions: list # list(float) list of growth directions in revolutions (ordered from least to greatest)
	growth_vectors: list 	# list(tuple) list of "unit" vector growth directions in form (y, x) 
	nucleation_time: int 	# time of nucleation 
	radii: list 			# list(float) the radius (in pixels) of the crystal at each time step (used in size calculations)
	areas: list 			# list(int): the area (in pixels) of the crystal at each time step
	candidate_points: dict 	# dict(tuple->float) dictionary of relative points to calculated size 


