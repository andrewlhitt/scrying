import scrying 
import tkinter as tk 
from tkinter import ttk 
import ttkwidgets as ttkw
import numpy as np 
from matplotlib import pyplot as plt 
import os
import tifffile as tif 
import array2gif
import pickle 
import warnings 

class scryUI: 

	__initial_size = 128 
	__initial_maximum_time = 100
	__initial_crystal_sides = 3 
	__initial_orientation_mode = 'updown'
	__initial_periodic_boundary = True
	__image_size = 128 

	def __init__(self, root):
		self.root = root 
		self.simulator = scrying.Simulator(save_evolution=True)
		self.imported_shape = None 
		self.imported_nucleation_data = None 
		self.crystal_sides = self.__initial_crystal_sides 
		self.crystal_symmetry = self.__initial_crystal_sides 
		self.shape_array = None
		self.blank_image = self._photo_image(np.zeros((self.__image_size,self.__image_size), dtype=np.uint8))
		self.blank_image_small = self._photo_image(np.zeros((64,64), dtype=np.uint8))
		
		self._setup_ui()

	def _setup_ui(self):
		self.root.title('SCRYiNG: A 2D Crystal Growth Simulator')
		self.root.geometry('450x550')
		self.root.resizable(False,False)
		self.root.columnconfigure(0, weight=1)
		self.root.rowconfigure(0, weight=1)

		self.menu = tk.Menu(self.root)
		self.root.config(menu=self.menu)
		self.import_menu = tk.Menu(self.menu)

		self.import_menu.add_command(label='... Nucleation Data (.csv)',command = self._import_nucleation_data)
		self.import_menu.add_command(label='... Shape Data (.csv)',command = self._import_shape_data)
		self.import_menu.add_separator()
		self.import_menu.add_command(label='... Config File (.scrying)',command = self._import_settings)
		self.menu.add_cascade(label='Import',menu=self.import_menu)

		self.export_menu = tk.Menu(self.menu)
		self.menu.add_cascade(label='Export',menu=self.export_menu)
		self.export_menu.add_command(label='... Config File (.scrying)',command = self._export_settings)

		# General Parameters
		# - Maximum Simulation Time 
		# - Image Size 

		self.frame_general = tk.Frame(self.root,relief='groove',bd=2)
		self.frame_general.grid(row=0,column=0,sticky='nsew')

		ttk.Label(self.frame_general,text='Maximum Time').grid(row=0,column=0)
		self.tickscale_maximum_time = ttkw.TickScale(self.frame_general,resolution=10,length=125,from_=10,to=500,labelpos='e')#,command = self._update_maximum_time_callback)
		self.tickscale_maximum_time.grid(row=0,column=1)
		self.tickscale_maximum_time.set(self.__initial_maximum_time)

		ttk.Label(self.frame_general,text='Image Size').grid(row=1,column=0)
		self.tickscale_image_size = ttkw.TickScale(self.frame_general,length=125,resolution=16,from_=32,to=256,labelpos='e')
		self.tickscale_image_size.grid(row=1,column=1)
		self.tickscale_image_size.set(self.__initial_size)

		
		ttk.Label(self.frame_general,text='Random Seed').grid(row=2,column=0)
		self.scaleentry_random_seed = ttkw.ScaleEntry(self.frame_general,scalewidth=115,from_=0,to=1024)
		#self.random_seed_str.trace('w',self._update_random_seed_callback)
		self.scaleentry_random_seed.grid(row=2,column=1)

		self.use_random_seed = tk.IntVar()
		ttk.Label(self.frame_general,text='Use Random Seed?').grid(row=3,column=0)
		self.checkbutton_use_random_seed = tk.Checkbutton(self.frame_general,variable=self.use_random_seed)
		self.checkbutton_use_random_seed.grid(row=3,column=1,sticky='w')

		# Nucleation Parameters
		# - Nucleation Rate 
		# - Maximum Crystals
		# - Orientation Mode
		# - Center Placement 
		# - > Nucleation Region Bound  
		self.frame_nucleation = tk.Frame(self.root,relief='groove',bd=2)
		self.frame_nucleation.grid(row=1,column=0,sticky='nsew')


		ttk.Label(self.frame_nucleation,text='Nucleation Rate').grid(row=0,column=0)
		self.tickscale_nucleation_rate = ttkw.TickScale(self.frame_nucleation,resolution=0.1,from_=0.1,to=5.0,labelpos='e')
		self.tickscale_nucleation_rate.grid(row=0,column=1)
		self.tickscale_nucleation_rate.set(0.3)

		ttk.Label(self.frame_nucleation,text='Maximum Crystals').grid(row=1,column=0)
		self.tickscale_max_crystals = ttkw.TickScale(self.frame_nucleation,resolution=1,from_=1,to=40,labelpos='e')
		self.tickscale_max_crystals.grid(row=1,column=1)
		self.tickscale_max_crystals.set(5)

		ttk.Label(self.frame_nucleation,text='New Crystal Orientation').grid(row=2,column=0)
		self.orientation_mode = tk.StringVar()
		self.orientation_mode.set('random')
		self.orientation_mode_menu = ttk.OptionMenu(self.frame_nucleation,self.orientation_mode,'random',*scrying.orientation_modes)
		self.orientation_mode_menu.grid(row=2,column=1)

		ttk.Label(self.frame_nucleation,text='New Crystal Center').grid(row=3,column=0)
		self.center_mode = tk.StringVar()
		self.center_mode.set('anywhere')
		self.center_mode_menu = ttk.OptionMenu(self.frame_nucleation,self.center_mode,'anywhere',*scrying.center_modes)
		self.center_mode_menu.grid(row=3,column=1)

		self.use_nucleation_data = tk.IntVar()
		ttk.Label(self.frame_nucleation,text='Use Nucleation Data?').grid(row=4,column=0)
		self.checkbutton_use_nucleation_data = tk.Checkbutton(self.frame_nucleation,variable=self.use_nucleation_data)
		self.checkbutton_use_nucleation_data.grid(row=4,column=1,sticky='w')


		# Growth Parameters
		# - Growth Rate <can skip> 
		# - Growth Rate Mode <can skip> 
		# - Periodic Boundary? 
		# - Crystal Shape: Triangle, Square, Hexagon, Custom 
		self.frame_growth = tk.Frame(self.root,relief='groove',bd=2)
		self.frame_growth.grid(row=2,column=0,sticky='nsew')

		ttk.Label(self.frame_growth,text='Growth Rate').grid(row=0,column=0)
		self.tickscale_growth_rate = ttkw.TickScale(self.frame_growth,resolution=0.1,from_=0.1,to=5.0,labelpos='e')
		self.tickscale_growth_rate.grid(row=0,column=1)
		self.tickscale_growth_rate.set(1)

		self.periodic_boundary = tk.IntVar()
		ttk.Label(self.frame_growth,text='Periodic Boundary?').grid(row=1,column=0)
		self.checkbutton_periodic_boundary = tk.Checkbutton(self.frame_growth,variable=self.periodic_boundary)
		self.checkbutton_periodic_boundary.grid(row=1,column=1,sticky='w')

		ttk.Label(self.frame_growth,text='Crystal Shape').grid(row=2,column=0)
		self.crystal_shape = tk.StringVar()
		self.crystal_shape.set('triangle')
		# self.crystal_shape.trace('w',)
		
		self.crystal_shape_menu = ttk.OptionMenu(self.frame_growth,self.crystal_shape,'triangle',*['triangle','square','hexagon','custom'], command = self._update_crystal_shape_callback)
		self.crystal_shape_menu.grid(row=2,column=1)


		# tk.Label(self.frame_growth,text='Early Stage Crystals').grid(row=0,column=2)
		self.shape_preview=tk.Label(self.frame_growth,image=self.blank_image_small) #,text='Early Stage',compound='center')
		self.shape_preview.image = self.blank_image_small
		self.shape_preview.grid(row=0,column=2,rowspan=3,sticky='e')

		# Save Parameters
		# - Snapshot Mode
		# - Snapshot Parameter (Time or Coverage) 
		# - End Sim After Snapshot?
		# - Stop Nucleation After Snapshot? 
		# - Save Time-Series: None/.tif/.gif 
		# - Directory 
		self.frame_snapshot = tk.Frame(self.root,relief='groove',bd=2)
		self.frame_snapshot.grid(row=3,column=0,sticky='nsew')

		ttk.Label(self.frame_snapshot,text='Snapshot Mode').grid(row=0,column=0)
		self.snapshot_mode = tk.StringVar()
		self.snapshot_mode.set('area')
		self.snapshot_mode_menu = ttk.OptionMenu(self.frame_snapshot,self.snapshot_mode,'area',*scrying.snapshot_modes, command = self._update_snapshot_mode_callback)
		self.snapshot_mode_menu.grid(row=0,column=1)

		ttk.Label(self.frame_snapshot,text='Snapshot Parameter').grid(row=1,column=0)
		self.tickscale_snapshot_parameter = ttkw.TickScale(self.frame_snapshot,resolution=0.01,from_=0.1,to=1.0,labelpos='e')
		self.tickscale_snapshot_parameter.grid(row=1,column=1)
		self.tickscale_snapshot_parameter.set(0.25)


		ttk.Label(self.frame_snapshot,text='After the Snapshot...').grid(row=2,column=0)
		self.after_snapshot = tk.StringVar()
		self.after_snapshot.set('disable nucleation')
		self.simulator.change_settings(end_after_snapshot=False)
		self.simulator.change_settings(stop_nucleation_after_snapshot=True)
		self.after_snapshot_menu = ttk.OptionMenu(self.frame_snapshot,self.after_snapshot,'disable nucleation',*['continue','disable nucleation','stop'], command = self._update_after_snapshot_callback)
		# self.after_snapshot.set('disable nucleation')
		self.after_snapshot_menu.grid(row=2,column=1)

		ttk.Label(self.frame_snapshot,text='Directory').grid(row=3,column=0)
		self.entry_directory = ttk.Entry(self.frame_snapshot)
		self.entry_directory.grid(row=3,column=1)

		ttk.Label(self.frame_snapshot,text='Save Evolution').grid(row=4,column=0)
		self.save_evolution = tk.StringVar()
		self.save_evolution.set('none')
		self.save_evolution_menu = ttk.OptionMenu(self.frame_snapshot,self.save_evolution,'none',*['none','.gif','.tif'])
		self.save_evolution_menu.grid(row=4,column=1)

		ttk.Label(self.frame_snapshot,text='Number of Simulations').grid(row=5,column=0)
		self.tickscale_number_simulations = ttkw.TickScale(self.frame_snapshot,resolution=1,from_=1,to=100,labelpos='e')
		self.tickscale_number_simulations.grid(row=5,column=1)
		self.tickscale_number_simulations.set(1)

		ttk.Label(self.frame_snapshot,text='Max. Misorientation').grid(row=6,column=0)
		self.tickscale_misorientation = ttkw.TickScale(self.frame_snapshot,resolution=5,from_=0,to=120,labelpos='e')
		self.tickscale_misorientation.grid(row=6,column=1)
		self.tickscale_misorientation.set(0)

		# Previews 
		# Shape 
		# Early Stage (Snapshot)
		# Final Stage (Film)
		self.frame_preview = tk.Frame(self.root,relief='groove',bd=2,width=130)
		self.frame_preview.grid(row=0,column=1,rowspan=4,sticky='nsew')
		self.button_preview_sim = ttk.Button(self.frame_preview, text="Preview Simulation", command = self._button_preview_simulation)
		self.button_preview_sim.grid(row=0)

		tk.Label(self.frame_preview,text='Early Image ("Snapshot")').grid(row=1)
		self.snapshot=tk.Label(self.frame_preview,image=self.blank_image) #,text='Early Stage',compound='center')
		self.snapshot.image = self.blank_image
		self.snapshot.grid(row=2)

		tk.Label(self.frame_preview,text='Final Image').grid(row=3)
		self.final_image =tk.Label(self.frame_preview,image=self.blank_image) #,text='Final Grain Structure',compound='center')
		self.final_image.image = self.blank_image
		self.final_image.grid(row=4)

		tk.Label(self.frame_preview,text='Crystal Structure').grid(row=5)
		self.crystal_structure =tk.Label(self.frame_preview,image=self.blank_image) #,text='Final Grain Structure',compound='center')
		self.crystal_structure.image = self.blank_image
		self.crystal_structure.grid(row=6)


		self.button_run_sim = ttk.Button(self.frame_preview, text="Run Simulation", command = self._button_run_simulations)
		self.button_run_sim.grid(row=7)


	def _import_nucleation_data(self):
		filename = tk.filedialog.askopenfilename(filetypes=(('CSV','*.csv'),))
		try: 
			data = np.genfromtxt(filename,delimiter=',')
			self.imported_nucleation_data = data 
			self.has_imported_data = True
			self.simulator.import_nucleation_data(data,angle_unit='degrees',chirality='ccw',final_column='size',autoconfigure_snapshot=True,has_header_row=True,has_index_column=False)

		except: 
			print('WARNING: Nucleation data importing failed.')
			self.has_imported_data = False



	def _import_shape_data(self):
		filename = tk.filedialog.askopenfilename(filetypes=(('CSV','*.csv'),))
		try: 
			data = np.genfromtxt(filename,delimiter=',')[1:]
			# additional parsing into appropriate format 
			self.imported_shape = scrying.get_shape_array(data)
			# to-do: add configuration settings for imported data 
			# self.simulator.import_nucleation_data(data,angle_unit='radians',chirality='ccw',final_column='time',autoconfigure_snapshot=True,has_header_row=True)

		except: 
			print('WARNING: Shape data importing failed.')
			self.imported_shape = None
			# self.imported_start_time = 0


	def _photo_image(self,image: np.ndarray):
		# converts array to PhotoImage object for display purposes
		# normalizes numpy array to maximum value (for visibility)
		# credit: Adrian W. 
		# https://stackoverflow.com/questions/53308708/how-to-display-an-image-from-a-numpy-array-in-tkinter
		height, width = image.shape

		# normalize image 
		# image -= image.min()
		if image.max(): image *= int(255.0/image.max())

		data = f'P5 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
		return tk.PhotoImage(width=width, height=height, data=data, format='PPM')

	def _resizeImage(self,img, newWidth, newHeight):
		# rescale an image using tkinter functions 
		# credit: garydavenport73 (stackoverflow)
		# https://stackoverflow.com/questions/3177969/how-to-resize-an-image-using-tkinter
		oldWidth = img.width()
		oldHeight = img.height()
		newPhotoImage = tk.PhotoImage(width=newWidth, height=newHeight)
		for x in range(newWidth):
			for y in range(newHeight):
				xOld = int(x*oldWidth/newWidth)
				yOld = int(y*oldHeight/newHeight)
				rgb = '#%02x%02x%02x' % img.get(xOld, yOld)
				newPhotoImage.put(rgb, (x, y))
		return newPhotoImage

	def _update_crystal_shape_callback(self,*args):
		# print(f'crystal shape is updated to {self.crystal_shape.get()}')

		if self.crystal_shape.get() == 'triangle':
			self.crystal_sides = 3 
			self.crystal_symmetry = 3 
			self.tickscale_misorientation.configure(resolution=5,from_=0,to=120)
			self.tickscale_snapshot_parameter.set(0)
			self.shape_array = None 
		elif self.crystal_shape.get() == 'square':
			self.crystal_sides = 4
			self.crystal_symmetry = 4
			self.tickscale_misorientation.configure(resolution=5,from_=0,to=90)
			self.tickscale_snapshot_parameter.set(0)
			self.shape_array = None 
		elif self.crystal_shape.get() == 'hexagon':
			self.crystal_sides = 6 
			self.crystal_symmetry = 6
			self.tickscale_misorientation.configure(resolution=5,from_=0,to=60)
			self.tickscale_snapshot_parameter.set(0)
			self.shape_array = None 
		elif self.crystal_shape.get() == 'custom':
			if self.imported_shape is None: 
				print('WARNING: No shape array has been imported!')
				self.crystal_shape.set('triangle')
			else:
				self.crystal_sides = 3 # need to update 
				self.shape_array = self.imported_shape
				self.crystal_symmetry = 1 
				self.tickscale_misorientation.configure(resolution=5,from_=0,to=360)
				self.tickscale_snapshot_parameter.set(0)

	def _update_snapshot_mode_callback(self,*args):

		if self.snapshot_mode.get() == 'area':
			self.tickscale_snapshot_parameter.configure(resolution=0.01,from_=0.1,to=1.0,digits=2)
			self.tickscale_snapshot_parameter.set(0.25)
		elif self.snapshot_mode.get() == 'time':
			self.tickscale_snapshot_parameter.configure(resolution=1,from_=1,to=int(self.tickscale_maximum_time.get()),digits=0)
			self.tickscale_snapshot_parameter.set(25)
		elif self.crystal_shape.get() == 'none':
			pass

	def _update_after_snapshot_callback(self,*args):

		if self.after_snapshot.get() == 'continue':
			self.simulator.change_settings(end_after_snapshot=False)
			self.simulator.change_settings(stop_nucleation_after_snapshot=False)
		elif self.after_snapshot.get() == 'disable nucleation':
			self.simulator.change_settings(end_after_snapshot=False)
			self.simulator.change_settings(stop_nucleation_after_snapshot=True)
		elif self.after_snapshot.get() == 'stop':
			self.simulator.change_settings(end_after_snapshot=True)
			self.simulator.change_settings(stop_nucleation_after_snapshot=True)

	def _button_preview_simulation(self): 
		self._run_simulation(show_result = True, save_result = False)

	def _button_run_simulations(self): 
		self._run_simulation(show_result = True, save_result = True)

	def _run_simulation(self, show_result: bool = False, save_result: bool = False):

		# collect parameters 

		if self.use_random_seed.get():
			self.simulator.change_settings(random_seed = self.scaleentry_random_seed.value)
		else:
			self.simulator.change_settings(random_seed = None)


		self.simulator.change_settings(maximum_time = self.tickscale_maximum_time.get())
		self.simulator.change_settings(height = int(self.tickscale_image_size.get()))
		self.simulator.change_settings(width = int(self.tickscale_image_size.get()))
		self.simulator.change_settings(nucleation_rate = self.tickscale_nucleation_rate.get())
		self.simulator.change_settings(maximum_crystals = self.tickscale_max_crystals.get())
		self.simulator.change_settings(orientation_mode = self.orientation_mode.get())
		
		self.simulator.change_settings(growth_rate = self.tickscale_growth_rate.get())
		self.simulator.change_settings(periodic_boundary = self.periodic_boundary.get())
		
		self.simulator.change_settings(crystal_sides = self.crystal_sides)
		self.simulator.change_settings(shape_array = self.shape_array)
		
		snapshot_mode = self.snapshot_mode.get()
		self.simulator.change_settings(snapshot_mode = snapshot_mode)

		if snapshot_mode == 'time':
			self.simulator.change_settings(snapshot_time = self.tickscale_snapshot_parameter.get())
		if snapshot_mode == 'area':
			self.simulator.change_settings(snapshot_area = self.tickscale_snapshot_parameter.get())

		center_mode = self.center_mode.get()
		self.simulator.change_settings(center_mode=center_mode)
		
		if center_mode == 'central': self.simulator.change_settings(nucleation_region_percent=0.33)
		else: self.simulator.change_settings(nucleation_region_percent=0.)

		if self.use_nucleation_data.get(): 
			use_imported_data = True 
		else:
			use_imported_data = False

		if save_result == True:
			self._setup_directory()
			n = 0

			if self.use_nucleation_data.get(): 
				number_of_simulations = 1 
			else:
				number_of_simulations = self.tickscale_number_simulations.get()

			while n < number_of_simulations:
				self.simulator.run_simulation(use_imported_data=use_imported_data)
				plt.imsave(f'{self.save_directory}/{n:02d}_snapshot.png',self.simulator.get_image('snapshot'),cmap='gray')
				plt.imsave(f'{self.save_directory}/{n:02d}_final.png',self.simulator.get_grain_structure(self.simulator.get_image('final')),cmap='gray')
				
				if self.save_evolution.get() == '.tif':
					tif.imwrite(f'{self.save_directory}/{n:02d}_trajectory.tif',self.simulator.get_image_evolution())
				if self.save_evolution.get() == '.gif':
					array2gif.write_gif(self._make_gif(self.simulator.get_image_evolution()),f'{self.save_directory}/{n:02d}_trajectory.gif',fps=15)

				n += 1
		else: 
			self.simulator.run_simulation(use_imported_data=use_imported_data)

		# show results in the user interface 
		# if multiple simulations were run/saved, will show the last one
		if show_result == True:
			if self.simulator._snapshot_taken == True:
				new_snapshot = self._photo_image(self.simulator.get_image('snapshot'))
				new_snapshot = self._resizeImage(new_snapshot,128,128)
				self.snapshot.configure(image=new_snapshot)
				self.snapshot.image = new_snapshot
			else: 
				self.snapshot.configure(image=self.blank_image)
				self.snapshot.image = self.blank_image

			if self.after_snapshot.get() != 'stop':
				new_final = self._photo_image(self.simulator.get_image('final'))
				new_final = self._resizeImage(new_final,128,128)
				self.final_image.configure(image=new_final)
				self.final_image.image = new_final

				max_misorientation = self.tickscale_misorientation.get()/360
				new_crystal_structure = self._photo_image(self.simulator.get_grain_structure(self.simulator.get_image('final'),maximum_misorientation=max_misorientation,symmetry = self.crystal_symmetry))
				new_crystal_structure = self._resizeImage(new_crystal_structure,128,128)
				self.crystal_structure.configure(image=new_crystal_structure)
				self.crystal_structure.image = new_crystal_structure
			else:
				self.final_image.configure(image=self.blank_image)
				self.final_image.image = self.blank_image
				self.crystal_structure.configure(image=self.blank_image)
				self.crystal_structure.image = self.blank_image

	def _setup_directory(self):
		filename = self.entry_directory.get()

		# can do any additional validation required for directory here 
		if not filename: filename = 'default'

		self.save_directory = f'./{filename}'
		os.makedirs(self.save_directory,exist_ok = True)

	def _make_gif(self,image: np.ndarray):
		if image.max(): image *= int(255.0/image.max())
		image_rgb = list(np.stack([np.transpose(image,(0, 2, 1))]*3,axis=3))
		return image_rgb


	def _import_settings(self):
		try: 
			filename = tk.filedialog.askopenfilename(filetypes=(('SCRYiNG','*.scrying'),))
			with open(filename,'rb') as f: 
				settings = pickle.load(f)
		except FileNotFoundError:
			warnings.warn('Import of settings has failed.')
			return  		

		self.tickscale_maximum_time.set(settings['maximum_time'])
		self.scaleentry_random_seed._entry.delete(0,tk.END) 
		self.scaleentry_random_seed._entry.insert(0,settings['random_seed'])
		self.scaleentry_random_seed._variable.set(settings['random_seed'])
		self.use_random_seed.set(settings['use_random_seed'])

		self.tickscale_nucleation_rate.set(settings['nucleation_rate'])
		self.tickscale_max_crystals.set(settings['maximum_crystals'])
		self.orientation_mode.set(settings['new_orientation'])
		self.center_mode.set(settings['new_center'])

		# self.use_nucleation_data.set(settings['use_nucleation_data'])

		self.tickscale_growth_rate.set(settings['growth_rate'])
		self.periodic_boundary.set(settings['periodic_boundary'])
		if settings['imported_shape'] is not None: self.imported_shape = settings['imported_shape'] 
		self.crystal_shape.set(settings['crystal_shape'])
		self._update_crystal_shape_callback()
		
		self.snapshot_mode.set(settings['snapshot_mode'])
		self._update_snapshot_mode_callback()
		self.tickscale_snapshot_parameter.set(settings['snapshot_parameter'])
		self.after_snapshot.set(settings['after_snapshot'])
		self._update_after_snapshot_callback()

		# self.entry_directory.delete(0,tk.END)
		# self.entry_directory.insert(0,set(settings['save_directory']))
		self.tickscale_number_simulations.set(settings['number_of_simulations'])
		self.tickscale_misorientation.set(settings['maximum_misorientation'])


			

	def _export_settings(self):
		
		settings = dict()
		settings['maximum_time'] = self.tickscale_maximum_time.get()
		settings['image_size'] = int(self.tickscale_image_size.get())
		settings['random_seed'] = self.scaleentry_random_seed.value
		settings['use_random_seed'] = self.use_random_seed.get()
		settings['nucleation_rate'] = self.tickscale_nucleation_rate.get()
		settings['maximum_crystals'] = self.tickscale_max_crystals.get()
		settings['new_orientation'] = self.orientation_mode.get()
		settings['new_center'] = self.center_mode.get()

		# settings['use_nucleation_data'] = self.use_nucleation_data.get()

		settings['growth_rate'] = self.tickscale_growth_rate.get()
		settings['periodic_boundary'] = self.periodic_boundary.get()
		settings['crystal_shape'] = self.crystal_shape.get()
		# settings['shape_array'] = self.shape_array
		settings['imported_shape'] = self.imported_shape
		settings['snapshot_mode'] = self.snapshot_mode.get()
		settings['snapshot_parameter'] = self.tickscale_snapshot_parameter.get()
		settings['after_snapshot'] = self.after_snapshot.get()
		settings['save_directory'] = self.entry_directory.get()
		settings['number_of_simulations'] = self.tickscale_number_simulations.get()
		settings['maximum_misorientation'] = self.tickscale_misorientation.get()

		filename = tk.filedialog.asksaveasfilename(filetypes=(('SCRYiNG','*.scrying'),))
		with open(filename+'.scrying','wb') as f:
			pickle.dump(settings,f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	root = tk.Tk()
	scryUI(root)
	root.mainloop()