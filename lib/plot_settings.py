import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from io import StringIO
import os
import inspect


# ========================================================
# ======================== LABELS ========================
# ========================================================

# function : sets a given string as label via plt or axis
def SetLabelByString( labelstr, direction, axis = None ):
    if axis is None:
        if direction == 'x':
            plt.xlabel(r'%s'%labelstr)
        elif direction == 'y':
            plt.ylabel(r'%s'%labelstr)
        else: 
            raise RuntimeError( "Direction unknown! Allowed is only 'x' or 'y'." )
    else:
        if direction == 'x':
            axis.set_xlabel(r'%s'%labelstr)
        elif direction == 'y':
            axis.set_ylabel(r'%s'%labelstr)
        else:
            raise RuntimeError( "Direction unknown! Allowed is only 'x' or 'y'." )

# function : sets a label in certain style from variable + unit
def SetLabel( style, variable, unit, direction, axis = None ):
    if style == "STD":
        SetLabel_STD( variable, unit, direction, axis )
    elif style == "PRB":
        SetLabel_PRB( variable, unit, direction, axis )
    else:
        raise RuntimeError( "style unknown!" )

# function : sets a label in STD style from variable + unit | is called from the above function but can also be directly used
def SetLabel_STD( variable, unit, direction, axis = None ):
    if unit != None:
        labelstr = r"$" + variable + r"[" + unit + r"]$"
    else:
        labelstr = r"$" + variable + r"$"
    SetLabelByString( labelstr, direction, axis )

# function : sets a label in PRB style from variable + unit | is called from the above function but can also be directly used
def SetLabel_PRB( variable, unit, direction, axis = None ):
    if unit != None:
        labelstr = r"$" + variable + r"$ (units of $" + unit + r"$)"
    else:
        labelstr = r"$" + variable + r"$"
    SetLabelByString( labelstr, direction, axis )


# ========================================================
# ======================== FOLDERS =======================
# ========================================================

# function : builds a folder
def BuildFolder( path ):
    try:
        os.mkdir( path )
    except:
        pass

# function : builds a folder tree from an array of folders and a global path
def BuildFolderTree( tree, path = "" ):
    total_path = ""
    if path != "": # then add path to total_path
        if path[-1] != "/": # then add "/"
            total_path = "/"
        total_path = path + total_path
    
    for i in range( 0, len(np.atleast_1d(tree)) ): # build all subfolders
        total_path += np.atleast_1d(tree)[i] + "/"
        BuildFolder( total_path[:-1] )
    return total_path


# =======================================================
# ======================== CURVEFAMILY ==================
# =======================================================
CURVEFAMILY_description = "defines a curve family and certain plot settings for each curve to be plotted\n" 
CURVEFAMILY_description += "1.) construct the curve family providing the familysize and optionally allow_cycle and or an axis\n"
CURVEFAMILY_description += "2.) set any curve properties\n"
CURVEFAMILY_description += "\t- most of the properties can be either set shared (set_shared...) or individual (set_individual...)\n"
CURVEFAMILY_description += "\t- ranges are either iterated once or cyclically if allow_cycle is True\n"
CURVEFAMILY_description += "\t- any shared simple arguments, e.g., markersize can be handed over as kwargs with the shared_kwargs method \n"
CURVEFAMILY_description += "3.) plot your data with the plot method\n"
CURVEFAMILY_description += "\t- hand over x, y and optionally increase or other arguments that are allowed in plt.plot or plt.errorbar\n"
CURVEFAMILY_description += "\t- a certain property set can be skipped with skip_to_next or considered twice if increase is set to False\n"
CURVEFAMILY_description += "\t- any individual simple arguments, e.g., markersize can be handed over as kwargs in the plot method \n"

class CURVEFAMILY():
    # initial variables:
    familysize = 1
    curve_counter = 0
    allow_cycle = False # if a range of properties is exceeded, it can be reiterated if allow_cycle is True
    axis = None
    shared_kwargs = None

    # color:
    is_color_shared = True
    is_color_continuous = False
    shared_color = "black"
    color_map = 0
    color_map_cutout = [0,1]
    dcolor = 0
    color_shift = 0 # has to be set

    # linestyle:
    is_linestyle_shared = True
    shared_linestyle = "-"
    linestyle_range = 0 # has to be set

    # markerstyle:
    is_markerstyle_shared = True
    shared_markerstyle = ""
    shared_markersize = 5
    markerstyle_range = 0 # has to be set
    markerstyle_allow_cycle = False

    # markevery:
    is_markevery_shared = True
    shared_markevery = 1
    number_of_markers = 0 # has to be set

    # errorbars:
    plot_errorbars = False
    shared_errorbarstyle = ""

    # initialization
    def __init__( self, familysize, allow_cycle = False, axis = None ):
        self.familysize = familysize
        self.allow_cycle = allow_cycle
        self.axis = axis
        self.curve_counter = 0

    # shared kwargs
    def set_shared_kwargs( self, **kwargs ):
        self.shared_kwargs = kwargs 

    # return the curve counter
    def get_counter( self ):
        if self.curve_counter >= self.familysize:
            raise Exception("curve_counter exceeds the familysize")
        return self.curve_counter

    # increase the curve counter if demanded
    def increase_counter_if( self, increase ): 
        if increase: 
            self.curve_counter += 1
        if self.curve_counter == self.familysize:
            if self.allow_cycle:
                self.curve_counter = 0
            else:
                print("CURVEFAMILY has been completed")

    # list all available methods
    def help( self ):
        print( "========== class CURVEFAMILY ==========\n")
        print( "=== description ===" )
        print( CURVEFAMILY_description )
        print( "=== available methods ===")
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'): # remove private and protected stuff
                if inspect.ismethod(i[1]):
                    print(i[0])

    # list all members
    def list_members( self ):
        print( "=== class members ===")
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'): # remove private and protected stuff
                if not inspect.ismethod(i[1]): # remove methods
                    print(i)

    # list all property ranges
    def list_ranges( self ):
        print_dict( "color_maps", color_maps )
        print_dict( "color_ranges", color_ranges )
        print_dict( "linestyle_ranges", linestyle_ranges )
        print_dict( "markerstyle_ranges", markerstyle_ranges )

# ==================== COLORS =========================
    def set_shared_color( self, color ):
        self.is_color_shared = True
        self.shared_color = color 

    def set_individual_colors( self, color_map = "default", color_shift = 0, color_map_cutout = [0,1] ):
        self.is_color_shared = False
        self.color_shift = color_shift
        self.color_map_cutout = color_map_cutout
        if np.shape( color_map ) == (): # use predefined color map
            try: # self-defined discrete color range 
                self.color_map = color_ranges[color_map]
            except: # continuous color maps 
                self.is_color_continuous = True
                self.dcolor = 1.0 / self.familysize
                try: # of matplotlib
                    self.color_map = plt.get_cmap('%s'%color_map)
                except: # self-defined
                    self.color_map = color_maps[color_map]                    
        else: # use custom color map
            self.color_map = color_map

    def get_color( self, increase = False ):
        current_counter = self.get_counter()
        self.increase_counter_if( increase )
        if self.is_color_shared: # shared colors
            return self.shared_color
        else: # individual colors
            if self.is_color_continuous: # continuous color map
                cmap_value = (current_counter + self.color_shift) * self.dcolor
                cmap_value = cmap_value - np.floor(cmap_value) # map back to the range 0 -> 1
                cmap_value = self.color_map_cutout[0] + cmap_value * ( self.color_map_cutout[1] - self.color_map_cutout[0] ) # renormalize to given cutout
                return self.color_map( cmap_value )
            else: # discrete color range
                return self.color_map[ current_counter ]
# ====================================================

# ==================== LINESTYLES ====================
    def set_shared_linestyle( self, linestyle ):
        self.is_linestyle_shared = True
        self.shared_linestyle = linestyle

    def set_individual_linestyles( self, linestyle_range = "default" ):
        self.is_linestyle_shared = False
        if np.shape( linestyle_range ) == (): # use predefined linestyle_range
            self.linestyle_range = linestyle_ranges[linestyle_range]
        else: # use custom linestyle_range
            self.linestyle_range = linestyle_range

    def get_linestyle( self, increase = False ):
        current_counter = self.get_counter()
        self.increase_counter_if( increase )
        if self.is_linestyle_shared:
            return self.shared_linestyle
        else:
            return self.linestyle_range[ current_counter ]
# ====================================================
    
# ==================== MARKERSTYLES ==================
    def set_shared_markerstyle( self, markerstyle ):
        self.is_markerstyle_shared = True
        self.shared_markerstyle = markerstyle

    def set_individual_markerstyles( self, markerstyle_range = "default" ):
        self.is_markerstyle_shared = False
        if np.shape( markerstyle_range ) == (): # use predefined markerstyle_range
            self.markerstyle_range = markerstyle_ranges[markerstyle_range]
        else: # use custom markerstyle_range
            self.markerstyle_range = markerstyle_range

    def get_markerstyle( self, increase = False ):
        current_counter = self.get_counter()
        self.increase_counter_if( increase )
        if self.markerstyle_allow_cycle == True:
            current_counter = current_counter % len(self.markerstyle_range)
        if self.is_markerstyle_shared:
            return self.shared_markerstyle
        else:
            return self.markerstyle_range[ current_counter ]
# ====================================================

# ==================== MARKEVERIES ==================
    def set_shared_markevery( self, markevery ):
        self.is_markevery_shared = True
        self.shared_markevery = markevery

    def set_individual_markeveries( self, number_of_markers = 10 ):
        self.is_markevery_shared = False
        self.number_of_markers = number_of_markers

    def get_markevery( self, number_of_data, increase = False ):
        current_counter = self.get_counter()
        self.increase_counter_if( increase )

        if self.is_markevery_shared:
            return self.shared_markevery
        else:
            individual_markevery = number_of_data / self.number_of_markers
            return tuple( [ int( individual_markevery/self.familysize ) * current_counter, int( individual_markevery ) ] )
# ====================================================

# ==================== ERRORBARS ==================
    def set_shared_errorbarstyle( self, errorbarstyle = "default" ):
        self.plot_errorbars = True
        self.shared_errorbarstyle = errorbarstyle

    # this has to be done
# ====================================================
    def plot( self, x, y, increase = True, **kwargs ):
        plotter = None
        if self.axis is None: # plot with plt
            if self.plot_errorbars:
                plotter = plt.errorbar
            else:
                plotter = plt.plot
        else: # plot on a given axis
            if self.plot_errorbars:
                plotter = self.axis.errorbar
            else:
                plotter = self.axis.plot
        
        if self.shared_kwargs is None: # guarantee that shared_kwargs is not empty
            self.shared_kwargs = {"visible" : True}

        plotter( x, y, color = self.get_color(), ls = self.get_linestyle(), marker = self.get_markerstyle(), markevery = self.get_markevery( len(np.atleast_1d(x)) ), **kwargs, **self.shared_kwargs )
        self.increase_counter_if( increase )

    def skip_to_next( self ):
        self.curve_counter += 1

    def set_axis( self, axis ):
        self.axis = axis

# colors:
black_rgb = [0,0,0]
lightblack_rgb = [0,0,0,0.5]
white_rgb = [1,1,1]
tugreen_html = "#83B818"
tugreen_rgb = [0.514,0.722,0.094]
tudarkgreen_html = "#699313"
tulightgreen_html = "#9CC646"
tuorange_html = "#D98207"
tuorange_rgb = [0.851,0.510,0.027]

# color maps:
def generate_linear_cmap_3p( cl_rgb, cm_rgb, cr_rgb ):
    N = 256 # RGB range
    vals = np.ones((N,4))
    vals[:,0] = np.append( np.linspace( cl_rgb[0], cm_rgb[0], int(N/2) ), np.linspace( cm_rgb[0], cr_rgb[0], int(N/2) ) )
    vals[:,1] = np.append( np.linspace( cl_rgb[1], cm_rgb[1], int(N/2) ), np.linspace( cm_rgb[1], cr_rgb[1], int(N/2) ) )
    vals[:,2] = np.append( np.linspace( cl_rgb[2], cm_rgb[2], int(N/2) ), np.linspace( cm_rgb[2], cr_rgb[2], int(N/2) ) )
    return ListedColormap(vals)

cmptugreen =    generate_linear_cmap_3p( black_rgb, tugreen_rgb, white_rgb )
cmptuorange =   generate_linear_cmap_3p( black_rgb, tuorange_rgb, white_rgb )

# maps for different property ranges (using python Dicts):
color_maps = { "cmptugreen": cmptugreen, "cmptuorange": cmptuorange }

color_ranges = { "default": [], "nice": [] }
color_ranges["default"] = np.array(( "black", "blue", "red", "green", "orange", "purple", "deepskyblue", "magenta", "lime", "yellow" ))
color_ranges["nice"] = np.array(( "blue", "orange", "black", "green", "deepskyblue", "magenta" ))
#color_ranges["thesis_tugreen"] = np.array(( "black", "#1a2505", "#344a0a", "#e6f1d1", "#b5d474", "#b5d474", "#9cc646", tugreen, "#699313", "#4f6e0e",    ))

linestyle_ranges = { "default": [] }
linestyle_ranges["default"] = np.array(( "solid", "dotted", "dashed", "dashdot", "solid", "dotted", "dashed", "dashdot", "solid", "dotted" ))

markerstyle_ranges = { "default": [], "short": [] }
markerstyle_ranges["default"] = np.array(( "x", "s", "^", "v", "o", "D", "2", "1", "p", "P" ))
markerstyle_ranges["short"] = np.array(( ".", "x", "+", "1", "2" ))

def print_dict( dict_name, dict ):
    content = list(dict.items())
    print( "===%s===" %dict_name )
    for c in content:
        print( c[0], ":", c[1] )
    print( "\n" )