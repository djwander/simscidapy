# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:37:23 2020

@author: David Wander
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate 
from scipy.interpolate import interp1d, griddata
from scipy.optimize import curve_fit
from copy import deepcopy
import matplotlib.transforms as mtransforms
import pickle
import gwyfile as gwy
from Curve import Curve

def check_argument(value,default,not_specified_value):
    """ convenience function returning value if value is not default, else not_specified_value
    used to check arguments handed to functions and set them to their default value if not specified
    """
    return default if value is not_specified_value else value
    
class Map:
    _ints = [int,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64]
    _floats = [float,np.single,np.double,np.longdouble,np.float32,np.float64]
    _complex = [np.complex64,np.complex128,np.complex_]
    _numbers = _ints+_floats+_complex
    ### Initialization
    def __init__(self,
                data=None,
                x_label="",
                x_unit="",
                y_label="",
                y_unit="",
                data_label="",
                data_unit="",
                title="",
                center_position=None,
                size=None,
                data_range=None,
                angle=0,
                colormap="viridis",
                default_interpolation_args={'method':'linear','fill_value':0,'rescale':False}):
        """ Class for handling 2D data
        Arguments:
            data (2d np array): 2D numpy array representing the data of the map
            x_label (string): label of the x axis
            y_label (string): label of the y axis
            x_unit (string): unit of the x axis
            y_unit (string): unit of the y axis
            data_label (string): label of the data 
            data_unit (string): unit of the data
            center_position (float,float): x and y coordinate of the maps center point. if specified, data_range is ignored; default: (0,0)
            size (float,float): size_x and size_y of the map; default: number of data points - 1 (=data point spacing of 1)
            angle (float): rotation angle of the map in degree
            data_range ((float,float),(float,float)): data range of the map in x and y direction; use this or center_position and size.
            default_interpolation_args (dictionary): default arguments used for evaluate. keys:
                "method":   one of 'linear','nearest', 'cubic'
                "fill_value" (float): value used to fill points that are outside the second map 
                "rescale" (bool)
        """
        if type(data) is list:
            data = np.array(data)
        if data is None:
            self.data = np.zeros((0,0))
        elif type(data) is np.ndarray:
            if not len(data.shape) == 2:
                raise ValueError(f"data must be a 2D array but a {len(data.shape)}D array was given!")
            else:
                self.data = data
        else:
            raise ValueError(f"data must be a 2D array or None but {type(data)} was given!")
        self.data = self.data.astype(float)

        self.x_label = str(x_label)
        self.y_label = str(y_label)
        self.x_unit = str(x_unit)
        self.y_unit = str(y_unit)
        self.data_label = str(data_label)
        self.data_unit = str(data_unit)
        self.title = str(title)
        self.colormap = colormap

        self._set_position(center_position=center_position,size=size,data_range=data_range,angle=angle)
        
        self.default_imshow_args = {'origin':'lower','aspect':'auto'}
        
        invalid_interpolation_args = [k for k in default_interpolation_args.keys() if k not in ['method','fill_value','rescale']]
        if not invalid_interpolation_args == []:
            raise ValueError(f'Unknown interpolation argument passed to Map: {invalid_interpolation_args}')
        if not default_interpolation_args['method'] in ['nearest','linear','cubic']:
            raise ValueError(f"default_interpolation has to be one of 'nearest','linear','cubic', not '{default_interpolation}'")
        self.default_interpolation_args = default_interpolation_args

        self.default_add_interpolation_args = {'method':'linear','fill_value':0,'rescale':False}
        self.default_sub_interpolation_args = {'method':'linear','fill_value':0,'rescale':False}
        self.default_mul_interpolation_args = {'method':'linear','fill_value':1,'rescale':False}
        self.default_div_interpolation_args = {'method':'linear','fill_value':1,'rescale':False}

    def _set_position(self,center_position=None,size=None,data_range=None,angle=0):
        self.angle = float(angle)

        if data_range is not None and center_position is None and size is None: # use data_range only if neither center_position nor size are given
            try:
                center_position = np.array([(data_range[0][0]+data_range[0][1])/2 , (data_range[1][0]+data_range[1][1])/2 ])
                size = np.array([data_range[0][1]-data_range[0][0], data_range[1][1]-data_range[1][0] ])
            except:
                raise ValueError("data_range must be of the format ((float,float),(float,float))!")
        
        self.center_position = check_argument(center_position,(0,0),None)
        
        if self.data is None:
            default_size = (0,0)
        else:
            default_size = [self.data.shape[0]-1,self.data.shape[1]-1]
        self.size = check_argument(size,default_size,None)
        
        
    def from_curves(self,list_of_curves,y_spacing=1, x_range=(None,None), x_values=None,transpose=False):
        # extract the map from a list of curves.
        # list_of_curves: list of curve objects to extract the data from
        # y_spacing (float or list of floats): 
        #       if single float given: assume same spacing for all curves. start at y=0
        #       if list of floats: list of the absolute y values for each of the curves. the length of the list must coincide with len(list_of_curves), 
        # x_range (float,float): take only x-values between the two given numbers for the map. (None,None) for full available x range
        # x_values (list of floats): if specified: evaluate the curves at the given x_values. Use this if the curves do not have the same spacing in x. Note: values need to be equally spaced!
        # transpose (bool): False (default): the x-axis is the axis of the curves -> the curves are stacked in y direction. True: inverse -> curves stacked along x
        
        # initialize data array
        sy = len(list_of_curves) # size of the map along y
        sx = len(x_values) if x_values is not None else len(list_of_curves[0]._get_x_range_x_y(x_range)[0])
        self.data = np.zeros((sy,sx)) # transpose at the end 
        # fill it with data
        for i,c in enumerate(list_of_curves):
            if x_values is None:
                self.data[i] = c._get_x_range_x_y(x_range)[1]
            else:
                self.data[i] = c.evaluate(x_values)
        if not transpose:
            self.data = self.data.transpose()
        self.data = self.data.astype(float)
        # calculate the position of the map
        self.center_position = [0,0]
        self.size = [0,0]
        if x_values is not None:
            self.center_position[0] = (x_values[-1]+x_values[0])/2 
            self.size[0] = x_values[-1]-x_values[0]
        else:
            x_c1 = list_of_curves[0]._get_x_range_x_y(x_range)[0] # take x values of curve
            self.center_position[0] = (x_c1[-1]+x_c1[0])/2
            self.size[0] = x_c1[-1]-x_c1[0]
        if type(y_spacing) in self._numbers:
            self.center_position[1] = 0
            self.size[1] = (len(list_of_curves)-1)*y_spacing
        elif type(y_spacing) is list:
            self.center_position[1] = (float(y_spacing[-1])+float(y_spacing[0]))/2
            self.size[1] = float(y_spacing[-1])-float(y_spacing[0])
        else:
            raise ValueError("y_spacing must be either a float or a list of floats!")
        return self
    ### Helper functions
    @staticmethod
    def rot_mat(angle):
        """ generate 2D rotation matrix
        Arguemtens:
            angle(float): angle in degree
        Returns:
            2D numpy array - rotation matrix for rotation of angle """
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))
    ### Set functions
    def set_data_label(self,label):
        assert type(label) is str, f'The label has to be a string! {type(label)} was given!'
        self.data_label = label 
    def set_data_unit(self,unit):
        assert type(unit) is str, f'The label has to be a string! {type(unit)} was given!'
        self.data_unit = unit 
    def set_colormap(self,colormap):
        self.colormap = colormap    
    def set_title(self,title):
        self.title = str(title)
    ### Get functions
    def get_corners_data(self):
        rot_mat = self.rot_mat(self.angle)
        x = np.array([self.size[0]/2,0])
        xp = rot_mat.dot(x)
        y = np.array([0,self.size[1]/2])
        yp = rot_mat.dot(y)
        center = np.array(self.center_position)
        corners = [(1,1),(-1,1),(-1,-1),(1,-1)]
        res = []
        for c in corners:
            res.append(center + c[0]*xp + c[1]*yp)
        return res
    def get_corners_plot(self):
        rot_mat = self.rot_mat(self.angle)
        x = np.array([(self.data.shape[0]*self.get_x1_spacing())/2,0])
        xp = rot_mat.dot(x)
        y = np.array([0,(self.data.shape[1]*self.get_x2_spacing())/2])
        yp = rot_mat.dot(y)
        center = np.array(self.center_position)
        corners = [(1,1),(-1,1),(-1,-1),(1,-1)]
        res = []
        for c in corners:
            res.append(center + c[0]*xp + c[1]*yp)
        return res
    def get_x1_spacing(self):
        """ spacing of data points along the x1 axis of the grid (x1 is at self.angle from x)"""
        return self.size[0]/(self.data.shape[0]-1)
    def get_x2_spacing(self):
        """ spacing of data points along the x2 axis of the grid (x2 is at self.angle from y)"""
        return self.size[1]/(self.data.shape[1]-1)

    def get_grid_vectors(self):
        """ returns the two vectors spanning the grid
        Returns:
            """
        rot = self.rot_mat(self.angle)
        v1 = rot.dot(np.array([1,0])*self.get_x1_spacing())
        v2 = rot.dot(np.array([0,1])*self.get_x2_spacing())
        return (v1,v2)
        
    def get_pt_coordinates(self):
        """ get the coordinates of all points of the map 
        Returns:
            2D numpy array (x_pts, y_pts): x_pts holding all x-coordinates, and y_pts all the y-coordinates; 
        """
        x1, x2 = self.get_grid_vectors()
        ptsx = self.data.shape[0]
        ptsy = self.data.shape[1]
        x_pts = np.zeros(ptsx*ptsy)
        y_pts = np.zeros(ptsx*ptsy)
        i=0
        for y in range(ptsy):
            for x in range(ptsx): 
                v = (self.center_position+
                    (x-ptsx/2+0.5)*x1+
                    (y-ptsy/2+0.5)*x2)
                x_pts[i] = v[0]
                y_pts[i] = v[1]
                i+=1
        return (x_pts,y_pts)
    def get_xyz(self):
        """ get the x,y and z coordinates of the data points
        Returns:
            2D numpy array: (x,y,z) where x,y and z are 1D lists
        """
        x,y = self.get_pt_coordinates()
        z = self.evaluate(x,y,method='nearest')
        return np.array([x,y,z])

    def get_lines(self,indices=None,y_range=None):
        """ get a list of lines as Curve objects
        Arguments:
            indices(list of int): indices of the lines to return. 0=lowest line.
            y_range((float,float)): y_range (at the center along x direction) out of which all lines will be returned. ignored if indices is defined
        Returns:
            list of Curve objects holding the extra variables: map_index, map_y, map_angle, map_parent
        """
        if indices is not None: # use indices argument if it is defined
            ret = []
            x = (np.arange(0,self.data.shape[0])-(self.data.shape[0]-1)/2)*self.get_x1_spacing()+self.center_position[0]
            for i in indices:
                if not type(i) in self._ints:
                    raise ValueError(f'Invalid index "{i}" passed to Curve.get_lines. Indices have to be integers!')
                if i < -self.data.shape[1] or i > self.data.shape[1]-1:
                    raise ValueError(f'Index out of bounds: {i} has to be within [0-{self.data.shape[1]-1}]!')
                if i < 0:
                    i = self.data.shape[1]+i
                y = (i-(self.data.shape[1]-1)/2)*self.get_x2_spacing()
                title = f'Index: {i}, y={y}'
                if self.y_unit is not None:
                    title += '('+self.y_unit+')'
                c = Curve(x,self.data[:,i],
                        x_label=self.x_label,x_unit=self.x_unit,
                        y_label=self.data_label,y_unit=self.data_unit,
                        title=title)
                c.map_index = i
                c.map_y = y
                c.map_angle = self.angle
                c.map_parent = self
                ret.append(c)
            return ret
        elif y_range is not None: # if no indices are given but a y range
            def y_to_y_index(map,y):
                ind = (y-map.center_position[1])/(map.get_x2_spacing()*np.cos(map.angle)) + (map.data.shape[1]-1)/2
                if ind < 0:
                    ind = 0
                elif ind > map.data.shape[1]-1:
                    ind = map.data.shape[1]-1
                return ind

            index_low = 0 if y_range[0] is None else y_to_y_index(self,y_range[0])
            index_up  = self.data.shape[1]-1  if y_range[0] is None else y_to_y_index(self,y_range[1])

            indices = np.arange(math.ceil(index_low),int(index_up)+1)
            return self.get_lines(indices=indices)
        else:
            return []
    ### Data manipulation
    def evaluate(self,pt_x,pt_y,method=None,fill_value=None, rescale=None):
        """ evaluate the value of the map at given points using interpolation
        uses scipy.interpolate.griddata. see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html for more information
        Arguments:
            method(string)[opt] Method of interpolation. One of
                nearest:    return the value at the data point closest to the point of interpolation. See NearestNDInterpolator for more details.
                linear:     tessellate the input point set to n-dimensional simplices, and interpolate linearly on each simplex. See LinearNDInterpolator for more details.
                cubic (1-D): return the value determined from a cubic spline.
                cubic (2-D): return the value determined from a piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface. See CloughTocher2DInterpolator for more details.

            fill_value(float)[opt]: Value used to fill in for requested points outside of the convex hull of the input points. If not provided, then self.default_fill_value is used. This option has no effect for the ‘nearest’ method.
            rescale(bool)[opt]: Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.
        """
        method = check_argument(method,self.default_interpolation_args['method'],None)
        fill_value = check_argument(fill_value,self.default_interpolation_args['fill_value'],None)
        rescale = check_argument(rescale,self.default_interpolation_args['rescale'],None)
        x_pts,y_pts=self.get_pt_coordinates()
        return griddata(points=(x_pts,y_pts),values=self.data.transpose().flatten(),
                        xi=(pt_x,pt_y),
                        method=method,fill_value=fill_value,rescale=rescale)
    def apply_transformation(self,transformation):
        """ apply an arbitrary transformation to the data points
        Arguments:
            func transformation(x,y,z): 
                takes x,y,z numpy arrays of x, y and z values of the data points
                returns z a numpy list with the new z values of the data points
        Returns:
            Map object: holding the new values. the object itself is changed as well"""
        x,y,z = self.get_xyz()
        z_n = transformation(x,y,z)
        sx,sy = self.data.shape
        self.data = np.transpose(np.reshape(z_n,(sx,sy)))
        return self
    def apply_transformation_pointwise(self,transformation):
        """ apply an arbitrary transformation to the data points
        Arguments:
            func transformation(x,y,z): 
                takes for each point individually x,y,z 
                returns z (float) the new value at this point
        Returns:
            Map object: holding the new values. the object itself is changed as well"""
        x,y,z = self.get_xyz()
        z_n = np.zeros(x.shape)
        for i,_ in enumerate(x):
            z_n[i] = transformation(x[i],y[i],z[i])
        sx,sy = self.data.shape
        self.data = np.transpose(np.reshape(z_n,(sx,sy)))
        return self
    def apply_transformation_linewise(self,transformation,direction='h'):
        """ apply an arbitrary transformation to the data points
        Arguments:
            func transformation(c, y, y_index): 
                takes a Curve object c, holding the data of the line, a float y holding the y position of the line in units of the map and the index y_index of the line 
                returns (curve) the new values of the line (dont change the x-axis points on which the curve is defined!)
            direction (string): 'h' for horizontal lines (default); 'v' for vertical lines
        Returns:
            Map object: holding the new values. the object itself is changed as well"""
        if direction == 'v':
            self.data = self.data.T
            self.size = np.array([self.size[1],self.size[0]])

        y_spacing = self.get_x2_spacing()
        center_position = self.center_position
        angle = self.angle

        c_old = self.get_lines(y_range=(None,None))
        c_new = [transformation(c,c.map_y,i) for i,c in enumerate(c_old)]

        transpose = True if direction == 'v' else False
        self = self.from_curves(c_new,y_spacing,transpose=transpose)
        self.center_position = center_position
        self.angle = angle
        if transpose:
            self.size = np.array([self.size[1],self.size[0]])

    def cropped(self,center_position=None,
                size=None,
                interpolate=False,
                angle=0,
                x_pts=None,
                y_pts=None,
                method=None,fill_value=None, rescale=False):
        """ get a cropped map based on interpolation
        Arguments:
            center_position (float,float): x and y coordinate of the maps center point. if specified, data_range is ignored; default: (0,0)
            size (float,float): size_x and size_y of the map; default: number of data points
            interpolate (bool): 
                False: do not interpolate. just take all data inside the given area. in this case the new map can be smaller in size than specified by size. all below arguments are ignored
                True: calculate new map by interpolating the old one on the given area. the new map will have the exact size as specified by size
            angle (float): rotation angle of the map in degree
            x_pts (int): number of points of the new map along x or None to keep the spacing of the old map
            y_pts (int): number of points of the new map along y or None to keep the spacing of the old map
            method(string)[opt] Method of interpolation. One of
                nearest:    return the value at the data point closest to the point of interpolation. See NearestNDInterpolator for more details.
                linear:     tessellate the input point set to n-dimensional simplices, and interpolate linearly on each simplex. See LinearNDInterpolator for more details.
                cubic (1-D): return the value determined from a cubic spline.
                cubic (2-D): return the value determined from a piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface. See CloughTocher2DInterpolator for more details.

            fill_value(float)[opt]: Value used to fill in for requested points outside of the convex hull of the input points. If not provided, then self.default_fill_value is used. This option has no effect for the ‘nearest’ method.
            rescale(bool)[opt]: Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.
        Returns:
            Map object - a cropped copy of the old object. all other properties stay unchanged
        """
        ret = deepcopy(self) # copy all properties
        # set default values for unspecified arguments
        center_position = check_argument(center_position,self.center_position,None)
        size = check_argument(size,self.size,None)
        # calculate position of data points
        if interpolate:
            if x_pts is None:
                x1_spacing = np.abs(np.cos(angle)*self.get_x1_spacing() + np.sin(angle)*self.get_x2_spacing()) # use a spacing similar to the original map
                x_pts = int(size[0]/x1_spacing)+1
            if y_pts is None:
                x2_spacing = np.abs(np.cos(angle)*self.get_x2_spacing() + np.sin(angle)*self.get_x1_spacing())
                y_pts = int(size[1]/x2_spacing)+1
            ret.data = np.zeros((x_pts,y_pts))
            ret._set_position(center_position=center_position,size=size,angle=angle) # set new position
            pt_x,pt_y = ret.get_pt_coordinates() # calculate new grid
        else:
            # get position of maps data points
            pt_x,pt_y = self.get_pt_coordinates()
            pt_vecs = np.transpose([pt_x,pt_y])
            pt_vecs -= center_position # vector of the center_point to the point of the grid
            # rotate by -angle
            rot = self.rot_mat(-self.angle)
            pt_vecs = np.dot(rot,pt_vecs.T).T
            # get all points inside the new size
            pt_indices = [i for i in range(len(pt_x)) if np.abs(pt_vecs[i][0]) <= size[0]/2 and np.abs(pt_vecs[i][1]) <= size[1]/2]
            pt_x = pt_x[pt_indices]
            pt_y = pt_y[pt_indices]
            # calculate new shape of the array (number of points in both dimensions)
            end_of_columns = np.where(np.diff(pt_y) != 0) 
            shape_x = end_of_columns[0][0] + 1 if len(end_of_columns[0]) > 0 else 1
            shape_y = int(len(pt_y)/shape_x)
            if len(pt_y) == 0:
                shape_x = 0
            ret.data = np.zeros((shape_x,shape_y))
            # calculate new size of the array (note: in general this is a bit smaller than what the used passed as size)
            veclenx = self.get_x1_spacing()
            vecleny = self.get_x2_spacing()
            size = (((shape_x-1)*veclenx,(shape_y-1)*vecleny))
            ret.center_position = center_position
            angle = self.angle
    
        sx,sy = ret.data.shape
        ret.data = self.evaluate(pt_x,pt_y,method=method,fill_value=fill_value,rescale=rescale)
        ret.data = np.reshape(ret.data,(sy,sx)).T
        ret.size = size
        ret.angle = angle
        return ret
    def where(self,condition):
        """ Find all points for which condition is fulfilled
        Arguments:
            condiction: function taking x,y,z and returning a boolean
        Returns:
            pts_x, pts_y, pts_z: lists of x,y and z values of the points fulfilling condition
        """
        x,y,z = self.get_xyz()
        res_x = []
        res_y = []
        res_z = []
        for i,_ in enumerate(x):
            if condition(x[i],y[i],z[i]):
                res_x.append(x[i])
                res_y.append(y[i])
                res_z.append(z[i])
        return np.array([res_x,res_y,res_z])

    ### profile extraction
    def profile(self,points,interpolation_args=None):
        """ Extract a 1D profile along the arbitrary trajectory given by points
        Arguments:
            points(int or list of float tuples): points on which the profile will be evaluated. 
                if a float is given: number of points evenly spaced between start and stop
                if a list of float tuples is given: list of points (for arbitrary point spacing/trajectories) -> the x axis of the resulting profile is the total distance from the start point via the defined points
            interpolation_args(dict): arguments passed to evaluate 
                method(string)[opt] Method of interpolation. One of
                    nearest:    return the value at the data point closest to the point of interpolation. See NearestNDInterpolator for more details.
                    linear:     tessellate the input point set to n-dimensional simplices, and interpolate linearly on each simplex. See LinearNDInterpolator for more details.
                    cubic (1-D): return the value determined from a cubic spline.
                    cubic (2-D): return the value determined from a piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface. See CloughTocher2DInterpolator for more details.

                fill_value(float)[opt]: Value used to fill in for requested points outside of the convex hull of the input points. If not provided, then self.default_fill_value is used. This option has no effect for the ‘nearest’ method.
                rescale(bool)[opt]: Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.
        Returns:
            Profile object holding the interpolated value of the map versus the total distance from the start point along the given trajectory. 
        """
        if interpolation_args is None:
            interpolation_args = self.default_interpolation_args
        points = np.array(points)

        pt_x,pt_y = np.array(points).T
        y = self.evaluate(pt_x,pt_y,**interpolation_args)
        x = np.zeros(len(pt_x))
        for i,_ in enumerate(x[1:]):
            x[i+1] = x[i]+ np.sqrt( (pt_x[i+1]-pt_x[i])**2 + 
                            (pt_y[i+1]-pt_y[i])**2)

        pt_start = points[0]
        pt_stop = points[-1]
        return Profile(map_positions = points,parent_map=self,
                    x=x,y=y,
                    y_label=self.data_label,y_unit=self.data_unit,
                    x_label="distance",x_unit=self.x_unit,
                    title=f'profile from {pt_start} to {pt_stop}')

    @staticmethod
    def _linearly_spaced_points(pt_start,pt_stop,num_points):
        """ generate a list of coordinates of num_points linearly spaced between pt_start and pt_stop
        Arguments:
            pt_start(float,float): start coordinate
            pt_stop(float,float): stop coordinate
            num_points(int): number of points 
        """
        x_spacing = (pt_stop[0] - pt_start[0])/(num_points-1)
        x = np.linspace(0,num_points-1,num_points)*x_spacing+pt_start[0]
        y_spacing = (pt_stop[1] - pt_start[1])/(num_points-1)
        y = np.linspace(0,num_points-1,num_points)*y_spacing+pt_start[1]
        points = np.array([x,y]).T
        return points

    def straight_profile(self,pt_start,pt_stop,num_points,interpolation_args=None):
        """ Extract a 1D profile between pt_start and pt_stop
        Arguments:
            pt_start(float,float): position of the point where the profile will start in units of x and y 
            pt_stop(float,float): position of the point where the profile will stop in units of x and y
            num_points(int): number of equally spaced points on which the profile is evaluated on
            interpolation_args(dict): arguments passed to evaluate 
                method(string)[opt] Method of interpolation. One of
                    nearest:    return the value at the data point closest to the point of interpolation. See NearestNDInterpolator for more details.
                    linear:     tessellate the input point set to n-dimensional simplices, and interpolate linearly on each simplex. See LinearNDInterpolator for more details.
                    cubic (1-D): return the value determined from a cubic spline.
                    cubic (2-D): return the value determined from a piecewise cubic, continuously differentiable (C1), and approximately curvature-minimizing polynomial surface. See CloughTocher2DInterpolator for more details.

                fill_value(float)[opt]: Value used to fill in for requested points outside of the convex hull of the input points. If not provided, then self.default_fill_value is used. This option has no effect for the ‘nearest’ method.
                rescale(bool)[opt]: Rescale points to unit cube before performing interpolation. This is useful if some of the input dimensions have incommensurable units and differ by many orders of magnitude.
        Returns:
            Profile object holding the interpolated value of the map versus the total distance from the start point along the given trajectory. 
        """
        points = self._linearly_spaced_points(pt_start,pt_stop,num_points)
        return self.profile(points,interpolation_args)


    def __add__(self,other):
        """ add number to all values or 
        add second map to first point wise on the first map's points 
        using self.default_add_interpolation_args (standard: linear) interpolation"""
        ret = deepcopy(self)
        if type(other) in self._numbers:
            ret.data += other
            return ret
        elif type(other)==Map:
            x_pt, y_pt = self.get_pt_coordinates()
            sx, sy = self.data.shape
            ret.data += np.transpose(np.reshape(other.evaluate(x_pt,y_pt,**other.default_add_interpolation_args),(sy,sx)))
            return ret
        else:
            raise ValueError(f"Can not add object of type {type(other)} to a Map!")
    def __sub__(self,other):
        """ subtract number from all values or 
        subtract second map from first point wise on the first map's points 
        using self.default_sub_interpolation_args (standard: linear) interpolation"""
        ret = deepcopy(self)
        if type(other) in self._numbers:
            ret.data -= other
            return ret
        elif type(other)==Map:
            x_pt, y_pt = self.get_pt_coordinates()
            sx, sy = self.data.shape
            ret.data -= np.transpose(np.reshape(other.evaluate(x_pt,y_pt,**other.default_sub_interpolation_args),(sy,sx)))
            return ret
        else:
            raise ValueError(f"Can not subtract object of type {type(other)} from a Map!")
    
    def __mul__(self,other):
        """ multiply all values with or 
        multiply second map with first, point wise on the first map's points 
        using self.default_sub_interpolation_args (standard: linear) interpolation"""
        ret = deepcopy(self)
        if type(other) in self._numbers:
            ret.data *= other
            return ret
        elif type(other)==Map:
            x_pt, y_pt = self.get_pt_coordinates()
            sx, sy = self.data.shape
            ret.data *= np.transpose(np.reshape(other.evaluate(x_pt,y_pt,**other.default_mul_interpolation_args),(sy,sx)))
            return ret
        else:
            raise ValueError(f"Can not multiply object of type {type(other)} with a Map!")
        
    def __truediv__(self,other):
        """ divide all values by a number or 
        divide first map point wise by the second, on the first map's points 
        using self.default_sub_interpolation_args (standard: linear) interpolation"""
        ret = deepcopy(self)
        if type(other) in self._numbers:
            ret.data /= other
            return ret
        elif type(other)==Map:
            x_pt, y_pt = self.get_pt_coordinates()
            sx, sy = self.data.shape
            ret.data /= np.transpose(np.reshape(other.evaluate(x_pt,y_pt,**other.default_div_interpolation_args),(sy,sx)))
            return ret
        else:
            raise ValueError(f"Can not subtract object of type {type(other)} from a Map!")
        
        
    def select_rect(self,x_range,y_range,outside_value=0):
        """ convenience function to select a rectangle of the data for further data manipulation
        Arguments:
            x_range(float,float): 
            y_range(float,float):   
            outside_value(float): value of data points outside the selection. default:0 for multiplications/divisions 1 can be useful
        Returns: 
            copy of the map where all values outside the selected area are set to outside_value
        """
        ret = deepcopy(self)
        corners = np.array(self.get_corners_data())
        x_range = list(x_range)
        y_range = list(y_range)
        x_range[0] = x_range[0] if x_range[0] is not None else np.min(corners[:,0])
        x_range[1] = x_range[1] if x_range[1] is not None else np.max(corners[:,0])
        y_range[0] = y_range[0] if y_range[0] is not None else np.min(corners[:,1])
        y_range[1] = y_range[1] if y_range[1] is not None else np.max(corners[:,1])

        def my_rect(x,y,z):
            if (x < x_range[0] or x > x_range[1] or 
                y < y_range[0] or y > y_range[1]):
                return outside_value
            else:
                return z
        ret.apply_transformation_pointwise(my_rect)
        return ret

    def mean(self):
        """ return the mean value
        Returns:
            float - the mean value """
        return np.mean(self.data).astype(float)
    def max(self):
        """ return the maximum value
        Returns:
            float - the maximum value """
        return np.max(self.data).astype(float)
    def min(self):
        """ return the minimum value
        Returns:
            float - the minimum value """
        return np.min(self.data).astype(float)
    
    def subtract_mean(self):
        """ subtract of each data point the mean value of the map
        Returns:
            Map - the same object with the mean subtracted"""
        self.data -= self.mean()
        return self
    def mean_around(self,pt,radius=None):
        """ get mean value within radius around pt.
        Arguments:
            pt(tuple of float): coordinates of the point where to take the mean
            radius(float): radius within to average. if None: only take the value of the nearest point
        Returns:
            float - mean value of the map within radius about pt
        """
        nppt = np.array(pt).astype(float)
        if nppt.shape != (2,):
            raise ValueError(f"'{pt}' is not a valid argument for pt in mean_around! Give a tuple of floats!")

        if radius is None:
            return self.evaluate(nppt[0],nppt[1], method='nearest')
        else:
            rs = radius**2
            _,_,z = self.where(lambda x,y,z: (x-nppt[0])**2+(y-nppt[1])**2 <= rs)
            return np.mean(z)
    
    def three_point_plane_fit(self,pts,radius=None):
        """
        Fit a plane going through the three given points.
        Arguments:
            pts(list of float tuples): the points to fit the plane through in units of x and y. The list must have 3 entries.
            radius(float): average the value of all points inside radius around the given coordinates. Take nearest value if None is given. (default)
        Returns:
            a,b,c,d (floats) - parameters characterizing the plane according to a*x+b*y+c*z = d
        """
        pts = np.array(pts).astype(float)
        if pts.shape != (3,2):
            raise ValueError(f"'pts' as to be a list of 3 float tuples (or a (3,2) numpy array)! However a list of dimension {pts.shape} was given.")
        z = [self.mean_around(pts[i],radius) for i in range(len(pts))]

        ### calculate plane from three points
        # a plane is described by a*x+b*y+c*z = d
        p1 = np.append(pts[0],z[0])
        p2 = np.append(pts[1],z[1])
        p3 = np.append(pts[2],z[2])
        # two vectors in the plane
        v1 = p3 - p1
        v2 = p2 - p1

        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp

        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)
        return (a,b,c,d)
    def three_point_level(self,pts,radius=None):
        """
        Subtract a plane going through the three given points.
        Arguments:
            pts(list of float tuples): the points to fit the plane through in units of x and y. The list must have 3 entries.
            radius(float): average the value of all points inside radius around the given coordinates. Take nearest value if None is given. (default)
        Returns:
            Map - the same map after subtracting the fitted plane
        """
        a,b,c,d = self.three_point_plane_fit(pts,radius)
        self.apply_transformation(lambda x,y,z: z-(d-a*x-b*y)/c)
        return self
    def subtract_polynomial_linewise(self,degree,direction='h'):
        """ Subtract a polynomial fit to each line of the map individually. 
        Useful for example to remove the background drift in SPM data.
        Arguments:
            degree (int): degree of the polynomial to subtract
            direction (string): direction in which to cut the lines: 'h' for horizontal cut, 'v' for vertical cut
        Returns:
            Map - the same map after subtracting the fitted polynomials
        """
        def sub_poly(c,y,y_i):
            x,y = c.get_x_y()
            poly_coefs = np.polynomial.polynomial.polyfit(x,y,degree)
            c_poly = Curve(x,np.polynomial.polynomial.polyval(x,poly_coefs))
            return c-c_poly
        self.apply_transformation_linewise(sub_poly,direction=direction)
        return self

    def FFT(self):
        ret = deepcopy(self)
        ret.data = np.fft.fftshift(
                                np.fft.fft2(self.data))
        ret._set_position(data_range=((-1*int( (self.data.shape[0])/2)*4*self.get_x1_spacing(),int( (self.data.shape[0]-1)/2)*4*self.get_x1_spacing()),
                                      (-1*int( (self.data.shape[1])/2)*4*self.get_x2_spacing(),int( (self.data.shape[1]-1)/2)*4*self.get_x2_spacing())))
        ret.x_unit = self.x_unit+r'^{-1}'
        ret.y_unit = self.y_unit+r'^{-1}'
        return ret
    def IFFT(self,take_real=True):
        ret = deepcopy(self)
        ret.data = np.fft.ifft2(np.fft.ifftshift(
                                self.data))
        if take_real:
            ret.data = np.real(ret.data)
        ret._set_position(size=(self.data.shape[0]/self.size[0],
                                self.data.shape[1]/self.size[1]))
        ret.x_unit = self.x_unit+r'^{-1}'
        ret.y_unit = self.y_unit+r'^{-1}'
        return ret

    def fit(self,function,init_parameters):
        """

        adapted from https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
        """

        # This is the callable that is passed to curve_fit. M is a (2,N) array
        # where N is the total number of data points in Z, which will be ravelled
        # to one dimension.
        def _function(M,*args):
            x, y = M
            return function(x, y, *args)

        x,y,z = self.get_xyz()
        # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
        xdata = np.vstack((x.ravel(), y.ravel()))
        # Do the fit, using our custom _gaussian function which understands our
        # flattened (ravelled) ordering of the data points.
        return curve_fit(_function, xdata, z.ravel(), init_parameters)

    def fit_1D(self,function,init_parameters,weight=lambda x,y,z:z):
        """ Fit a 1D curve y=f(x) to the data. All data points of the map are used for fitting and weighted by the value given by weight. (default: weight=z)
        Arguments:
            function: function f of type y=f(x,params) to fit to the data
            init_parameters: initial parameter for the fit
            weight: function that determines the weight of the data point x,y. The fit minimizes w*(f-data)**2 where w is weight normalized to 1.
                Arguments: x,y,z (numpy lists)
                Returns: weight (float)
        Returns:
            popt: optimized parameters as returned from scipy.curve_fit
            pcov: covariance matrix as returned from scipy.curve_fit
        """
        x,y,z = self.get_xyz()
        
        # curve fit optimizes ((f-data)/sigma)**2; -> calculate sigma from weight
        w = weight(x,y,z)
        x = x[w!=0]
        y = y[w!=0]
        z = z[w!=0]
        w = w[w!=0]

        norm_w = w/np.max(w)
        sigma = np.sqrt(1/norm_w)
        return curve_fit(function,x,y,init_parameters,sigma=sigma)
    ### Plotting
    def setup_plot(self,ax):
        ax.set_title(self.title)
        
        xlabel = self.x_label
        if self.x_unit is not "":
            xlabel += f" ({self.x_unit})"
        ylabel = self.y_label
        if self.y_unit is not "":
            ylabel += f" ({self.y_unit})"    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        corners = self.get_corners_plot()
        ax.set_xlim(np.min(corners,axis=0)[0],np.max(corners,axis=0)[0])
        ax.set_ylim(np.min(corners,axis=0)[1],np.max(corners,axis=0)[1])

    def plot(self,ax,ax_colorbar=None,imshow_args=None):
        # plot the map to the specified axes
        # ax: axes to plot the map to
        # ax_colorbar: axes to plot the colorbar into. if None: no colorbar
        # imshow_args: additional arguments passed to imshow. e.g. vmin and vmax for defining a manual color range
        if imshow_args is None:
            imshow_args = {}
        im_args = deepcopy(self.default_imshow_args) # use default imshow args
        im_args.update(imshow_args) # and append/overwrite by imshow_args passed as argument

        if type(self.colormap) is str:
            cmap = plt.get_cmap(self.colormap)
        else: 
            cmap = self.colormap

        tr = mtransforms.Affine2D().rotate_deg(self.angle).translate(*self.center_position)
        shape = self.data.shape
        ex = self.data.shape[0]*self.get_x1_spacing()
        ey = self.data.shape[1]*self.get_x2_spacing()
        d = self.data.transpose()
        if d.dtype in self._complex:
            d = np.abs(d)
        im = ax.imshow(d, 
                cmap = cmap,
                extent = (-ex/2,ex/2,-ey/2,ey/2),
                **im_args)
        trans_data = tr + ax.transData
        im.set_transform(trans_data)
        #ax.autoscale()
        if ax_colorbar is not None:
            # Create colorbar
            cbar = ax_colorbar.figure.colorbar(im, cax=ax_colorbar)
            ylabel = self.data_label 
            if self.data_unit is not "":
                ylabel += " (" + str(self.data_unit) + ")"
            cbar.ax.set_ylabel(ylabel, rotation=-90, va="bottom")

    def plot_standalone(self,with_colorbar=True,colorbar_range=None,block=True,xlim={},ylim={}):
        """ Plot the map in a new window.
        Arguments:
            with_colorbar (bool): whether or not the scale of the color map should be shown
            colorbar_range ((float,float)): limits to be used for the color map. None for auto calculated limits
            block (bool): whether or not to block the execution of the program when showing the plot
            xlim (dict): dictionary passed to axes.set_xlim. The "left","right" entries specify the x limits of the map
            ylim (dict): dictionary passed to axes.set_ylim. The "bottom", "top" entries specify the y limits of the map
        """
        fig, ax = plt.subplots()
        fig.canvas.set_window_title(self.data_label)
        cbar_ax=None
        if with_colorbar:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        imshow_args = {}
        if colorbar_range is not None:
            imshow_args['vmin'] = colorbar_range[0]
            imshow_args['vmax'] = colorbar_range[1]
        
        self.setup_plot(ax)
        self.plot(ax,ax_colorbar=cbar_ax,imshow_args=imshow_args)
        ax.set_xlim(**xlim)
        ax.set_ylim(**ylim)
        plt.show(block=block)
    

    ### File I/O
    def save(self,file_path):
        dict = vars(self)
        pickle.dump(dict,open( file_path, "wb" ),protocol=4)

    def load(self,file_path):
        dict = pickle.load( open( file_path, "rb" ))
        self.__dict__ = dict
        return self

    def save_csv(self,file_path,precision=2,separator='\t'):
        """ Save the map to a csv file.
        Arguments:
            file_path (string): file path to the file to save into
            precision (int): floating point precision
            separator (string): character separating columns
        """
        with open(file_path,'w') as f:
            f.write(f'{self.data_label} ({self.data_unit})\n')
            f.write(f'data_label: {self.data_label}\n')
            f.write(f'data_unit: {self.data_unit}\n')
            f.write(f'x_label: {self.x_label}\n')
            f.write(f'x_unit: {self.x_unit}\n')
            f.write(f'y_label: {self.y_label}\n')
            f.write(f'y_unit: {self.y_unit}\n')
            f.write(f'center position: {self.center_position}\n')
            f.write(f'size: {self.size}\n')
            f.write(f'angle: {self.angle}\n')
            f.write(f'END OF HEADER\n')
            d = np.array2string(np.flip(self.data,axis=0),
                                precision=precision,
                                separator=separator)
            d=d.replace('[','')
            d=d.replace(']','')
            d=d.replace(' ','')
            f.write(d)
    
    def load_csv(self,file_path):
        with open(file_path,'r') as f:
            lines = f.readlines()
            ih = lines.index('END OF HEADER\n')
            
            header = lines[:ih]
            dheader = {}
            for i,h in enumerate(header[1:]):
                key = h[:h.index(':')]
                value = h[h.index(':')+2:-1]
                dheader[key] = value
            
            if 'data_label' in dheader: self.data_label = dheader['data_label']
            if 'data_unit' in dheader: self.data_unit = dheader['data_unit']
            if 'x_label' in dheader: self.x_label = dheader['x_label']
            if 'x_unit' in dheader: self.x_unit = dheader['x_unit']
            if 'y_label' in dheader: self.y_label = dheader['y_label']
            if 'y_unit' in dheader: self.y_unit = dheader['y_unit']
            if 'center position' in dheader: 
                cp_str = dheader['center position'].strip('(').strip(')').strip().split(',')
                self.center_position = [float(cp_str[0]),float(cp_str[1])]
            if 'size' in dheader: 
                s_str = dheader['size'].strip('(').strip(')').strip().split(',')
                self.size = [float(s_str[0]),float(s_str[1])]
            if 'angle' in dheader: self.angle = float(dheader['angle'])

            data = lines[ih+1:]
            for i,d in enumerate(data):
                data[i] = np.fromstring(d,sep='\t')
            data = np.flip(np.array(data),axis=0)
            self.data = data
        return self    

class GwyddionMap(Map):
    """ Class that handles maps from and to gwyddion (.gwy) """
    def __init__(self, file_path, channel,
                x_label="x",
                x_unit="m",
                y_label="y",
                y_unit="m",
                data_label=None,
                data_unit="",
                colormap="afmhot",
                default_interpolation_args={'method':'nearest','fill_value':0,'rescale':False}):
        """ Class for handling 2D gwyddion maps
        Arguments: 
            file_path (string): full path to the .gwy file from which to load the map
            channel (string): name of the channel to load
            x_label (string): label of the x axis; default: x
            y_label (string): label of the y axis; default: y
            x_unit (string): unit of the x axis; default: m
            y_unit (string): unit of the y axis; default: m
            data_label (string): label of the data; if None: equal to channel
            data_unit (string): unit of the data
        """
        self.load(file_path,channel)
        data_label = check_argument(data_label,channel,None)
        super().__init__(data=self.data,center_position=self.center_position,size=self.size,
                        x_label=x_label,x_unit=x_unit,y_label=y_label,y_unit=y_unit,
                        data_label=data_label,colormap=colormap,default_interpolation_args=default_interpolation_args)
        


    @staticmethod
    def save_map_to_gwy(map, file_path):
        """ save given map to a .gwy file (gwyddion file format)
        Arguments:
            file_path(string): file path to which the map will be saved"""
        obj = gwy.objects.GwyContainer()
        obj['/0/data/title'] = f"{map.data_label} ({map.data_unit})"
        obj['/0/data'] = gwy.objects.GwyDataField(
                data=map.data.T,
                xoff=map.center_position[0]-map.size[0]/2,
                yoff=map.center_position[1]-map.size[1]/2,
                xreal=map.size[0],
                yreal=map.size[1],
                si_unit_xy=gwy.objects.GwySIUnit(unitstr=map.x_unit))
        obj.tofile(file_path)

    @staticmethod
    def load_channels(file_path):
        """ load available channels in a .gwy file (gwyddion file format)
        Arumgents:
            file_path(string): full file path of the file to open
        Returns:
            list of available channels"""
        obj = gwy.load(file_path)
        # get a dictionary with the datafield titles as keys and the
        # datafield objects as values.
        channels = gwy.util.get_datafields(obj)   
        return channels.keys()

    def load(self,file_path,channel):
        """ load from a .gwy file (gwyddion file format)
        Arumgents:
            file_path(string): full file path of the file to open
            channel(string):   channel to load (only one channel per map object can be loaded)
        Returns:
            Map - the map holding the specified channel from the .gwy file"""
        obj = gwy.load(file_path)
        # get a dictionary with the datafield titles as keys and the
        # datafield objects as values.
        channels = gwy.util.get_datafields(obj)
        ch = channels[channel]
        self.data = ch.data.T
        sx = ch.xreal
        sy = ch.yreal
        cp = (ch.xoff+sx/2,ch.yoff+sy/2) # gwyddion offset is upper left corner position
        self._set_position(center_position=cp,size=(sx,sy))
        return self

    def save(self,file_path):
        """ save to a .gwy file (gwyddion file format)
        Arguments:
            file_path(string): file path to which the map will be saved"""
        GwyddionMap.save_map_to_gwy(self,file_path)

class Profile(Curve):
    """
    Profile extracted from a Map 
    """
    def __init__(self,
                 map_positions,
                 parent_map,
                 x = None,
                 y = None,
                 xy = None,
                 plot_args = None,
                 trajectory_plot_args = None,
                 title = "",
                 x_label = "",
                 y_label = "",
                 x_unit = None,
                 y_unit = None,
                 x_scale = "linear",
                 y_scale = "linear"):
        """ Curve class offering a natural way of working with 1D data sets as recorded in most measurements.
        Arguments:
            map_positions(list of tuples of floats): coordinates of the data points on the map of which the profile was extracted from
            parent_map(Map): Map of which the profile was extracted
            x (list of floats)[opt]: x values of the data points. ignored when y is not provided. x and y need to have same dimension
            y (list of floats)[opt]: y values of the data points. ignored when x is not provided. x and y need to have same dimension
            xy (list of tuples)[opt]: list of data points, each data point being a (x,y) tuple. ignored if x and y are given.
            plot_args (dictionary)[opt]: dictionary specifying plotting arguments. These arguments are passed to pyplot.plot when plotting the curve
            title (string)[opt]: title of the curve, used for the legend of a plot and it's title when plot by plot_standalone
            x_label (string)[opt]: label put on the x-axis
            y_label (string)[opt]: label put on the y-axis
            x_unit (string)[opt]:  unit of the x-axis values - also added to the x-axis label
            y_unit (string)[opt]:  unit of the y-axis values - also added to the y-axis label
            x_scale (string)[opt]: scale of the x-axis ("linear" or "log") only affects plot_standalone and setup_plot
            y_scale (string)[opt]: scale of the y-axis ("linear" or "log") only affects plot_standalone and setup_plot
        """
        self._map_positions = map_positions
        self._parent_map = parent_map
        super().__init__(x,y,xy,plot_args,title,x_label,y_label,x_unit,y_unit,x_scale,y_scale)
        self._trajectory_plot_args = check_argument(trajectory_plot_args,self._plot_args,None)

    def plot_trajectory(self,ax,plot_args=None):
        """ plot the trajectory along which the profile was extracted
        """
        if plot_args == None:
            plot_args = self._trajectory_plot_args if not self._trajectory_plot_args==None else {}
        ax.plot(self._map_positions.T[0],self._map_positions.T[1],**plot_args)


### Useful functions

## for fit_1d

def lower_cutoff_weight(z,thres):
    """ weight function to be used for fit_1D
    sets the weight of points with z<thres to 0, for all other points to z
    Arguments:
        z(1D np array): z values of points
        thres(float): threshold value
    """
    z[z<thres] = 0
    return z

def upper_cutoff_weight(z,thres):
    """ weight function to be used for fit_1D
    sets the weight of points with z>thres to 0, for all other points to z
    Arguments:
        z(1D np array): z values of points
        thres(float): threshold value
    """
    z[z>thres] = 0
    return z

def range_weight(z,ll,ul):
    """ weight function to be used for fit_1D
    sets the weight of points with ll<z<ul to z, for all other points to 0
    Arguments:
        z(1D np array): z values of points
        ll(float): lower limit
        ul(float): upper limit
    """
    z = lower_cutoff_weight(z,ll)
    z = upper_cutoff_weight(z,ul)
    return z
