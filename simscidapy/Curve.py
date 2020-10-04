# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:09:23 2019

@author: David Wander
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate 
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from copy import deepcopy
import scipy.signal as sig
import pickle
import math
from numba import njit, prange

def moving_average(data, N):
    """
    Calculates the moving average of data within a window of N data points.
    (the new value at one point x is the average of all N values around x)

    Arguments:
        data:       1D numpy array with the data to average
        N:          size of the window in number of data points
    Returns:
        1D numpy array with the moving average of data
    """
    return np.convolve(data, np.ones((N)) / N, mode='valid')  # use convolution with same weight for all values

@njit(parallel = True)
def compiled_convolution(f,g,x_spacing,index_shifts):
    """ Calculate the convolution using a compiled function for speeding up the execution. 
    Note: the function is compiled during its first execution. A speedup can only be expected for subsequent calls.
    Arguments:
        f (numpy array): y values of the function f
        g (numpy array): y values of the function g
        x_spacing (float): distance along x between neighboring data points. note: all points have to be spaced equally!
        index_shifts (int numpy array): list of 
    """
    res = np.zeros(len(index_shifts))
    len_g = len(g)
    for i in prange(len(index_shifts)):
        res[i] = x_spacing * np.sum(
            f[index_shifts[i]:index_shifts[i]+len_g]*g
        )
    return res

def default_csv_header_parser(header,delimiter=';'):
    """ default parser for csv headers:
    ignore all header lines but the last, which is formatted like this: # xlabel(xunit);ylabel(yunit)
    Arguments:
        header (list of strings):   the full text of the header split by lines
        delimiter (string):         character separating columns of the csv file
    Returns:
        dictionary holding the axis labels and units
    """
    header = header[-1][2:] # take last row of header (by default there is only one anyways) and remove leading '#'
    cols = header.split(delimiter)
    res = {}
    res['x_label'] = cols[0][:cols[0].index('(')]
    res['x_unit'] = cols[0][cols[0].index('(')+1:cols[0].index(')')]
    res['y_label'] = cols[1][:cols[1].index('(')]
    res['y_unit'] = cols[1][cols[1].index('(')+1:cols[1].index(')')]
    return res

class Curve:
    """ Curve class offering a natural way of working with 1D data sets as recorded in most measurements."""

    available_plot_props = ["plot_args","title","x_label","y_label","x_unit","y_unit","x_scale","y_scale"]
    def __init__(self,
                 x = None,
                 y = None,
                 xy = None,
                 plot_args = None,
                 title = "",
                 x_label = "",
                 y_label = "",
                 x_unit = None,
                 y_unit = None,
                 x_scale = "linear",
                 y_scale = "linear"):
        """ Curve class offering a natural way of working with 1D data sets as recorded in most measurements.
        Arguments:
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
        self.set_data(x,y,xy)

        self._plot_args = plot_args
        self._title = title
        self._x_label = x_label
        self._y_label = y_label
        self._x_unit = x_unit
        self._y_unit = y_unit
        self._x_scale = x_scale
        self._y_scale = y_scale
        
        self.default_add_outside_range_value = "extrapolate"
        self.default_sub_outside_range_value = "extrapolate"
        self.default_mul_outside_range_value = "extrapolate"
        self.default_div_outside_range_value = "extrapolate"
        self.default_pow_outside_range_value = "extrapolate"
### Setters
    def set_data(self,x=None,y=None,xy=None):
        """ Set the data points of the curve
        Arguments:
            x (list of floats)[opt]: x values of the data points. ignored when y is not provided. x and y need to have same dimension
            y (list of floats)[opt]: y values of the data points. ignored when x is not provided. x and y need to have same dimension
            xy (list of tuples)[opt]: list of data points, each data point being a (x,y) tuple. ignored if x and y are given.
            if neither x & y nor xy are given, an empty list will be set as data (no poitns)
        Returns:
            Curve object: self
        """
        if x is not None and y is not None:
            self._x = np.array(x)
            self._y = np.array(y)
        elif xy is not None:
            xy = np.array(xy)
            xy_t = np.transpose(xy)
            self._x = xy_t[0]
            self._y = xy_t[1]
        else:
            self._x = np.array([])
            self._y = np.array([])
        
        self._sort_x()
        return self

    def set_plot_properties(self, props):
        """ set the plot properties of the curve. 
        Arguments:
           props (dictionary): contains the properties that should be changed.
                               possible options: "plot_args","title","x_label","y_label","x_unit","y_unit","x_scale","y_scale"
        Returns:
            Curve object with the applied properties
        """
        for p in props: 
            if not p in self.available_plot_props:
                raise ValueError(f"There is no plot property called {p}!")
        self._plot_args = props["plot_args"] if "plot_args" in props else self._plot_args
        self._title = props["title"] if "title" in props else self._title
        self._x_label = props["x_label"] if "x_label" in props else self._x_label
        self._y_label = props["y_label"] if "y_label" in props else self._y_label
        self._x_unit = props["x_unit"] if "x_unit" in props else self._x_unit
        self._y_unit = props["y_unit"] if "y_unit" in props else self._y_unit
        self._x_scale = props["x_scale"] if "x_scale" in props else self._x_scale
        self._y_scale = props["y_scale"] if "y_scale" in props else self._y_scale
        return self
    def _sort_x(self):
        """ helper function. internally sorts data points in ascending x order. """
        i = np.argsort(self._x)
        self._x = self._x[i]
        self._y = self._y[i]
### Getters
    def _get_x_range_x_y(self,x_range=(None,None)):
        """ internal helper function. get all data points lying in a given x_range in units of the x values. use get_x_y instead!
        Arguments:
            x_range(float,float): [lower_lim,upper_lim] in units of x values. 
                    None if values should be taken from the start/to the end
        Returns: [x,y] x,y numpy arrays with x_range[0] < x <= x_range[1]
        """
        index_x = [0,0]
        if x_range[0] is None or x_range[0] < self._x[0]:
            index_x[0] = 0
        elif x_range[0] > self._x[-1]:
            return [[],[]]
        else:
            index_x[0] = np.argmax(x_range[0]<=self._x)
        
        if x_range[1] is None or x_range[1] >= self._x[-1]:
            index_x[1] = len(self._x)
        elif x_range[1] < self._x[0]:
            return [[],[]]
        else:
            index_x[1] = np.argmax(x_range[1]<self._x)
        return [self._x[index_x[0]:index_x[1]],
                self._y[index_x[0]:index_x[1]]]
    def get_x(self,x_range=(None,None)):
        """ get a numpy list of the x values of all data points in the given x-range
        Arguments:
            x_range(float,float): [lower_lim,upper_lim] in units of x values. 
                    None if values should be taken from the start/to the end
        Returns:
            1D numpy array: x data in the given range
        """
        return np.real(self._get_x_range_x_y(x_range)[0])
    def get_y(self,x_range=(None,None)):
        """ get a numpy list of the y values of all data points in the given x-range
        Arguments:
            x_range(float,float): [lower_lim,upper_lim] in units of x values. 
                    None if values should be taken from the start/to the end
        Returns:
            1D numpy array: y data in the given range
        """
        return np.real(self._get_x_range_x_y(x_range)[1])
    def get_x_y(self,x_range=(None,None)):
        """ get a 2D numpy array of the values of all data points in the given x-range
        Arguments:
            x_range(float,float): [lower_lim,upper_lim] in units of x values. 
                    None if values should be taken from the start/to the end
        Returns:
            2D numpy array: [x_data,y_data]
        """
        return np.real(self._get_x_range_x_y(x_range))
    def get_xy(self,x_range=(None,None)):
        """ get a 2D numpy array of the values of all data points in the given x-range
        Arguments:
            x_range(float,float): [lower_lim,upper_lim] in units of x values. 
                    None if values should be taken from the start/to the end
        Returns:
            2D numpy array: [[pt1_x,pt1_y],[pt2_x,pt2_y],...]
        """
        return np.real(np.transpose(self._get_x_range_x_y(x_range)))
    def get_plot_properties(self):
        """ get all plot properties
        Arguments:
            None
        Returns:
            dictionary holding the plot properties. keys are:
                plot_args, title, x_label, x_unit, y_label, y_unit, x_scale, y_scale
        """
        d['plot_args']  = self._plot_args
        d['title']      = self._title
        d['x_label']    = self._x_label
        d['x_unit']     = self._x_unit
        d['x_scale']    = self._x_scale
        d['y_label']    = self._y_label
        d['y_unit']     = self._y_unit
        d['y_scale']    = self._y_scale
        return d
        
### Data IO
    def load_from_csv(self,fname,delimiter=';',skiprows=1,columns=(0,1), comments='#',header_parser=default_csv_header_parser):
        """ Load the data of a curve from a csv file. Overwrites the current data of the curve object.
        Arguments:
            fname(string): filename of the csv file to open
            delimiter(string): the delimiter character separating columns in the csv file. default ';'
            skiprows(int):  number of header lines to skip default 1
            columns(tuple of int): column index (0 based) to use for x and y axis of the curve. default: (0,1) -> first column x, second column y 
            comments(string): The characters or list of characters used to indicate the start of a comment. None implies no comments. For backwards compatibility, byte strings will be decoded as ‘latin1’. The default is ‘#’.
            header_parser(func): function parsing the header information. 
                Arguments: 
                    header(list of strings): list containing the header lines (number of lines = skiprows)
                    delimiter (string): the delimiter of the csv file
                Returns:
                    dictionary containing plot properties that will be applied to the curve object. for valid elements refer to available_plot_props
                Default: take last line of header which is formatted like this: # xlabel(xunit);ylabel(yunit)
        Returns:
            Curve object: self
        """
        # read header
        header = []
        with open(fname,'r') as f:
            for i in range(skiprows):
                header.append(f.readline())
        parsed = header_parser(header,delimiter)
        self.set_plot_properties(parsed)

        # read data
        data = np.loadtxt(fname,delimiter=delimiter,skiprows=skiprows,comments=comments)
        data= np.transpose(data)
        self._x = data[columns[0]].astype(float)
        self._y = data[columns[1]].astype(float)
        self._sort_x()
        return self

    def save_to_csv(self,fname,delimiter=';',fmt="%.4e"):
        """ Save the curve to a csv file. Does not save formatting information
        Arguments:
            fname(string): filename to which the curve will be saved
            delimiter(string): delimiter character used to separate columns
            fmt(string): format string specifying mainly the precision example: %.4e means scientific notation (e) with 4 digits after floating point; %.4f means float number with 4 digits after floating point
                        can be a list [x_fmt,y_fmt] as well. e.g. ['%.1e','%.4e'] if different format is required          
        """
        out = np.transpose(np.array([self._x,self._y]))
        header = f"{self._x_label}({self._x_unit}){delimiter}{self._y_label}({self._y_unit})"
        np.savetxt(fname,out,delimiter=delimiter,header=header,fmt=fmt)
    
    def pickle_save(self,fname):
        """ Save the curve object to a file from where it can be fully restored with pickle_load. (including all formatting arguments)
        Arguments:
            fname(string): filename to save the curve object to
        """
        pickle.dump(self, open( fname, "wb" ) )
        
    def pickle_load(self,fname):
        """ Load the curve object from a file where it had be saved to using pickle_save. (including all formatting arguments) 
            current data of the this object will be overwritten!
        Arguments:
            fname(string): filename to save the curve object to
        Return: 
            curve object as loaded from file
        """
        self = pickle.load( open( fname, "rb" ) )
        return self
    
### Plotting
    def setup_plot(self,ax):
        """ set up the axes of a plot (labels, title, scaling) using the formatting information of this curve object.
        Arguments:
            ax(matplotlib.axes): the axes that should be setup
        """
        ax.set_title(self._title)
        
        xlabel = self._x_label
        if self._x_unit is not None:
            xlabel += f" ({self._x_unit})"
        ylabel = self._y_label
        if self._y_unit is not None:
            ylabel += f" ({self._y_unit})"    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(self._x_scale)
        ax.set_yscale(self._y_scale)

    def plot(self,ax,x_range=(None,None),plot_args=None):
        """ plot the curve to the given axes
        Arguments:
            ax(matplotlib.axes): axes to plot into
            x_range(float,float): the x-range (in units of x data) of the curve that should be plotted. None for full range
            plot_args(dictionary): additional arguments passed to axes.plot, allowing for custom plot style. see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot for details
        """
        if plot_args == None:
            plot_args = self._plot_args if not self._plot_args==None else {}
        label = plot_args['label'] if 'label' in plot_args else self._title
        if 'label' in plot_args:
            del plot_args['label']
        x,y = self.get_x_y(x_range)
        if np.max(np.abs(np.imag(y))) > 0: # there is a complex number
            y = np.abs(y) # plot the absolute value
        ax.plot(x,y,label=label,**plot_args)
    
    def plot_standalone(self,x_range=(None,None),fig=None,ax=None,block=True, plot_args=None):
        """ Plot the curve in its own figure, setting up all axis and showing it.
        Arguments:
            x_range [opt] (float,float): x range of the data to show. if None: full available range
            fig [opt] (matplotlib.figure): figure to plot the plot into. only has an effect if ax is given as well.
            ax [opt] (matplotlib.axes): axes to plot into. If None: create new figure and axes
            block [opt] (bool): if True, the execution of the plot is blocked until the user closes the plot window
            plot_args(dictionary): additional arguments passed to axes.plot, allowing for custom plot style. see https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot for details
        """
        if ax is None:
            fig, ax = plt.subplots()
        if fig is not None:
            fig.canvas.set_window_title(self._title)
        self.setup_plot(ax)
        if plot_args is not None and 'x_label' in plot_args:
            ax.set_xlabel(plot_args['x_label'])
            del plot_args['x_label']
        if plot_args is not None and 'y_label' in plot_args:
            ax.set_ylabel(plot_args['y_label'])
            del plot_args['y_label']    
        self.plot(ax,x_range=x_range,plot_args=plot_args)
        if fig is not None:
            if block:
                plt.show(block=True)
            else:
                fig.show()
    @staticmethod
    def plot_several_curves(list_of_curves,block=True,legend=True,title=""):
        """ Plot all given curves in a new figure.
        Arguments:
            list_of_curves (list of Curve): Curves to be plotted in the same figure
            block [opt] (bool): if True, the execution of the plot is blocked until the user closes the plot window
            legend [opt] (bool): whether or not to show a legend
            title [opt] (string): title of the plot
        """
        _, ax = plt.subplots()
        list_of_curves[0].setup_plot(ax)
        ax.set_title(title)
        for c in list_of_curves:
            c.plot(ax)
        if legend:
            plt.legend()
        plt.show(block=block)
### Data treatment
    def crop(self,x_range):
        """ crop the curve to the given x_range (irreversible!)
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values. 
                   None if values should be taken from the start/to the end
        Returns:
            Curve: the cropped curve. note that the old curve object is overwritten
        """
        cropped = self._get_x_range_x_y(x_range)
        self._x = cropped[0]
        self._y = cropped[1]
        return self
        
    def cropped(self,x_range):
        """ get a copy of the curve cropped to the given x_range (does not affect the curve itself)
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values. 
                   None if values should be taken from the start/to the end
        Returns:
            Curve: the new, cropped curve
        """
        res = deepcopy(self)
        res.crop(x_range)
        return res
        
    def evaluate(self,x,interpolation="spline",interpolation_args={"k":3},outside_range_value="extrapolation"):
        """ evaluate the curve at the given point(s) by interpolating between surrounding data points.
        Arguments:
            x: number or numpy array with x values where the function should be evaluated at
            interpolation: see https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
                  "linear":   linear interpolation
                  "cubic": cubic spline 
                  "nearest":  value of nearest data point
                  "previous": value of previous data point
                  "next":     value of next data point
                  "spline":   spline interpolation of arbitrary order defined by interpolation_args: k
            interpolation_args: ignored for "cubic","nearest","previous","next" interpolations
                              for "spline" interpolation: passed to scipy.interpolate.splrep
                                  most important: "k": order of the spline
            outside_range_value(float or string): 
                "extrapolation" (default) extrapolate the data using the same method as for intrapolation
                "nearest": fill with the neares value of the actual data
                float: fill with this number

        Returns:  
            the value of the interpolation at the given position(s) x
        """
        # initialize ret (necessary if all x values are outside self._x)
        scalars = [float,np.float16,np.float32,np.float64,int,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64]
        if type(x) not in scalars+[np.ndarray,list]:
            raise ValueError(f"type of x has to be float, int or numpy.ndarray, not {type(x)}!")
        type_x = type(x)
        if type(x) in scalars:
            x = np.array([x])
        if type(x) is list:
            x = np.array(x).astype(float)
        #self._sort_x() # data needs to be in ascending x order. however, this is done in the constructor. trust on this here for performance reasons
        if interpolation=="spline":
            if len(self._x) < interpolation_args["k"]+1:
                raise ValueError(f'Trying to interpolate a curve of {len(self._x)} points with a {interpolation_args["k"]}th order spline. There must be at least {interpolation_args["k"]+1} points for this to work!')
            spl = interpolate.splrep(np.real(self._x), np.real(self._y), **interpolation_args)
            ret = interpolate.splev(x,spl)
        else: # calculate interpolation with interp1d
            # note: interp1d does not do extrapolation. -> do this 'manually'
            i_l = np.where(x<np.min(self._x))
            i_r = np.where(x>np.max(self._x))
            i_c = np.where( (x>=np.min(self._x)) & (x<=np.max(self._x)))
            if interpolation in ['nearest','previous','next']:
                ret_l = np.ones(len(i_l))*self._x[0]
                ret_r = np.ones(len(i_l))*self._x[-1]
            elif interpolation is 'linear':
                a_l = (self._y[1]-self._y[0])/(self._x[1]-self._x[0])
                b_l = self._y[0]-a_l*self._x[0]
                ret_l = a_l*x[i_l]+b_l
                a_r = (self._y[-1]-self._y[-2])/(self._x[-2]-self._x[-1])
                b_r = self._y[-1]-a_r*self._x[-1]
                ret_r = a_r*x[i_r]+b_r
    
            ret_c = interp1d(np.real(self._x), np.real(self._y), kind=interpolation)(x[i_c])
            ret = np.zeros(len(x))
            ret[i_l] = ret_l
            ret[i_c] = ret_c
            ret[i_r] = ret_r
        if outside_range_value is not "extrapolation":
            i = np.where((x<np.min(self._x))| (x>np.max(self._x))) # indices of x outside self._x
            if outside_range_value == "nearest":
                ret[i] = self.evaluate(x[i],interpolation='nearest',outside_range_value="extrapolation")
            elif type(outside_range_value) in [float,int]:
                ret[i] = outside_range_value
        
        if type_x in scalars:
            return ret[0]
        else:
            return ret
            
    def where(self,y,x_precision=None,interpolation="spline",interpolation_args={"k":3}):
        """ numerically find all x values where the curve takes the value y
        Arguments:
                y(float): y value to look for
                x_precision(float): maximum difference between the returned value and the actual value in units of x data. if None: 1/1000 of the full x-range
                interpolation: see https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
                  "cubic": cubic spline 
                  "nearest":  value of nearest data point
                  "previous": value of previous data point
                  "next":     value of next data point
                  "spline":   spline interpolation of arbitrary order defined by interpolation_args: k
              interpolation_args: ignored for "cubic","nearest","previous","next" interpolations
                  for "spline" interpolation: passed to scipy.interpolate.splrep
                              most important: "k": order of the spline
        returns:
              list of x values. can be empty if no point found!
        """
        if x_precision is None:
            x_precision = (np.max(self._x)-np.min(self._x))/1000

        # first find all data points where the curve crosses the given value
        sign = np.sign(self._y - y) # there is a sign change where they are equal
        sign[sign==0] = 1 # if the curve takes the exact value on a data point the sign change would be 1,0,-1 -> problem for the next step
        indices = np.argwhere( np.abs(np.diff(sign)) > 0 )
        
        # refine the x using interpolation
        def find_x_recursively(self,y,x1,x2,slope,x_precision,interpolation,interpolation_args):
            xmiddle = (x1+x2)/2
            if x2-x1 < x_precision:
                return xmiddle
            else:
                ymiddle = self.evaluate(xmiddle,interpolation,interpolation_args)-y
                if np.sign(ymiddle) == slope:
                    return find_x_recursively(self,y,x1,xmiddle,slope,x_precision,interpolation,interpolation_args)    
                else:
                    return find_x_recursively(self,y,xmiddle,x2,slope,x_precision,interpolation,interpolation_args)

        x_res = np.zeros(len(indices))    
        for i,v in enumerate(indices):
            j = v[0]
            x_res[i] = find_x_recursively(self,y,self._x[j],self._x[j+1],np.sign(self._y[j+1]-self._y[j]),
                                            x_precision,interpolation,interpolation_args)
        return x_res
    

    def get_spline(self,x_range=(None,None),spline_args={"k":3}):
        """ Calculate a spline interpolation to the data using scipy.interpolate.BSpline
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values. 
                   None if values should be taken from the start/to the end
            spline_args: arguments passed to BSpline. for details see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
        Returns:
            scipy.interpolate.BSpline
        """
        x,y = self.get_x_y(x_range)
        tck = interpolate.splrep(x,y,**spline_args)
        return interpolate.BSpline(*tck)
    
    def local_maxima(self,x_range=(None,None),x_precision=None,interpolation="spline",interpolation_args={"k":3}):
        """ numerically find local maxima of the curve
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values in which to sear for local maxima 
                   None if values should be taken from the start/to the end
            x_precision(float): maximum difference between the returned value and the actual value in units of x data. if None: 1/1000 of the full x-range
            interpolation(string): 
                  "cubic":    cubic spline 
                  "previous": in case of flat top: return x value of previous data point wrt maximum, otherwise the maximum value itself
                  "next":     in case of flat top: return x value of next data point wrt maximum, otherwise the maximum value itself
                  "spline":   spline interpolation of arbitrary order defined by interpolation_args: k
                  "none":     no interpolation -> can lead to x values between two data point in case of a flat top maximum
              interpolation_args: ignored for "cubic","previous","next" interpolations
                  for "spline" interpolation: passed to scipy.interpolate.splrep
                              most important: "k": order of the spline
        Returns:
            1D numpy array containing the x values of the local maxima
        """
        x,y = self.get_x_y(x_range)
        if x_precision is None:
            x_precision = (np.max(x)-np.min(x))/1000
        
        # first find all indices where the derivative changes sign
        dy = np.diff(y)
        sign = np.sign(dy)
        sign_change = np.diff(sign)
        ind_sign_change = np.argwhere(sign_change < 0)
        res = []
        for i in ind_sign_change:
            i = i[0]
            if sign_change[i] == -1: # there is a 0 slope part involved
                if sign[i] == 0: # it is the end of a plateau
                    continue
                elif sign[i] > 0: # it is the beginning of a plateau
                    i_plateau_end = np.where(sign_change[i+1:] !=0)
                    if len(i_plateau_end[0]) == 0: # this is a plateau at the end of the data
                        continue
                    i_plateau_end = i+1+i_plateau_end[0][0]
                    if sign[i_plateau_end+1] > 0: # its going further up after the plateau -> it is a saddle point
                        continue # ignore 
            # find the exact position of the maximum
            if interpolation in ['none','previous','next']:
                if sign[i+1] == 0: # if its a flat top maximum take the middle of the plateau
                    # find the end of the plateau
                    x1 = x[i+1]
                    x2 = x[i_plateau_end+1]
                    if interpolation is 'none':
                        res.append((x1+x2)/2)
                    elif interpolation is 'previous':
                        icenter = (i+1+i_plateau_end+1)/2
                        res.append(x[int(icenter)])
                    elif interpolation is 'next':
                        icenter = (i+1+i_plateau_end+1)/2
                        res.append(x[math.ceil(icenter)])
                else:
                    res.append(x[i+1])
            else:
                if interpolation=="spline":
                    spl = interpolate.splrep(np.real(self._x), np.real(self._y), **interpolation_args)
                    interp_func = lambda x: interpolate.splev(x,spl)
                else:
                    interp_func = interp1d(np.real(self._x), np.real(self._y), kind=interpolation)
                # walk up in steps of precision, until the curve comes back down
                xp = x[i+1]
                y_prev = interp_func(xp)
                y_next = interp_func(xp+2*x_precision)
                while y_next > y_prev and xp < x[i+2]:
                    xp += x_precision
                    y_prev = y_next
                    y_next = interp_func(xp+2*x_precision)
                res.append(xp+x_precision/2)
        return np.array(res)
    def local_minima(self,x_range=(None,None),x_precision=None,interpolation="spline",interpolation_args={"k":3}): 
        """ numerically find local minima of the curve
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values in which to sear for local minima 
                   None if values should be taken from the start/to the end
            x_precision(float): maximum difference between the returned value and the actual value in units of x data. if None: 1/1000 of the full x-range
                interpolation: see https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
                  "cubic": cubic spline 
                  "nearest":  value of nearest data point
                  "previous": value of previous data point
                  "next":     value of next data point
                  "spline":   spline interpolation of arbitrary order defined by interpolation_args: k
              interpolation_args: ignored for "cubic","nearest","previous","next" interpolations
                  for "spline" interpolation: passed to scipy.interpolate.splrep
                              most important: "k": order of the spline
        Returns:
            1D numpy array containing the x values of the local maxima
        """
        self._y *= -1
        res = self.local_maxima(x_range=x_range,x_precision=x_precision,interpolation=interpolation,interpolation_args=interpolation_args)
        self._y *= -1
        return res

    def envelope(self,x_range=(None,None),limit="upper",x_window=None):
        """ get envelope based on the curves local extrema. For noisy data, prior smoothing can lead to a significant speed up.
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values. 
                   None if values should be taken from the start/to the end
            limit(string): 'upper' or 'lower' envelope
            x_window(float): take the most exposed value on this x-scale. filters out small local extrema caused by noise. default: None - no filtering, just use all local extrema
        """
        if limit is "upper":
            xextrema = self.local_maxima(x_range=x_range)
            fcomp = lambda y: np.argmax(y)
        else:
            xextrema = self.local_minima(x_range=x_range)
            fcomp = lambda y:np.argmin(y)
        y = self.evaluate(xextrema)
        if x_window is not None: # filter maxima list
            i_xres =[]
            for _,x in enumerate(xextrema):
                pt_to_consider = np.array([j for j,pt in enumerate(xextrema) if np.abs(pt-x) < x_window/2]).astype(int)
                i_kept = pt_to_consider[fcomp(y[pt_to_consider])]
                if not i_kept in i_xres:
                    i_xres.append(i_kept)
            i_xres = np.array(i_xres)
            y = y[i_xres]
            xextrema = xextrema[i_xres]                   
        res = deepcopy(self)
        res._x = xextrema
        res._y = y
        return res
    
    def get_maximum(self,x_range=(None,None),interpolation="none",interpolation_args={},npoints=1000):
        """ numerically estimate the global maxmium in the given range
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values for which the maximum is searched 
                    None if values should be taken from the start/to the end of the curve
            interpolation(string):  "none": search for the maximum of the discrete data points without interpolation (default)
                                    "cubic": interpolate with cubic spline
                                    "spline": interpolate with spline of order k. k has to be given in interpolation_args
            npoints(int):  only relevant for interpolation. number of points, equally spread over the given range, for which the spline is evaluated on to find the maximum
        Returns:
            (x,y) of the numerically estimated maximum in the given range
        """
        x,y = self.get_x_y(x_range)
        res = [0,0]
        if interpolation in ["none","nearest","previous","next"]:
            x_eval = x
            y_eval = y
        elif interpolation in ["cubic","spline"]:
            if interpolation =="cubic":
                interpolation_args = {"k":3}
            spl = interpolate.splrep(x, y, **interpolation_args)
            x_eval = np.linspace(x[0],x[-1],npoints)
            y_eval = interpolate.splev(x_eval,spl)
        
        ind_max = np.argmax(y_eval)
        res[0] = x_eval[ind_max]
        res[1] = y_eval[ind_max]
        return res
            
    def get_minimum(self,x_range=(None,None),interpolation="none",interpolation_args={},npoints=1000):
        """ numerically estimate the minimum in the given range
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values for which the minimum is searched 
                    None if values should be taken from the start/to the end of the curve
            interpolation(string):  "none": search for the minimum of the discrete data points without interpolation (default)
                                    "cubic": interpolate with cubic spline
                                    "spline": interpolate with spline of order k. k has to be given in interpolation_args
            npoints(int):  only relevant for spline interpolation. number of points, equally spread over the given range, for which the spline is evaluated on to find the minimum
        Returns:
            (x,y) of the numerically estimated minimum in the given range
        """
        self._y *= -1
        res = self.get_maximum(x_range,interpolation,interpolation_args,npoints)
        self._y *= -1
        return (res[0],-1*res[1])

    def mean(self,x_range=(None,None)):
        """ Calculate the mean value of the curve over the given range.
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values for which the minimum is searched 
                    None if values should be taken from the start/to the end of the curve
        Returns:
            float: mean value
        """
        y = self.get_y(x_range)
        return np.mean(y)
    def std(self,x_range=(None,None)):
        """ Calculate the standard deviation of the curve over the given range.
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values for which the minimum is searched 
                    None if values should be taken from the start/to the end of the curve
        Returns:
            float: standard deviation
        """
        y = self.get_y(x_range)
        return np.std(y)
        
    def fit(self,function,x_range=(None,None),**kwargs):
        """fit the given function to the curve in the specified range.
        Based on scipy.optimize.curve_fit
        Arguments:
              function (callable: f(x, ..)): The model function, f(x, …). It must take the independent variable as the first argument 
                                              and the parameters to fit as separate remaining arguments. 
              x_range (tuple (x_start, x_stop)): Range in units of the x data in which the fit should be applied. 
                                                  None if the whole curve should be fitted
              kwargs:     Additional arguments passed to curve_fit
        Returns:
               poptarray
               Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
               pcov2d array
               The estimated covariance of popt. The diagonals provide the variance of the parameter estimate. To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
        How the sigma parameter affects the estimated covariance depends on absolute_sigma argument, as described above.
        If the Jacobian matrix at the solution doesn’t have a full rank, then ‘lm’ method returns a matrix filled with np.inf, on the other hand ‘trf’ and ‘dogbox’ methods use Moore-Penrose pseudoinverse to compute the covariance matrix.
        """
        x,y = self.get_x_y(x_range)
        return curve_fit(function,x,y,**kwargs)
    
    def FFT(self,x_range=(None,None), interpolate=True,**kwargs):
        """ FFT of the curve in the given x_range.
        Arguments:
            x_range: [lower_lim,upper_lim] in units of x values for which the FFT is calculated 
                    None if values should be taken from the start/to the end of the curve
            interpolate (bool): True: interpolate to obtain evenly spaced data; False: use data points as is, assuming they are evenly spaced
            kwargs (dict): interpolation arguments passed to Curve.evaluate
        Returns: 
            Curve object: new curve with complex amplitudes.
        """
        x_RS, y_RS = self._get_x_range_x_y(x_range)
        if interpolate:
            ll = x_RS[0] if x_range[0] is None else x_range[0]
            ul = x_RS[-1] if x_range[1] is None else x_range[1]
            x_RS = np.linspace(ll,ul,len(x_RS))
            y_RS = self.evaluate(x_RS,**kwargs)

        n_pt_y = len(y_RS)

        dt = x_RS[1] - x_RS[0]
        fa = 1.0/dt # scan frequency
        N = int(n_pt_y/2+1)
        x_FFT = np.linspace(0, fa/2, N, endpoint=True)
        y_FFT = np.fft.rfft(y_RS)/N
        ret = Curve(x=x_FFT,y=y_FFT)
        return ret 

    def IFFT(self):
        """ inverse FFT
        Returns:
            Curve object: inverse FFT. IFFT(FFT) = identity
        """
        x_FS, y_FS = (self._x,self._y)
        n_pt_y = len(y_FS)
        dt = x_FS[1] - x_FS[0]
        fa = 1.0/dt # scan frequency
        N = int((n_pt_y-1)*2)
        x_RS = np.linspace(0, fa, N, endpoint=True)
        y_RS = np.fft.irfft(y_FS)*N/2
        return Curve(x=x_RS,y=y_RS)
    def PSD(self, df,x_range=(None,None)):
        """ Power Spectral Density using welch method
        note: only works well for evenly spaced data. otherwise: interpolation - might cause some weird results
        Arguments:
            df (float) width per segment in fourier space
            x_range (float,float): range of the data to use for the PSD calculation. if None: maximum available range
        Returns:
            f, psd
        """
        x_RS, _ = self._get_x_range_x_y(x_range)
        x_spacing = x_RS[1]-x_RS[0] # use distance between first two data points as sampling freq
        x = np.linspace(x_RS[0],x_RS[-1],int((x_RS[-1]-x_RS[0])/x_spacing)+1)
        y = self.evaluate(x)
        Fs = 1/x_spacing
        f, psd = sig.welch(y,Fs,'hanning',int(Fs/df),return_onesided=0)
        return Curve(f,psd,y_label='PSD')

    def __add__(self,other):
        """ add number to all values or 
        add second curve to first point wise on the first curve's points 
        using default cubic spline interpolation"""
        if type(other)==float or type(other)==int:
            ret = deepcopy(self)
            ret._y += other
            return ret
        elif type(other)==Curve:
            ret = deepcopy(self)
            interpol_order=3
            if len(other._x) < 4:
                interpol_order = len(other._x)-1
            ret._y = self._y+other.evaluate(self._x,outside_range_value=other.default_add_outside_range_value,interpolation_args={"k":interpol_order})
            return ret
    def __sub__(self,other):
        """subtract number from all values or 
        subtract second curve from first point wise on the first curve's points 
        using default cubic spline interpolation"""
        if type(other)==float or type(other)==int:
            ret = deepcopy(self)
            ret._y -= other
            return ret
        elif type(other)==Curve:
            ret = deepcopy(self)
            interpol_order=3
            if len(other._x) < 4:
                interpol_order = len(other._x)-1
            ret._y = self._y-other.evaluate(self._x,outside_range_value=other.default_sub_outside_range_value,interpolation_args={"k":interpol_order})
            return ret
    def __mul__(self,other):
        """multiply all values with number or
        mutliply second curve with first point wise on the first curve's points 
        using default cubic spline interpolation"""
        if type(other)==float or type(other)==int:
            ret = deepcopy(self)
            ret._y *= other
            return ret
        elif type(other)==Curve:
            ret = deepcopy(self)
            interpol_order=3
            if len(other._x) < 4:
                interpol_order = len(other._x)-1
            ret._y = self._y*other.evaluate(self._x,outside_range_value=other.default_mul_outside_range_value,interpolation_args={"k":interpol_order})
            return ret
    __rmul__ = __mul__
    def __truediv__(self,other):
        """divide all values by number or
        divide first curve by second point wise on the first curve's points 
        using default cubic spline interpolation"""
        if type(other)==float or type(other)==int:
            ret = deepcopy(self)
            ret._y /= other
            return ret
        elif type(other)==Curve:
            ret = deepcopy(self)
            interpol_order=3
            if len(other._x) < 4:
                interpol_order = len(other._x)-1
            ret._y = self._y/other.evaluate(self._x,outside_range_value=other.default_div_outside_range_value,interpolation_args={"k":interpol_order})
            return ret
    def __pow__(self,other):
        """ calculate the power function using the given number as exponent or
        on the first curve's points, using second curve as exponent using default cubic spline interpolation"""
        if type(other)==float or type(other)==int:
            ret = deepcopy(self)
            ret._y = np.power(ret._y,other)
            return ret
        elif type(other)==Curve:
            ret = deepcopy(self)
            interpol_order=3
            if len(other._x) < 4:
                interpol_order = len(other._x)-1
            ret._y = np.power(ret._y,other.evaluate(self._x,outside_range_value=other.default_pow_outside_range_value,interpolation_args={"k":interpol_order}))
            return ret
            
    def apply_transformation(self,transformation):
        """ apply an arbitrary transformation to the data points
        Arguments:
            func transformation(x,y): 
                takes x,y numpy arrays of x and y values of the data points
                returns (x,y), tuple of two lists with the new x and y values of the data points
        Returns:
            Curve object: holding the new values. the object itself is changed as well"""
        xn, yn = transformation(self._x,self._y)
        if type(xn) is not np.ndarray or type(yn) is not np.ndarray:
            raise TypeError(f"x or y after transformation have to be numpy.ndarray but are {type(xn)} and {type(yn)}")
        elif len(xn) != len(yn):
            raise ValueError(f"The x and y data must have the same dimension but they are {len(xn)}(x) and {len(yn)}(y) instead!")
        self._x, self._y = xn,yn
        self._sort_x()
        return self

    def remove_data_points(self,condition):
        """ delete all datapoints that fulfill condition
        Arguments:
            func condition(x,y):
                takes x,y coordinates of the data point
                returns True or False; True for deleting the point, False for keeping it
        Returns:
            Curve object: holding the new values. the object itself is changed as well"""
        indices = [i for i,x in enumerate(self._x) if condition(x,self._y[i])]
        self._x=np.delete(self._x,indices)
        self._y=np.delete(self._y,indices)
        return self
    def append(self,c):
        """ append another curve to the data. if there is an overlap: take data of the first curve
        Arumgents:
            c(Curve): the curve to be appended
        Returns: 
            Curve: the new, concatenated curve; the old curve gets changed as well!
        """
        i = np.where(c._x > np.max(self._x)) # indices of c._x that will be appended
        self._x = np.concatenate((self._x,c._x[i]))
        self._y = np.concatenate((self._y,c._y[i]))
        return self


    def smoothen(self,filtering, filtering_parameter,x_range=(None,None)):
        """
        smooth data, using the algorithm specified by filtering.
        
        Arguments:
            filtering:      following possibilities:
                            "": no filtering
                            moving average:
                                filtering_parameter: size of the window for the moving average
                            Gauss:
                                filtering_parameter[0]: Standard deviation for Gaussian kernel
                                filtering_parameter[1]: order of the gaussian filter (int)
                            Savitzky-Golay:
                                filtering_parameter[0]: window_length (int)
                                    The length of the filter window (i.e. the number of coefficients).
                                    window_length must be a positive odd integer.
                                filtering_parameter[1]: polyorder (int)
                                    The order of the polynomial used to fit the samples. polyorder must be less than window_length.
                            lowess: smoothing using local linear estimates, following
                                Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
                                filtering_parameter: Between 0 and 1. The fraction of the data used when estimating each data-value.
                                x values have to be provided!
            filtering_paratmeter: see filtering
            x_range (float,float): range of the data to use for the PSD calculation. if None: maximum available range
        Returns:
            Curve object: holding the new values. the object itself is changed as well
        """
        x,data = self.get_x_y(x_range)
        if filtering == "moving average":
            data = moving_average(data, filtering_parameter)
            lnew = int(len(x)-filtering_parameter+1) # length of the new array
            x = np.add(x[0:lnew],
                            (x[1]-x[0])*(filtering_parameter-1)/2*np.ones(lnew)) # this assumes linearly spaced z!
        elif filtering == "Gauss":
            # documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
            from scipy.ndimage import gaussian_filter
            data = gaussian_filter(data, order=filtering_parameter[1], sigma=filtering_parameter[0])
        elif filtering == "Savitzky-Golay":
            # documentation: https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html
            from scipy.signal import savgol_filter
            data = savgol_filter(data, filtering_parameter[0], filtering_parameter[1])
        elif filtering == "lowess":
            # documentation: http://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
            from statsmodels.nonparametric.smoothers_lowess import lowess
            data = lowess(data,x,frac=filtering_parameter).transpose()[1]
        self._y[np.argwhere(self._x==x[0])[0][0]:
                np.argwhere(self._x==x[-1])[0][0]+1] = data
        return self

    def smoothed(self,filtering, filtering_parameter,x_range=(None,None)):
        """ get a copy of the curve, smoothed by the specified algorithm in the given range
        Arguments:
            filtering:      following possibilities:
                            "": no filtering
                            moving average:
                                filtering_parameter: size of the window for the moving average
                            Gauss:
                                filtering_parameter[0]: Standard deviation for Gaussian kernel
                                filtering_parameter[1]: order of the gaussian filter (int)
                            Savitzky-Golay:
                                filtering_parameter[0]: window_length (int)
                                    The length of the filter window (i.e. the number of coefficients).
                                    window_length must be a positive odd integer.
                                filtering_parameter[1]: polyorder (int)
                                    The order of the polynomial used to fit the samples. polyorder must be less than window_length.
                            lowess: smoothing using local linear estimates, following
                                Cleveland, W.S. (1979) “Robust Locally Weighted Regression and Smoothing Scatterplots”. Journal of the American Statistical Association 74 (368): 829-836.
                                filtering_parameter: Between 0 and 1. The fraction of the data used when estimating each data-value.
                                x values have to be provided!
            filtering_paratmeter: see filtering
            x_range (float,float): range of the data to use for the PSD calculation. if None: maximum available range
        Returns:
            Curve object: holding the new values. the original object stays unchanged
        """
        ret = deepcopy(self)
        ret.smoothen(filtering,filtering_parameter,x_range)
        return ret

    def derivative(self,x_range=(None,None)):
        """ calculate the numerical derivative of the curve using numpy.gradient
        Arguments:
            x_range (float,float): range of the data to use for the calculation. if None: maximum available range
        Return:
            Curve object: holding the derivative"""
        x,y = self.get_x_y(x_range)
        ret = deepcopy(self)
        ret._x = x
        ret._y = np.gradient(y,x)
        return ret
    def antiderivative(self,x_range=(None,None)):
        """ calculate the antiderivative of the curve using trapezoidal integration
        Arguments:
            x_range (float,float): range of the data to use for the calculation. if None: maximum available range
        Return:
            Curve object: holding the antiderivative"""
        self._sort_x() # make sure x values are in ascending order
        x,y = self.get_x_y(x_range)
        res = np.zeros(len(x))
        for i in range(len(x)-1):
            res[i+1] = res[i] + (x[i+1]-x[i])*0.5*(y[i]+y[i+1])
        ret = deepcopy(self)
        ret._x = x
        ret._y = res
        return ret
    def integrate(self,x_range=(None,None),interpolation="spline",interpolation_args={"k":3}):
        """ Numerically integrate the curve over the given x_range using trapezoidal integration.
        Interpolate for start and end using given interpolation (default: cubic spline) 
        Arguements:
            x_range (float,float): range over which to integrate in units of x data
            interpolation: see https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
               "cubic": cubic spline 
               "nearest":  value of nearest data point
               "previous": value of previous data point
               "next":     value of next data point
               "spline":   spline interpolation of arbitrary order defined by interpolation_args: k
            interpolation_args: ignored for "cubic","nearest","previous","next" interpolations
                                for "spline" interpolation: passed to scipy.interpolate.splrep
                                most important: "k": order of the spline
        Returns:
            Integral (float)
        
        """
        ad = self.antiderivative()
        xmin = x_range[0] if x_range[0] is not None else self._x[0]
        xmax = x_range[1] if x_range[1] is not None else self._x[-1]
        return (ad.evaluate(xmax,interpolation,interpolation_args)-
                ad.evaluate(xmin,interpolation,interpolation_args))
    
    def convoluted_with(self,g,x_resolution,integration_factor=1,x_range=(None,None),interpolation='linear'):
        """ Convolute the first curve (f) with another given curve (g).
        (f*g)(x) = int f(t)g(t-x) dt
        Arguments:
            g (Curve object): curve to be convoluted with. return self*curve
            x_resolution (float): spacing of x data points of the convolution
            integration_factor (int): how much denser x points should be spaced for the integration
            x_range (float,float): range on which the convolution is calculated. by default only areas of full overlap
            interpolation (string): interpolation method used. default: linear. for details see evaluate
        Returns:
            Convolution (f*g) (Curve object)
        """
        # calculate "conv" = f*g; f = self, g = curve
        f_x_min = self._x[0]
        f_x_max = self._x[-1]

        g_x_min = g._x[0]
        g_x_max = g._x[-1]

        conv_x_min = x_range[0] if x_range[0] is not None else f_x_min-g_x_min
        conv_x_max = x_range[1] if x_range[1] is not None else f_x_max-g_x_max
        conv_n_pts = int((conv_x_max-conv_x_min)/x_resolution)
        conv_x_center = (conv_x_max+conv_x_min)/2
        conv_x_min = conv_x_center-conv_n_pts/2*x_resolution
        conv_x_max = conv_x_center+conv_n_pts/2*x_resolution
        
        offset_from_minimum = (conv_x_min-(f_x_min-g_x_min))/2
        f_x_min += offset_from_minimum
        f_x_max -= offset_from_minimum
        g_x_min += offset_from_minimum
        g_x_max -= offset_from_minimum

        conv_x = np.arange(conv_x_min,conv_x_max+x_resolution*0.1,x_resolution)
        int_res = x_resolution/integration_factor
        f_x = np.arange(f_x_min,f_x_max+int_res*0.1,int_res)
        g_x = np.arange(g_x_min,g_x_max+int_res*0.1,int_res)

        f_y = self.evaluate(f_x,interpolation=interpolation)
        g_y = g.evaluate(g_x,interpolation=interpolation)

        # convolution
        int_indices = np.arange(0,conv_n_pts+1)*integration_factor
        conv_y = compiled_convolution(f_y,g_y,int_res,int_indices) 

        ret = deepcopy(self)
        ret.set_data(x=conv_x,y=conv_y)
        return ret
