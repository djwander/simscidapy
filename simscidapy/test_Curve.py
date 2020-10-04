"""
unit tests for Curve.py
"""
import pytest
from Curve import Curve
from Curve import default_csv_header_parser
import numpy as np
import os

### Setters
def test_set_data():
    # no data given
    c = Curve()
    assert np.array_equal(c.get_x(),np.array([]))
    assert np.array_equal(c.get_y(),np.array([]))
    # x,y given 
    c.set_data(x=[3,2,4],y=[3,2,4])
    assert np.array_equal(c.get_x(),np.array([2,3,4])) # make sure data gets sorted in ascending order as well
    assert np.array_equal(c.get_y(),np.array([2,3,4]))
    # xy given
    c.set_data(xy=[(2,5),(5,4),(1,3)])
    assert np.array_equal(c.get_x(),np.array([1,2,5]))
    assert np.array_equal(c.get_y(),np.array([3,5,4]))
    del c

def test_set_plot_properties():
    c = Curve()
    c.set_plot_properties(props={
        "plot_args":{'color':'red'},
        "title":"title",
        "x_label":"x_label",
        "y_label":"y_label",
        "x_unit":"x_unit",
        "y_unit":"y_unit",
        "x_scale":"x_scale",
        "y_scale":"y_scale",
        "not defined param":"not defined"
    })
    assert c._plot_args is {'color':'red'}
    assert c._title is "title"
    assert c._x_label is "x_label"
    assert c._y_label is "y_label"
    assert c._x_unit is "x_unit"
    assert c._y_unit is "y_unit"
    assert c._x_scale is "x_scale"
    assert c._y_scale is "y_scale"
    # just change one parameter
    c.set_plot_properties(props={'title':'new title'})
    assert c._title is 'new title'
    assert c._x_label is 'x_label' # other properties don't change
    del c

def test_sort_x():
    c = Curve()
    c._x = np.array([2,1,3])
    c._y = c._x
    c._sort_x()
    assert np.array_equal(c._x,np.array([1,2,3]))
    assert np.array_equal(c._y,np.array([1,2,3]))
    c1 = Curve()
    c1._x = np.array([2,1,1,3])
    c1._y = np.array([2,0,1,3])
    c1._sort_x()
    assert np.array_equal(c1._x,np.array([1,1,2,3]))
    assert np.array_equal(c1._y,np.array([0,1,2,3]))

### Getters
def test_get_x_range_x_y():
    x = np.linspace(1,10,10)
    c1 = Curve(x,x)
    # no limits
    x1,y1 = c1._get_x_range_x_y()
    assert np.array_equal(x1,x)
    assert np.array_equal(y1,x)
    # only upper limit
    x2,y2 = c1._get_x_range_x_y(x_range=(None,5.5))
    assert np.array_equal(x2,np.linspace(1,5,5))
    assert np.array_equal(y2,np.linspace(1,5,5))
    # only lower limit
    x3,y3 = c1._get_x_range_x_y(x_range=(5.5,None))
    assert np.array_equal(x3,np.linspace(6,10,5))
    assert np.array_equal(y3,np.linspace(6,10,5))
    # both limits
    x4,y4 = c1._get_x_range_x_y(x_range=(2.1,5))
    assert np.array_equal(x4,np.array([3,4,5]))
    assert np.array_equal(y4,np.array([3,4,5]))
    # lower limit > biggest value
    x5,y5 = c1._get_x_range_x_y(x_range=(11,None))
    assert np.array_equal(x5,np.array([]))
    assert np.array_equal(y5,np.array([]))
    # upper limit < smallest value
    x6,y6 = c1._get_x_range_x_y(x_range=(None,-2))
    assert np.array_equal(x6,np.array([]))
    assert np.array_equal(y6,np.array([]))

def test_get_x():
    x = np.linspace(1,10,10)
    c1 = Curve(x,x)
    # no limits
    x1 = c1.get_x()
    assert np.array_equal(x1,x)
    # only upper limit
    x2 = c1.get_x(x_range=(None,5.5))
    assert np.array_equal(x2,np.linspace(1,5,5))
    # only lower limit
    x3 = c1.get_x(x_range=(5.5,None))
    assert np.array_equal(x3,np.linspace(6,10,5))
    # both limits
    x4 = c1.get_x(x_range=(2.1,5))
    assert np.array_equal(x4,np.array([3,4,5]))
    # lower limit > biggest value
    x5 = c1.get_x(x_range=(11,None))
    assert np.array_equal(x5,np.array([]))
    # upper limit < smallest value
    x6 = c1.get_x(x_range=(None,-2))
    assert np.array_equal(x6,np.array([]))

def test_get_y():
    y = np.linspace(1,10,10)
    c1 = Curve(y,y)
    # no limits
    y1 = c1.get_y()
    assert np.array_equal(y1,y)
    # only upper limit
    y2 = c1.get_y(x_range=(None,5.5))
    assert np.array_equal(y2,np.linspace(1,5,5))
    # only lower limit
    y3 = c1.get_y(x_range=(5.5,None))
    assert np.array_equal(y3,np.linspace(6,10,5))
    # both limits
    y4 = c1.get_y(x_range=(2.1,5))
    assert np.array_equal(y4,np.array([3,4,5]))
    # lower limit > biggest value
    y5 = c1.get_y(x_range=(11,None))
    assert np.array_equal(y5,np.array([]))
    # upper limit < smallest value
    y6 = c1.get_y(x_range=(None,-2))
    assert np.array_equal(y6,np.array([]))

def test_get_x_y():
    x = np.linspace(1,10,10)
    c1 = Curve(x,x)
    # no limits
    x1,y1 = c1.get_x_y()
    assert np.array_equal(x1,x)
    assert np.array_equal(y1,x)
    # only upper limit
    x2,y2 = c1.get_x_y(x_range=(None,5.5))
    assert np.array_equal(x2,np.linspace(1,5,5))
    assert np.array_equal(y2,np.linspace(1,5,5))
    # only lower limit
    x3,y3 = c1.get_x_y(x_range=(5.5,None))
    assert np.array_equal(x3,np.linspace(6,10,5))
    assert np.array_equal(y3,np.linspace(6,10,5))
    # both limits
    x4,y4 = c1.get_x_y(x_range=(2.1,5))
    assert np.array_equal(x4,np.array([3,4,5]))
    assert np.array_equal(y4,np.array([3,4,5]))
    # lower limit > biggest value
    x5,y5 = c1.get_x_y(x_range=(11,None))
    assert np.array_equal(x5,np.array([]))
    assert np.array_equal(y5,np.array([]))
    # upper limit < smallest value
    x6,y6 = c1.get_x_y(x_range=(None,-2))
    assert np.array_equal(x6,np.array([]))
    assert np.array_equal(y6,np.array([]))

def test_get_xy():
    x = np.linspace(1,3,3)
    c1 = Curve(x,x)
    # no limits
    pts1 = c1.get_xy()
    assert np.array_equal(pts1,np.array([[1,1],[2,2],[3,3]]))
    # only upper limit
    pts2 = c1.get_xy(x_range=(None,2.5))
    assert np.array_equal(pts2,np.array([[1,1],[2,2]]))
    # only lower limit
    pts3 = c1.get_xy(x_range=(1.23,None))
    assert np.array_equal(pts3,np.array([[2,2],[3,3]]))
    # both limits
    pts4 = c1.get_xy(x_range=(1.1,2.4))
    assert np.array_equal(pts4,np.array([[2,2]]))
    # lower limit > biggest value
    pts5 = c1.get_xy(x_range=(11,None))
    assert np.array_equal(pts5,np.zeros((0,2)))
    # upper limit < smallest value
    pts6 = c1.get_xy(x_range=(None,-2))
    assert np.array_equal(pts6,np.zeros((0,2)))

### Data IO
def test_default_csv_header_parser():
    header = ["# xtest(xunit);ytest(yunit)"]
    parsed = default_csv_header_parser(header)
    assert parsed['x_label'] == 'xtest'
    assert parsed['x_unit'] == 'xunit'
    assert parsed['y_label'] == 'ytest'
    assert parsed['y_unit'] == 'yunit'

def test_csv_io():
    x = np.array([0.1234567,1.1234567e15])
    c = Curve(x,2*x,x_label="xTest",x_unit="xUnit",y_label="yTest",y_unit="yUnit")
    # using default arguments
    c.save_to_csv("CSV-Test.csv")
    c2 = Curve()
    c2.load_from_csv('CSV-Test.csv')
    eps=1e-8
    assert np.max(np.abs(c2._x-np.array([0.12346,1.1235e15])))<eps # default precision for saving: 5 significant (note: rounding)
    assert np.max(np.abs(c2._y-np.array([0.24691,2.2469e15])))<eps # default precision for saving: 5 significant (note: rounding)
    assert c2._x_label == "xTest" # 
    assert c2._x_unit == "xUnit"
    assert c2._y_label == "yTest" # 
    assert c2._y_unit == "yUnit"
    # no parsing of header (if csv from other code and/or no need to parse)
    c3 = Curve()
    c3.load_from_csv('CSV-Test.csv',header_parser=lambda h,d: {})
    assert np.max(np.abs(c3._x-np.array([0.12346,1.1235e15])))<eps # default precision for saving: 5 significant (note: rounding)
    assert np.max(np.abs(c3._y-np.array([0.24691,2.2469e15])))<eps # default precision for saving: 5 significant (note: rounding)
    assert c3._x_label == "" 
    assert c3._x_unit == None
    assert c3._y_label == "" 
    assert c3._y_unit == None
    # using different order of columns: 1st y, 2nd x
    c4 = Curve()
    c4.load_from_csv('CSV-Test.csv',columns=(1,0))
    assert np.max(np.abs(c4._x-np.array([0.24691,2.2469e15])))<eps 
    assert np.max(np.abs(c4._y-np.array([0.12346,1.1235e15])))<eps 
    
    # using a different delimiter and different precision
    c.save_to_csv("CSV-Test.csv",delimiter='\t',fmt='%.1e')
    c5 = Curve()
    c5.load_from_csv('CSV-Test.csv',delimiter='\t')
    assert np.max(np.abs(c5._x-np.array([0.12,1.1e15])))<eps # default precision for saving: 5 significant (note: rounding)
    assert np.max(np.abs(c5._y-np.array([0.25,2.2e15])))<eps

def test_pickle_io():
    x = np.array([1,2,3.5e10])
    c1 = Curve(x,x*2,x_label='pickle test',plot_args={'color':'red'},x_scale='log')
    c1.random_userdefined_variable = 5
    c1.pickle_save('test-pickle.cur')
    del c1
    c2 = Curve().pickle_load('test-pickle.cur')
    assert np.array_equal(c2._x,x)
    assert np.array_equal(c2._y,x*2)
    assert c2._x_label == 'pickle test'
    assert c2._plot_args == {'color':'red'}
    assert c2._x_scale == "log"
    assert c2.random_userdefined_variable == 5
    os.remove("test-pickle.cur")


### Data treatment
def test_crop():
    x = np.linspace(1,10,10)
    # standard usage defining both limits
    c1 = Curve(x,x)
    cc = c1.crop((2.5,6))
    assert np.array_equal(c1.get_x(),np.linspace(3,6,4)) # cc is properly cropped c1
    assert np.array_equal(c1.get_y(),np.linspace(3,6,4))
    assert np.array_equal(c1.get_x(),np.linspace(3,6,4)) # but c1 got modified as well
    assert np.array_equal(c1.get_y(),np.linspace(3,6,4)) # -> cc = c1
    # standard usage defining upper limit only
    c2 = Curve(x,x)
    c2.crop((None,6))
    assert np.array_equal(c2.get_x(),np.linspace(1,6,6))
    assert np.array_equal(c2.get_y(),np.linspace(1,6,6))
    # standard usage defining lower limit only
    c3 = Curve(x,x)
    c3.crop((2.2,None))
    assert np.array_equal(c3.get_x(),np.linspace(3,10,8))
    assert np.array_equal(c3.get_y(),np.linspace(3,10,8))
    # crop empty range
    c4 = Curve(x,x)
    c4.crop((2.5,2.7))
    assert np.array_equal(c4.get_x(),np.array([]))
    assert np.array_equal(c4.get_y(),np.array([]))
    # range outside defined data
    c5 = Curve(x,x)
    c5.crop((-5,-0.4))
    assert np.array_equal(c5.get_x(),np.array([]))
    assert np.array_equal(c5.get_y(),np.array([]))
    # range outside defined data
    c6 = Curve(x,x)
    c6.crop((20,26))
    assert np.array_equal(c6.get_x(),np.array([]))
    assert np.array_equal(c6.get_y(),np.array([]))

def test_cropped():
    x = np.linspace(1,10,10)
    # standard usage defining both limits
    c1 = Curve(x,x)
    c2 = c1.cropped((2.5,6))
    assert np.array_equal(c2.get_x(),np.linspace(3,6,4))
    assert np.array_equal(c2.get_y(),np.linspace(3,6,4))
    # here c1 is unchanged (opposed to the behaviour of crop)
    assert np.array_equal(c1.get_x(),np.linspace(1,10,10))
    assert np.array_equal(c1.get_y(),np.linspace(1,10,10))
    # rest already tested in test_crop

def test_evaluate():
    x = np.linspace(1,10,10)
    y = x
    c1 = Curve(x=x,y=y)
    eps = 1e-5
    # evaluation on data points
    assert np.max(np.abs(c1.evaluate(x)-y))<eps 
    # interpolation
    assert (c1.evaluate(4.5)-4.5)<eps # using standard cubic spline interpolation
    assert (c1.evaluate(4.3,interpolation='nearest')-4)<eps # using nearest interpolation
    assert (c1.evaluate(4.5,interpolation='nearest')-5)<eps # using nearest interpolation
    assert (c1.evaluate(4.9,interpolation='previous')-4)<eps # using previous interpolation
    assert (c1.evaluate(4.01,interpolation='next')-5)<eps # using next interpolation
    c2 = Curve(x=x,y=x**3)
    assert (c2.evaluate(4.5)-4.5**3)<eps # using standard cubic spline interpolation
    assert (c2.evaluate(4.5,interpolation='spline',interpolation_args={'k':1})-(4**3+5**3)/2)<eps # using linear interpolation
    # extrapolation
    assert (c2.evaluate(-1)+1)<eps # extrapolation: default as intrapolation -> cubic slpine
    assert (c2.evaluate(-1,outside_range_value=0)==0)
    assert np.max(np.abs(c2.evaluate([-1,5],outside_range_value=0)-np.array([0,5**3]))) < eps
    # evaluating at a scalar value
    assert c1.evaluate(1) == 1

def test_where():
    x = np.linspace(1,10,10)
    y = (x-5)**2
    c1 = Curve(x,y)
    pts1 = c1.where(1)
    assert np.array_equal(pts1,np.array[4,6])
    # cubic interpolation
    pts2 = c1.where(2) # default interpolation uses cubic spline -> gives a precise result: default precision: full_xrange/1000 = 1/100
    eps = 0.01
    assert np.max(np.abs(pts2-[5-np.sqrt(2),5+np.sqrt(2)])) < eps
    # higher precision
    eps = 1e-5
    pts3 = c1.where(2,x_precision=eps)
    assert np.max(np.abs(pts3-[5-np.sqrt(2),5+np.sqrt(2)])) < eps
    # other interpolation: nearest
    pts4 = c1.where(3.5,x_precision=eps,interpolation='nearest')
    assert np.array_equal(pts4,np.array([3,7]))
    # point not on curve
    pts5 = c1.where(-1)
    assert np.array_equal(pts5,np.array([]))
def test_local_maxima():
    # x = np.linspace(1,10,1000)
    # y = np.cos(2*np.pi*x)
    # c = Curve(x,y)
    # eps = 0.01
    # max1 = c.local_maxima(x_precision=eps)
    # assert np.max(np.abs(max1-np.linspace(2,9,8))) < eps
    
    y = np.array([3,3,1,2,1,1,2,2,3,4,4,4,4,1,3,3,3])
    x = np.arange(len(y))
    c = Curve(x,y)
    m = c.local_maxima(interpolation='none')
    assert np.array_equal(m,np.array([3,10.5]))
    m2 = c.local_maxima(interpolation='previous')
    assert np.array_equal(m2,np.array([3,10]))
    m3 = c.local_maxima(interpolation='next')
    assert np.array_equal(m3,np.array([3,11]))
    # test interpolation
    x = np.linspace(1,10,10)
    y = -(x-4.5)**2
    c2 = Curve(x,y)
    m4 = c2.local_maxima(interpolation='cubic',x_precision=0.01)
    assert m4[0]-4.5 < 0.01
    m5 = c2.local_maxima(interpolation='spline',interpolation_args={'k':2},x_precision=0.01)
    assert m5[0]-4.5 < 0.01
    


def test_local_minima():
    y = -1*np.array([3,3,1,2,1,1,2,2,3,4,4,4,4,1,3,3,3])
    x = np.arange(len(y))
    c = Curve(x,y)
    m = c.local_minima(interpolation='none')
    assert np.array_equal(m,np.array([3,10.5]))
    m2 = c.local_minima(interpolation='previous')
    assert np.array_equal(m2,np.array([3,10]))
    m3 = c.local_minima(interpolation='next')
    assert np.array_equal(m3,np.array([3,11]))
    # test interpolation
    x = np.linspace(1,10,10)
    y = (x-4.5)**2
    c2 = Curve(x,y)
    m4 = c2.local_minima(interpolation='cubic',x_precision=0.01)
    assert m4[0]-4.5 < 0.01
    m5 = c2.local_minima(interpolation='spline',interpolation_args={'k':2},x_precision=0.01)
    assert m5[0]-4.5 < 0.01

def test_envelope():
    x = np.linspace(1,10,1000)
    y = np.cos(2*np.pi*x)
    c = Curve(x,y)
    eps = 0.01
    env_up = c.envelope()
    assert np.max(np.abs(env_up.evaluate(x)-1)) < eps # upper envelope of a cos is a constant at +1
    env_low = c.envelope(limit='lower')
    assert np.max(np.abs(env_low.evaluate(x)+1)) < eps # lower envelope of a cos is a constant at -1
    ## could use some more testing e.g. for x_window

def test_get_maximum():
    x = np.linspace(1,10,10)+0.1
    y = -(x-5)**2
    c = Curve(x,y)
    # no interpolation
    m1 = c.get_maximum() 
    assert m1[0]==5.1 and np.abs(m1[1]+0.1**2)<1e-5
    # cubic interpolation to get a more precise result for smooth curves
    m2 = c.get_maximum(interpolation='cubic') # precision can be tuned via npoitns
    assert m2[0]-5<0.01

def test_get_minimum():
    x = np.linspace(1,10,10)+0.1
    y = (x-5)**2
    c = Curve(x,y)
    # no interpolation
    m1 = c.get_minimum() 
    assert m1[0]==5.1 and np.abs(m1[1]-0.1**2) < 1e-5
    # cubic interpolation to get a more precise result for smooth curves
    m2 = c.get_minimum(interpolation='cubic') # precision can be tuned via npoitns
    assert m2[0]-5<0.01

def test_mean():
    x = np.linspace(1,10,10)
    c = Curve(x,x)
    m = c.mean()
    assert m==5.5
    m2 = c.mean(x_range=(3,6)) # mean of 3,4,5,6
    assert m2==4.5
    c2 = c*2
    m3 = c2.mean()
    assert m3==11

def test_std():
    x = np.linspace(1,3,3)
    c = Curve(x,x)
    s = c.std()
    assert np.abs(s-np.sqrt(2/3))<1e-5

def test_fit():
    def lin(x,a,b):
        return a*x+b
    x = np.linspace(1,10,10)
    c1 = Curve(x,x)
    popt,pcov = c1.fit(lin)
    assert np.abs(popt[0]-1)<1e-5 and popt[1]<1e-5
    c2 = Curve(x,3*x+5)
    popt,pcov = c2.fit(lin)
    assert np.abs(popt[0]-3)<1e-5 and np.abs(popt[1]-5)<1e-5

def test_FFT():
    x = np.linspace(1,100,10000)
    y = np.cos(2*np.pi*x)+2*np.cos(2*np.pi*3*x)
    c = Curve(x,y)
    fft = c.FFT()
    peaks = fft.local_maxima(x_precision=0.001)
    assert np.max(np.abs(peaks-np.array([1,3])))<0.001 # frequencies are 1 and 3
    assert np.abs(fft.evaluate(peaks[0])-1) < 0.1 # amplitude at 1: 1
    assert np.abs(fft.evaluate(peaks[1])-2) < 0.1 # amplitude at 3: 2

    # same with unevenly spaced data points, making use of interpolate=True
    x = np.random.rand(10000)*100 
    y = np.cos(2*np.pi*x)+2*np.cos(2*np.pi*3*x)
    c = Curve(x,y)
    fft = c.FFT()
    fft.smoothen(filtering="Gauss",filtering_parameter=[3,1])
    peaks = fft.local_maxima(x_precision=0.001)
    assert np.max(np.abs(peaks-np.array([1,3])))<0.01 # frequencies are 1 and 3
    assert np.abs(fft.evaluate(peaks[0])-1) < 0.2 # amplitude at 1: 1
    assert np.abs(fft.evaluate(peaks[1])-2) < 0.2 # amplitude at 3: 2

def test_IFFT():
    x = np.linspace(1,100,10000)
    y = np.cos(2*np.pi*(x-0.123))+2*np.cos(2*np.pi*3*x)
    c = Curve(x,y)
    fft = c.FFT()
    ifft = fft.IFFT()
    eps = 0.001
    assert np.max(np.abs(ifft.get_y()-y))<eps # ifft is the inverse of FFT -> IFFT(FFT) = 1
def test_PSD():
    # TODO implement
    pass
def test_add():
    # same x values
    x1 = np.array([1,2,3])
    c1 = Curve(x1,x1)
    c2 = Curve(x1,-1*x1)
    c3 = c1 + c2
    assert np.array_equal(c3.get_y(),np.array([0,0,0]))
    # different x values
    x2 = np.array([0,1,2,3])
    x3 = np.array([1.5,2.5,3.5]) 
    c4 = Curve(x2,x2)
    c5 = Curve(x3,-1*x3)
    c6 = c4 + c5
    assert np.array_equal(c6.get_x(),c4.get_x()) # the new curve is defined on the x values of the first of the two added curves
    assert not np.array_equal(c6.get_x(),c5.get_x()) 
    assert np.max(np.abs(c6.get_y()-np.array([0,0,0,0])))<1e-5 # we added y=+x and y=-x -> y=0 even if the curves are not defined on the same x values
    # defining a different behaviour for extrapolation
    x4 = np.linspace(1,10,10)
    c7 = Curve(x4,2*x4)
    x5 = np.linspace(3,7,10)
    c8 = Curve(x5,-1*(x5-5)**2+4)
    c9 = c7 + c8 # default extrapolation (cubic spline)
    assert np.max(np.abs(c9.get_y()-(2*x4+4-1*(x4-5)**2)))<1e-5
    c8.default_add_outside_range_value = 0  # all values outside range set to 0
    c10 = c7 + c8 
    assert np.max(np.abs(c10.get_y()-np.array([0,0,0,3,4,3,0,0,0,0])-2*x4))<1e-5
    # simpler example
    x = np.array([1,2,3,4,5])
    c11 = Curve(x,x/x)
    c12 = Curve(np.array([2,4]),np.array([1,1]))
    c12.default_add_outside_range_value = 5
    c13 = c11+c12
    assert np.array_equal(c13.get_y(),np.array([6,2,2,2,6]))

def test_sub():
    # same x values
    x1 = np.array([1,2,3])
    c1 = Curve(x1,x1)
    c2 = Curve(x1,x1)
    c3 = c1 - c2
    assert np.array_equal(c3.get_y(),np.array([0,0,0]))
    # different x values
    x2 = np.array([0,1,2,3])
    x3 = np.array([1.5,2.5,3.5]) 
    c4 = Curve(x2,x2)
    c5 = Curve(x3,x3)
    c6 = c4 - c5
    assert np.array_equal(c6.get_x(),c4.get_x()) # the new curve is defined on the x values of the first of the two old curves
    assert not np.array_equal(c6.get_x(),c5.get_x()) 
    assert np.max(np.abs(c6.get_y()-np.array([0,0,0,0])))<1e-5 # we subtracted y=+x from y=x -> y=0 even if the curves are not defined on the same x values
    # defining a different behaviour for extrapolation
    x4 = np.linspace(1,10,10)
    c7 = Curve(x4,2*x4)
    x5 = np.linspace(3,7,10)
    c8 = Curve(x5,(x5-5)**2-4)
    c9 = c7 - c8 # default extrapolation (cubic spline)
    assert np.max(np.abs(c9.get_y()-(2*x4+4-(x4-5)**2)))<1e-5
    c8.default_sub_outside_range_value = 0  # all values outside range set to 0
    c10 = c7 - c8 
    assert np.max(np.abs(c10.get_y()-np.array([0,0,0,3,4,3,0,0,0,0])-2*x4))<1e-5
    # simpler example
    x = np.array([1,2,3,4,5])
    c11 = Curve(x,x/x)
    c12 = Curve(np.array([2,4]),np.array([1,1]))
    c12.default_sub_outside_range_value = 5
    c13 = c11-c12
    assert np.array_equal(c13.get_y(),np.array([-4,0,0,0,-4]))

def test_mul():
    x = np.linspace(1,10,10)
    c1 = Curve(x,x)
    c2 = Curve(x,x)
    c3 = c1*c2
    eps = 1e-5
    assert np.max(np.abs(c3.get_y()-x**2)) < eps
    # setting outside_range value: simple way of masking areas of a curve
    c4 = Curve(np.array([3,5]),np.array([1,1])) 
    c4.default_mul_outside_range_value = 0 
    c5 = c1*c4
    assert np.array_equal(c5.get_y(),np.array([0,0,3,4,5,0,0,0,0,0]))

def test_div():
    x = np.linspace(1,10,10)
    c1 = Curve(x,x)
    c2 = Curve(x,x)
    c3 = c1/c2
    eps = 1e-5
    assert np.max(np.abs(c3.get_y()-x/x)) < eps
    # setting outside_range value: simple way of masking areas of a curve
    c4 = Curve(np.array([3,5.3]),np.array([3,5.3])) 
    c4.default_mul_outside_range_value = 1 
    c5 = c1*c4
    assert np.array_equal(c5.get_y(),np.array([1,2,9,16,25,6,7,8,9,10]))

def test_pow():
    x = np.linspace(1,10,10)
    c1 = Curve(x,2*np.ones(10))
    c2 = Curve(x,x)
    c3 = c1**c2
    eps = 1e-5
    assert np.max(np.abs(c3.get_y()-2**x)) < eps
    # setting outside_range value: simple way of masking areas of a curve
    c4 = Curve(np.array([3,5.3]),np.array([3,5.3])) 
    c4.default_pow_outside_range_value = 0
    c5 = c1**c4
    assert np.array_equal(c5.get_y(),np.array([1,1,8,16,32,1,1,1,1,1]))

def test_apply_transformation():
    x = np.array([1,10,100,1000])
    y = np.array([0,1,2,3])
    c1 = Curve(x,y)
    # classic use: go from log to linear scaling
    c2 = c1.apply_transformation(lambda x,y: (np.log10(x),y))
    assert np.array_equal(c2.get_x(),np.array([0,1,2,3]))
    assert np.array_equal(c1.get_x(),np.array([0,1,2,3])) # note: the original curve object is changed as well
    # after applying transformation, data points are sorted in ascending order again
    x = np.array([1,2,3])
    c3 = Curve(x,x)
    c4 = c3.apply_transformation(lambda x,y: (-x,y))
    assert np.array_equal(c4.get_x(),np.array([-3,-2,-1]))
    assert np.array_equal(c4.get_y(),np.array([3,2,1]))
    # x and y have to be of same dimension
    c5 = Curve(x,x)
    with pytest.raises(TypeError):
        c5.apply_transformation(lambda x,y: (x[0],y)) # x[0] is a np.float -> wrong type
    c6 = Curve(x,x)
    with pytest.raises(ValueError):
        c6.apply_transformation(lambda x,y: (x[0:1],y)) # x of length 2 whereas y of length 3 -> not matching dimensions

def test_remove_data_points():
    x = np.linspace(1,10,10)
    c1 = Curve(x,x)
    c2 = c1.remove_data_points(lambda x,y: x>5)
    assert np.array_equal(c2.get_x(),np.array([1,2,3,4,5]))
    assert np.array_equal(c2.get_y(),np.array([1,2,3,4,5]))
    assert np.array_equal(c1.get_x(),np.array([1,2,3,4,5])) # old curve gets changed as well
    c3 = Curve(x,x)
    c4 = c3.remove_data_points(lambda x,y: y>5)
    assert np.array_equal(c4.get_y(),np.array([1,2,3,4,5]))
    c3 = Curve(x,x)
    c4 = c3.remove_data_points(lambda x,y: np.logical_and(y>5,x<=7))
    assert np.array_equal(c4.get_y(),np.array([1,2,3,4,5,8,9,10]))

def test_append():
    x1 = np.linspace(1,10,10)
    c1 = Curve(x1,x1)
    # appending itself
    c1.append(c1) 
    assert np.array_equal(c1._x, x1) # should not have any effect
    assert np.array_equal(c1._y, x1)
    # append with some overlap
    x2 = np.linspace(6,15,10)
    c2 = Curve(x2,x2)
    c3 = c1.append(c2)
    assert c3 == c1
    assert np.array_equal(c3._x, np.linspace(1,15,15))
    assert np.array_equal(c3._y, np.linspace(1,15,15))
    # append with no overlap
    x3 = np.array([1,2,3])
    c4 = Curve(x3,x3)
    x4 = np.array([6,7,8])
    c5 = Curve(x4,x4)
    c6 = c4.append(c5)
    assert np.array_equal(c6._x, np.array([1,2,3,6,7,8]))
    assert np.array_equal(c6._y, np.array([1,2,3,6,7,8]))
    # append with full overlap
    x5 = np.linspace(1,10,5)
    x6 = np.linspace(2,9,4)
    c7 = Curve(x5,x5)
    c8 = Curve(x6,x6)
    c9 = c7.append(c8)
    assert np.array_equal(c9._x,x5)
    assert np.array_equal(c9._y,x5)

# tests for smoothing missing

def test_derivative():
    x = np.linspace(1,10,10)
    c1 = Curve(x,x)
    d1 = c1.derivative()
    assert np.abs(d1.evaluate(4.3)-1) < 1e-5
    c2 = Curve(x,x**2)
    d2 = c2.derivative()
    x_eval = np.array([2,5,4.2])
    assert np.max(np.abs(d2.evaluate(x_eval)-2*x_eval)) < 1e-2

def test_antiderivative():
    x = np.linspace(0,10,11)
    c1 = Curve(x,x)
    ad1 = c1.antiderivative()
    assert np.array_equal(ad1.get_y(),0.5*x**2)
def test_integrate():
    x = np.linspace(0,10,11)
    c1 = Curve(x,x)
    int1 = c1.integrate(x_range=(3,5))
    assert int1 - 8 < 1e-5
    int2 = c1.integrate(x_range=(2.5,6.2))
    assert int2-22.345 < 1e-3
def test_convoluted_with():
    x = np.array([-2,-1,-0.99999,0,0.99999,1,2])
    y = np.array([ 0, 0, 1      ,1, 1,     0, 0])
    c1 = Curve(x,y)
    c2 = Curve(10*x,y)
    c3 = c2.convoluted_with(c1,0.1,10)
    assert (c3.evaluate(0)-2) < 2e-2
    assert (c3.evaluate(10)-1) < 2e-2
    assert (c3.evaluate(11)-0) < 2e-2

test_set_data()
test_sort_x()
test_get_x_range_x_y()
test_get_x()
test_get_y()
test_get_x_y()
test_get_xy()
test_default_csv_header_parser()
test_csv_io()
test_pickle_io()
test_crop()
test_cropped()
test_evaluate()
test_local_maxima()
test_local_minima()
test_envelope()
test_get_maximum()
test_get_minimum()
test_mean()
test_std()
test_fit()
test_FFT()
test_IFFT()
test_add()
test_sub()
test_mul()
test_div()
test_pow()
test_apply_transformation()
test_remove_data_points()
test_append()
test_derivative()
test_antiderivative()
test_integrate()
test_convoluted_with()
print("Tests successful")