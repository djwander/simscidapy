"""
unit tests for Map.py
"""
import pytest
from Map import Map
from Curve import Curve
import numpy as np

def test_init():
    """ Test initialization """
    m0 = Map() # empty map
    
    ### different ways of defining the maps position and size
    dummy = np.zeros((10,10))
    dummy[0][0] = 1
    dummy[9][0] = 2

    # invalid inputs
    with pytest.raises(ValueError):
        m2 = Map(dummy,data_range=(("a","b"),(1,2))) # should raise ValueError
    with pytest.raises(ValueError):
        m3 = Map(dummy,data_range=(1,2,3,4)) # should raise ValueError

    # default
    m1 = Map(dummy) 
    assert np.array_equal(m1.center_position,np.array((0,0))) # center at 0,0
    assert np.array_equal(m1.size,np.array((10-1,10-1))) # default: spacing of 1 -> size = (number of points-1)*1 
    #print(f"m1\ncenter_position: {m1.center_position} \nsize: {m1.size}")

    # specifying range covered by the data
    m4 = Map(dummy,data_range=((0,5),(-10,10)))
    assert np.array_equal(m4.center_position,np.array((2.5,0))) # center of the range x=(0,5) y=(-10,10)
    assert np.array_equal(m4.size,np.array((5,20)))
    #print(f"m4\ncenter_position: {m4.center_position} \nsize: {m4.size}")

    # specifying center position only -> size from shape
    m5 = Map(dummy,center_position=(10,10)) 
    assert np.array_equal(m5.center_position,np.array((10,10)))
    assert np.array_equal(m5.size,np.array((9,9))) # no size given -> default point spacing of 1
    #print(f"m5\ncenter_position: {m5.center_position} \nsize: {m5.size}")

    # specifying center position and size
    m6 = Map(dummy,center_position=(10,10),size=(2.5,2.5)) 
    assert np.array_equal(m6.center_position,np.array((10,10)))
    assert np.array_equal(m6.size,np.array((2.5,2.5)))
    #print(f"m6\ncenter_position: {m6.center_position} \nsize: {m6.size}")

    # initialization of angle and colormap
    m7 = Map(angle=27.3,colormap='jet')
    assert m7.angle == 27.3
    assert m7.colormap == 'jet'

    ### initialization of labels and units
    m8 = Map(data_label="dummy data",data_unit="a.u.",x_label="x points",x_unit="a.u.2",y_label="y points",y_unit="a.u.3")
    assert m8.data_label == 'dummy data'
    assert m8.data_unit == 'a.u.'
    assert m8.x_label == 'x points'
    assert m8.x_unit == 'a.u.2'
    assert m8.y_label == 'y points'
    assert m8.y_unit == 'a.u.3'

def test_rot_mat():
    mat0 = Map.rot_mat(0)
    assert np.array_equal(mat0,np.array([[1,0],[0,1]]))
    mat45 = Map.rot_mat(45)
    assert np.array_equal(mat45,np.array([[np.sqrt(2)/2,-np.sqrt(2)/2],[np.sqrt(2)/2,np.sqrt(2)/2]]))
    v1 = np.array([1,0])
    v2 = mat45.dot(v1)
    assert np.array_equal(v2,np.array([np.sqrt(2)/2,np.sqrt(2)/2]))

def test_get_x1_spacing():
    data = np.zeros((2,2))
    m = Map(data=data,size=(2,2))
    assert m.get_x1_spacing() == 2
    m2 = Map(data=data,size=(2,2),angle=30)
    assert m2.get_x1_spacing() == 2
    data = np.zeros((4,3))
    m3 = Map(data=data,center_position=(0,0),size=(2,2),angle=30)
    assert m3.get_x1_spacing() == 2/(4-1)
    m4 = Map(data=data,angle=30)
    assert m4.get_x1_spacing() == 1  # default spacing: 1

def test_get_x2_spacing():
    data = np.zeros((2,2))
    m = Map(data=data,size=(2,2))
    assert m.get_x2_spacing() == 2
    m2 = Map(data=data,size=(2,2),angle=30)
    assert m2.get_x2_spacing() == 2
    data = np.zeros((4,3))
    m2 = Map(data=data,size=(2,2),center_position=(0,0),angle=30)
    assert m2.get_x2_spacing() == 2/(3-1)
    m4 = Map(data=data,angle=30)
    assert m4.get_x2_spacing() == 1  # default spacing: 1

def test_get_grid_vectors():
    data = np.zeros((2,2))
    m = Map(data=data,size=(2,2))
    v1,v2 = m.get_grid_vectors()
    assert np.array_equal(v1,np.array([2,0]))
    assert np.array_equal(v2,np.array([0,2]))
    m1 = Map(data=data,size=(1,1),center_position=(0,0),angle=45)
    v1,v2 = m1.get_grid_vectors()
    assert np.array_equal(v1,np.array([np.sqrt(2)/2,np.sqrt(2)/2]))
    assert np.array_equal(v2,np.array([-np.sqrt(2)/2,np.sqrt(2)/2]))

def test_get_pt_coordinates():
    # the lists are built linewise, starting from the lower left corner
    data = np.zeros((2,2))
    m = Map(data=data) # default point spacing: 1
    x_pts, y_pts = m.get_pt_coordinates()
    assert np.array_equal(x_pts,np.array([-0.5,0.5,-0.5,0.5]))
    assert np.array_equal(y_pts,np.array([-0.5,-0.5,0.5,0.5])) 

    m1 = Map(data=data,angle=45)
    d = np.sqrt(2)/2
    x_pts, y_pts = m1.get_pt_coordinates()
    assert np.array_equal(x_pts,np.array([0,d,-d,0]))
    assert np.array_equal(y_pts,np.array([-d,0,0,d])) 

    m2 = Map(data=data,angle=45,size=(2,2),center_position=(1,2))
    d = np.sqrt(2)
    x_pts, y_pts = m2.get_pt_coordinates()
    assert np.max(x_pts-np.array([1,1+d,1-d,1]))<1e-10
    assert np.max(y_pts-np.array([2-d,2,2,2+d]))<1e-10

def test_get_xyz():
    d = np.array([[1,2/np.pi],[3,4]])
    m = Map(data=d)
    x,y,z = m.get_xyz()
    assert np.array_equal(x,np.array([-0.5,0.5,-0.5,0.5]))
    assert np.array_equal(y,np.array([-0.5,-0.5,0.5,0.5]))
    assert np.array_equal(z,np.array([1,3,2/np.pi,4]))

def test_get_lines():
    x = np.linspace(1,10,10)
    list_of_curves = [Curve(x,x),Curve(x,x**2),Curve(x,x**3)]
    map = Map().from_curves(list_of_curves)

    with pytest.raises(ValueError):
        map.get_lines(['a'])
    with pytest.raises(ValueError):
        map.get_lines([1.4])
    with pytest.raises(ValueError):
        map.get_lines([3])
    with pytest.raises(ValueError):
        map.get_lines([-4])
    
    c1 = map.get_lines(indices=[0])[0]
    assert np.array_equal(c1.get_x(),x)
    assert np.array_equal(c1.get_y(),x)

    c2 = map.get_lines(indices=[-1])[0]
    assert np.array_equal(c2.get_x(),x)
    assert np.array_equal(c2.get_y(),x**3)

    c3 = map.get_lines(y_range=(-0.5,0.2))[0]
    assert np.array_equal(c3.get_x(),x)
    assert np.array_equal(c3.get_y(),x**2)

def test_evaluate():
    sx = 11
    sy = 21
    z = np.ones((sx,sy))
    # calculate data: z=x**2+y
    for x in range(sx):
        for y in range(sy):
            z[x][y] = (x+0.5-sx/2)**2+y+0.5-sy/2
    m = Map(data=z,data_range=((-5,5),(-10,10)))
    # directly on data points
    assert np.abs(m.evaluate(0,0)) < 1e-5
    assert np.abs(m.evaluate(0,8)-8) < 1e-5
    assert np.abs(m.evaluate(5,0)-25) < 1e-5
    assert np.abs(m.evaluate(5,5)-30) < 1e-5
    assert np.abs(m.evaluate(5,-5)-20) < 1e-5
    # linear interpolation
    assert np.abs(m.evaluate(0,1.5)-1.5) < 1e-5
    assert np.abs(m.evaluate(0.5,0)-0.5) < 1e-5
    assert np.abs(m.evaluate(1.5,0)-2.5) < 1e-5
    # cubic interpolation
    assert np.abs(m.evaluate(0,1.5,method='cubic')-1.5) < 1e-3
    assert np.abs(m.evaluate(0.5,0,method='cubic')-0.25) < 1e-3
    assert np.abs(m.evaluate(1.5,0,method='cubic')-2.25) < 1e-2
    # nearest interpolation
    assert np.abs(m.evaluate(0,1.3,method='nearest')-1) < 1e-5
    assert np.abs(m.evaluate(0.6,0,method='nearest')-1) < 1e-5
    assert np.abs(m.evaluate(0.5,0,method='nearest')-0) < 1e-5
    assert np.abs(m.evaluate(0,0.5,method='nearest')-1) < 1e-5
    assert np.abs(m.evaluate(1.5,0,method='nearest')-1) < 1e-5
    assert np.abs(m.evaluate(-1.5,0,method='nearest')-1) < 1e-5
    assert np.abs(m.evaluate(-2.5,0,method='nearest')-9) < 1e-5
    assert np.abs(m.evaluate(-3.5,0,method='nearest')-16) < 1e-5
    assert np.abs(m.evaluate(3.5,0,method='nearest')-16) < 1e-5

def test_apply_transformation():
    d = np.array([[1,3],[2,4]])
    m1 = Map(d)
    m1.apply_transformation(lambda x,y,z: z)
    assert np.array_equal(m1.data,d)
    m2 = Map(d)
    m2.apply_transformation(lambda x,y,z: x)
    assert np.array_equal(m2.data,np.array([[-0.5,-0.5],[0.5,0.5]]))

def test_cropped():
    # no interpolation note: the angle of the cropped map will always be the same as the angle of the old one
    d0 = np.zeros((5,3))
    for x in range(5):
        for y in range(3):
            d0[x][y] = int(f'{x}{y}')
     
    m0 = Map(d0, angle=30) 
    m0c = m0.cropped(size=(2.4,1.7))
    assert np.array_equal(m0c.data,np.array([[11],[21],[31]]))

    m1 = Map(d0,angle=30,center_position=(1,1))
    m1c = m1.cropped(size=(2.4,1.7),center_position=(1,1))
    assert np.array_equal(m1c.data,np.array([[11],[21],[31]]))

    m2 = Map(d0,angle=0,center_position=(1,1))
    m2c = m2.cropped(size=(2.4,1.7),center_position=(0,0))
    assert np.array_equal(m2c.data,np.array([[0],[10],[20]]))

    m3 = Map(d0,angle=45,center_position=(0,0))
    m3c = m3.cropped(size=(2.4,1.7),center_position=(0,-np.sqrt(2)))
    assert np.array_equal(m3c.data,np.array([[0],[10],[20]]))

    m4 = Map(d0,angle=45,center_position=(0,0))
    m4c = m3.cropped(size=(2.4,1.7),center_position=(0,-np.sqrt(2)),
                    angle=75,x_pts=100,y_pts=200,method='nearest',fill_value=5,rescale=True) # all these parameters should be ignored
    assert np.array_equal(m4c.data,np.array([[0],[10],[20]]))

    m5 = Map(d0,angle=0,center_position=(0,0))
    m5c = m4.cropped(size=(2.4,1.7),center_position=(10,0), # new map totally outside old one -> should give empty map
                    angle=75,x_pts=100,y_pts=200,method='nearest',fill_value=5,rescale=True) # all these parameters should be ignored
    assert np.array_equal(m5c.data,np.zeros((0,0)))
    
    # with interpolation
    m5 = Map([[1,1],[2,2]],size=(2,2))
    m5c = m5.cropped(size=(1,1),interpolate=True,x_pts=2,y_pts=2,fill_value=5) # crop without angle - using default linear interpolation
    assert np.array_equal(m5c.data,np.array([[1.25,1.25],[1.75,1.75]]))

    m6 = Map([[1,1],[2,2]],size=(2,2))
    m6c = m6.cropped(size=(6,6),interpolate=True) # not specifying number of points -> about the same density as before -> 4x4pts
    assert np.array_equal(m6c.data,np.array([[0,0,0,0],[0,1,1,0],[0,2,2,0],[0,0,0,0]])) # values outside old map are filled with default fill value (0)
def test_where():
    m = Map([[1,2,3],[4,5,6]])
    pts1 = m.where(lambda x,y,z: x>0)
    assert np.array_equal(pts1[0],[0.5,0.5,0.5])
    assert np.array_equal(pts1[2],[4,5,6])

    pts2 = m.where(lambda x,y,z: z%2==0)
    assert np.array_equal(pts2[2],[4,2,6])

    m2 = Map(np.zeros((5,5)))
    around = (0,0)
    r = 1.1
    pts4 = m2.where(lambda x,y,z: (x-around[0])**2+(y-around[1])**2 < r**2)
    assert np.array_equal(pts4[0],[0,-1,0,1,0])
    assert np.array_equal(pts4[1],[-1,0,0,0,1])

def test_profile():
    m = Map(np.zeros((11,11)))
    m.apply_transformation(lambda x,y,z: np.sqrt(x**2+y**2))
    p1 = m.profile([(-5,0),(0,0),(5,0)],
                interpolation_args={'method':'nearest'})
    assert np.array_equal(p1.get_x(),[0,5,10])
    assert np.array_equal(p1.get_y(),[5,0,5])


def test_linearly_spaced_points():
    pts1 = Map._linearly_spaced_points((0,0),(1,0),3)
    assert np.array_equal(pts1,np.array([[0,0],[0.5,0],[1,0]]))
    pts2 = Map._linearly_spaced_points((0,0),(0,1),3)
    assert np.array_equal(pts2,np.array([[0,0],[0,0.5],[0,1]]))
    pts3 = Map._linearly_spaced_points((0,0),(1,1),3)
    assert np.array_equal(pts3,np.array([[0,0],[0.5,0.5],[1,1]]))

def test_straight_profile():
    m = Map(np.zeros((11,11)))
    m.apply_transformation(lambda x,y,z: np.sqrt(x**2+y**2))
    p1 = m.straight_profile(pt_start=(-5,0),pt_stop=(5,0),num_points=11,
                interpolation_args={'method':'nearest'})
    assert np.array_equal(p1.get_x(),np.linspace(0,10,11))
    assert np.array_equal(p1.get_y(),np.abs(np.linspace(-5,5,11)))

def test_add():
    d = np.ones((2,2))
    m1 = Map(d)
    m2 = Map(d)
    m3 = m1+m2
    assert np.array_equal(m3.data,2*np.ones((2,2)))
    m4 = m1+3
    assert np.array_equal(m4.data,4*np.ones((2,2)))
    m5 = m1+1.5
    assert np.array_equal(m5.data,2.5*np.ones((2,2)))
    d2 = np.array([[1,2],[3,4]])
    m6 = Map(d2)
    m7 = m1+m6
    assert np.array_equal(m7.data,d2+1)
    d3 = np.array([[0,4],[0,4]])
    m8 = Map(d3,center_position=(0,0),size=(2,4)) 
    m9 = m1 + m8 # with linear interpolation, m8 is 1.5 and 2.5 at the data points of m1  (at y=-0.5 and +0.5)
    assert np.array_equal(m9.data,np.array([[2.5,3.5],[2.5,3.5]]))

    d4 = np.reshape(np.linspace(0,5,6),(2,3))
    d5 = np.reshape(np.linspace(0,3,4),(2,2))
    m10 = Map(d4)
    m11 = Map(d5)
    m12 = m11+m10
    assert np.array_equal(m12.data,np.array([[0.5,2.5],[5.5,7.5]]))

def test_sub():
    d = np.ones((2,2))
    m1 = Map(d)
    m2 = Map(d)
    m3 = m1-m2
    assert np.array_equal(m3.data,0*np.ones((2,2)))
    m4 = m1-3
    assert np.array_equal(m4.data,-2*np.ones((2,2)))
    m5 = m1-1.5
    assert np.array_equal(m5.data,-0.5*np.ones((2,2)))
    d2 = np.array([[1,2],[3,4]])
    m6 = Map(d2)
    m7 = m6-m1
    assert np.array_equal(m7.data,d2-1)
    d3 = np.array([[0,4],[0,4]])
    m8 = Map(d3,center_position=(0,0),size=(2,4)) 
    m9 = m1 - m8 # with linear interpolation, m8 is 1.5 and 2.5 at the data points of m1  (at y=-0.5 and +0.5)
    assert np.array_equal(m9.data,np.array([[-0.5,-1.5],[-0.5,-1.5]]))

    d4 = np.reshape(np.linspace(0,5,6),(2,3))
    d5 = np.reshape(np.linspace(0,3,4),(2,2))
    m10 = Map(d4)
    m11 = Map(d5)
    m12 = m11+m10
    assert np.array_equal(m12.data,np.array([[0.5,2.5],[5.5,7.5]]))

def test_mul():
    # multiply with integer
    d1 = np.array([[1,2],[3,4]])
    m1 = Map(d1)
    m2 = m1*2
    assert np.array_equal(m2.data,d1*2)
    # multiply with float
    m3 = m1*1.5
    assert np.array_equal(m3.data,d1*1.5)
    # multiply with other map of same size and position
    m4 = Map(2*np.ones((2,2)))
    m5 = m1*m4
    assert np.array_equal(m5.data,d1*2)
    # another example
    m6 = Map(d1)
    m7 = m1*m6
    assert np.array_equal(m7.data,d1**2)
    # test fill value
    m8 = Map(np.array([[1,2],[3,4],[5,6]]),center_position=(1.5,1))
    m9 = Map(2*np.ones((2,2)),center_position=(1,1)) 
    m10 = m8*m9 # default fill value: 1
    assert np.array_equal(m10.data,np.array([[2,4],[6,8],[5,6]]))
    # test linear interpolation
    m11 = Map(np.array([[1,2],[3,4],[5,6]]),center_position=(1.5,1))
    m12 = Map(np.array([[1,1],[3,3]]),data_range=((1,3),(-3,3)))  #map with z = x
    m13 = m11*m12 # default interpolation: linear
    # 1st column: x=0.5 which is outside [1,3] of the second map -> fill_value: 1
    # 2nd column x=1.5; 3rd column x=2.5
    assert np.array_equal(m13.data,np.array([[1*1,1*2],[1.5*3,1.5*4],[2.5*5,2.5*6]])) 
    # test nearest interpolation
    m14 = Map(np.array([[1,2],[3,4],[5,6]]),center_position=(1.5,1))
    m15 = Map(np.array([[1,1],[3,3]]),data_range=((0,4),(-3,3)))  #map with z = x
    m15.default_mul_interpolation_args = {'method':'nearest','fill_value':0,'rescale':False}
    m16 = m14*m15 
    # 1st column: x=0.5 which is outside [1,3] of the second map -> fill_value: 1
    # 2nd column x=1.5; 3rd column x=2.5
    assert np.array_equal(m16.data,np.array([[1*1,1*2],[1*3,1*4],[3*5,3*6]])) 

def test_truediv():
    # multiply with integer
    d1 = np.array([[1,2],[3,4]])
    m1 = Map(d1)
    m2 = m1/2
    assert np.array_equal(m2.data,d1/2)
    # multiply with float
    m3 = m1/1.5
    assert np.array_equal(m3.data,d1/1.5)
    # multiply with other map of same size and position
    m4 = Map(2*np.ones((2,2)))
    m5 = m1/m4
    assert np.array_equal(m5.data,d1/2)
    # another example
    m6 = Map(d1)
    m7 = m1/m6
    assert np.array_equal(m7.data,np.ones((2,2)))
    # test fill value
    m8 = Map(np.array([[1,2],[3,4],[5,6]]),center_position=(1.5,1))
    m9 = Map(2*np.ones((2,2)),center_position=(1,1)) 
    m10 = m8/m9 # default fill value: 1
    assert np.array_equal(m10.data,np.array([[1/2,2/2],[3/2,4/2],[5,6]]))
    # test linear interpolation
    m11 = Map(np.array([[1,2],[3,4],[5,6]]),center_position=(1.5,1))
    m12 = Map(np.array([[1,1],[3,3]]),data_range=((1,3),(-3,3)))  #map with z = x
    m13 = m11/m12 # default interpolation: linear
    # 1st column: x=0.5 which is outside [1,3] of the second map -> fill_value: 1
    # 2nd column x=1.5; 3rd column x=2.5
    assert np.array_equal(m13.data,np.array([[1/1,2/1],[3/1.5,4/1.5],[5/2.5,6/2.5]])) 
    # test nearest interpolation
    m14 = Map(np.array([[1,2],[3,4],[5,6]]),center_position=(1.5,1))
    m15 = Map(np.array([[1,1],[3,3]]),data_range=((0,4),(-3,3)))  #map with z = x
    m15.default_div_interpolation_args = {'method':'nearest','fill_value':0,'rescale':False}
    m16 = m14/m15 
    # 1st column: x=0.5 which is outside [1,3] of the second map -> fill_value: 1
    # 2nd column x=1.5; 3rd column x=2.5
    assert np.array_equal(m16.data,np.array([[1/1,2/1],[3/1,4/1],[5/3,6/3]])) 


def test_select_rect():
    raise NotImplementedError("continue here")
def test_mean():
    data = np.ones((2,2))
    m = Map(data = data)
    assert m.mean() == 1
    m2 = Map(data=np.array([[1,2],[3,4]]))
    assert m2.mean() == 2.5

def test_subtract_mean():
    data = np.ones((2,2))
    m = Map(data = data).subtract_mean()
    assert m.mean() == 0
    assert np.array_equal(m.data,np.array([[0,0],[0,0]]))
    m2 = Map(data=np.array([[1,2],[3,4]])).subtract_mean()
    assert m2.mean() == 0
    assert np.array_equal(m2.data,np.array([[-1.5,-0.5],[0.5,1.5]]))

def test_mean_around():
    m = Map([[1,2,3],[4,5,6],[7,8,9]])

    mean1 = m.mean_around((0,0.1))
    assert mean1 == 5

    mean2 = m.mean_around((0,0),radius=1)
    assert mean2 == np.mean([2,4,5,6,8])

    mean3 = m.mean_around((1,1),radius=1)
    assert mean3 == np.mean([9,8,6])

def test_three_point_level():
    m = Map(np.ones((11,11)))
    m.apply_transformation(lambda x,y,z: 2.5*x+1.75*y)

    m.three_point_level([(0,0),(3,0),(0,3)])
    _,_,z = m.get_xyz()
    assert np.max(np.abs(z))==0

def test_FFT():
    m = Map(np.ones((100,100)),size=(5,5))
    m.apply_transformation(lambda x,y,z: np.sin(x*2*np.pi)+2.3*np.sin(1.6*y*2*np.pi))
    fft = m.FFT()
    assert np.abs(fft.evaluate(1,0)) > 10*m.mean()
    assert np.abs(fft.evaluate(-1,0)) > 10*m.mean()
    assert np.abs(fft.evaluate(0,1.6)) > 10*m.mean()
    assert np.abs(fft.evaluate(0,-1.6)) > 10*m.mean()

def test_IFFT():
    m = Map(np.ones((100,100)),size=(5,5))
    m.apply_transformation(lambda x,y,z: np.sin(x*2*np.pi)+2.3*np.sin(1.6*y*2*np.pi))
    fft = m.FFT()
    ifft = fft.IFFT()
    diff = m-ifft
    assert np.max(np.abs(diff.get_xyz()[2])) < 1e-3 # IFFT(FFT) = 1

def test_apply_transformation_linewise():
    m = Map(np.zeros((5,4)),center_position=(2,3),size=(10,3),angle=30)
    slope = Curve([0,10],[0,10])
    def random_line_offset(c,y,y_index):
        return c+slope+y_index
    m.apply_transformation_linewise(random_line_offset,direction='v')
    x,y,z = m.get_xyz()
    z = z.reshape((4,5)).T
    assert np.array_equal(m.center_position,np.array([2,3]))
    assert np.array_equal(m.size,np.array([10,3]))
    assert z[0][0] == 0.5
    assert z[1][0] == 1.5
    assert z[1][1] == 2.5

def test_subtract_polynomial_linewise():
    x = np.linspace(0,1,256)
    l1 = Curve(x,np.sin(2*np.pi*x*30) + 10*(x-0.3)**2 + 5*x -2)
    l2 = Curve(x,np.sin(2*np.pi*x*30) + 10*(x-0.25)**3 + 5*x**2 -10)
    m = Map().from_curves([l1,l2])

    m.subtract_polynomial_linewise(3)
    assert np.max(np.abs(m.get_lines(indices=[0])[0].get_y()-np.sin(2*np.pi*x*30))) < 0.15
    assert np.max(np.abs(m.get_lines(indices=[1])[0].get_y()-np.sin(2*np.pi*x*30))) < 0.15

def test_fit():
    m = Map(np.zeros((100,100)),size=(10,10))
    m.apply_transformation(lambda x,y,z: (x-2.4)**2+(y+1)**2-3)

    def func(x,y,x0,y0,c):
        return (x-x0)**2+(y-y0)**2+c

    popt,_ = m.fit(func,(0,0,0))
    assert np.max(np.abs(popt-(2.4,-1,-3)))<1e-3

def test_fit_1D():
    m = Map(np.zeros((100,100)),size=(10,10))
    m.apply_transformation(lambda x,y,z: 1/((y-x**2)**2+0.1))

    def square(x,a,b):
        return a*x**2+b

    from Map import lower_cutoff_weight
    popt4,pcov4 = m.fit_1D(square,(1,0),weight=lambda x,y,z:lower_cutoff_weight(z,1))
    assert np.max(np.abs(popt4-(1,0))) < 3e-2

def test_save_csv():
    d = np.zeros((3,3))
    d[1][1] = 1.2345e-7
    d[0][0] = 3
    m = Map(d,
        data_label='test data',data_unit='TU',
        x_label='xL',x_unit='xU',
        y_label='yL',y_unit='yU',
        center_position=(1,2),size=(3,4),angle=5)
    m.save_csv("test_save.csv")

def test_load_csv():
    m2 = Map().load_csv('test_save.csv')
    assert m2.data_label == 'test data'
    assert m2.data_unit == 'TU'
    assert m2.x_label == 'xL'
    assert m2.x_unit == 'xU'
    assert m2.y_label == 'yL'
    assert m2.y_unit == 'yU'
    assert np.array_equal(np.array(m2.center_position),np.array((1,2)))
    assert np.array_equal(np.array(m2.size),np.array((3,4)))
    assert m2.angle == 5

    d = np.zeros((3,3))
    d[1][1] = 1.23e-7
    d[0][0] = 3
    assert np.array_equal(m2.data,d)

test_init()
test_rot_mat()
test_get_x1_spacing()
test_get_x2_spacing()
test_get_grid_vectors()
test_get_pt_coordinates()
test_get_xyz()
test_get_lines()
test_evaluate()
test_apply_transformation()
#test_cropped()
test_where()
test_mean()
test_linearly_spaced_points()
test_profile()
test_straight_profile()
test_add()
test_sub()
test_mul()
test_truediv()
test_subtract_mean()
test_mean_around()
test_three_point_level()
test_FFT()
test_IFFT()
test_apply_transformation_linewise()
test_subtract_polynomial_linewise()
test_fit()
test_fit_1D()
test_save_csv()
test_load_csv()
print('Test successfully finished')


## Plotting


# fig, ax = plt.subplots()
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# m6.setup_plot(ax)
# m6.plot(ax,ax_colorbar=cbar_ax,imshow_args={'vmin':-1,'vmax':3})
# fig.show()
# plt.show()

#m6.plot_standalone(with_colorbar=True,colorbar_range=(-1,3))

# # using user defined colormap
# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# viridis = cm.get_cmap('viridis', 256)   # start from viridis
# newcolors = viridis(np.linspace(0, 1, 256))
# pink = np.array([248/256, 24/256, 148/256, 1])
# newcolors[:25, :] = pink    # replace lowest part by pink color
# newcmp = ListedColormap(newcolors)
# m7 = deepcopy(m6)
# m7.colormap = newcmp
# m7.plot_standalone()

# plt.show()

# ## Saving and Loading
# m7.save("Test.map")
# del m7
# m8 = Map().load("Test.map")
# m8.plot_standalone()
# plt.show()

### Generating from Curves
# from Curve import Curve
# x = np.linspace(0,10,10)
# # equally spaced curves
# c1 = Curve(x=x,y=x)
# c2 = Curve(x=x,y=-1*x)
# m9 = Map(angle=30).from_curves([c1,c2],1).plot_standalone()
# plt.show()
# print("Done")

###############################################################################
#### Tests involving plots (should be checked by user)                   ######
###############################################################################
blk = False # block the execution of the program when showing a plot. 
            #       if False: just check if program produces errors; 
            #       if True: shows the plots -> user can check if the plots look as they should