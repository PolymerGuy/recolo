class  Fields(object):
    def __init__(self,deflection,press,slopes,curvatures):
        self.deflection = deflection
        self.press = press
        self.slope_x,self.slope_y = slopes
        self.curv_xx,self.curv_yy,self.curv_xy = curvatures