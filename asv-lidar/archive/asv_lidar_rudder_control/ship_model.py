import numpy as np

MASS = 500
THRUST_COEF = 0.04 #0.032 #(320N @ 100 RPM, based on 2xT500 thrusters)
DRAG_COEF = 10 #12.8    #(320N of drag at 5m/s)
TURN_COEF = 2800      #480Nm @ 10deg/s
MAX_RUD_ANGLE = 45
RUDDEROFFSET = 3
MOMINERTIA = 0.5*MASS*RUDDEROFFSET**2

class ShipModel:
    def __init__(self):
        self._v = 0.
        self._a = 0.
        self._h = 0.
        self._w = 0.
        self._dw = 0.
        self._last_t = None
        

    def _calc_forces(self, rpm,rud):
        thrust =    THRUST_COEF * rpm**2
        rud_angle = np.radians(MAX_RUD_ANGLE * rud/100)
        fwd_thrust = thrust * np.cos(rud_angle) - DRAG_COEF   * self._v**2
        
        rud_moment = thrust * np.sin(rud_angle) * RUDDEROFFSET
        moment = rud_moment - (TURN_COEF * self._w)
        
        return fwd_thrust, moment
    

    def update(self, rpm,rud, dt):
        
        # Verlet Integration
        d = self._v*dt + self._a*dt*dt*0.5
        self._h = self._h + self._w*dt + self._dw*dt*dt*0.5

        dx = d * np.sin(self._h)
        dy = d * np.cos(self._h)

        thrust, moment = self._calc_forces(rpm,rud)
        a = thrust / MASS
        dw = moment / MOMINERTIA

        self._v = self._v + (self._a + a)*dt*0.5
        self._w = self._w + (self._dw + dw)*dt*0.5

        self._a = a
        self._dw = dw

        return dx,dy,np.degrees(self._h), np.degrees(self._w)
    
if __name__ == '__main__':
    model = ShipModel()

    for t in range(20):
        dx,dy,h,w = model.update(100,0,t)
        print(f"{t}\t{model._v:.1f}\t{w:.1f}")
    
    for t in range(t+1,t+20):
        dx,dy,h,w = model.update(100,100,t)
        print(f"{t}\t{model._v:.1f}\t{w:.1f}")