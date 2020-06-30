"""
Note that this constrains the dependent variable from going *any further* past the constraints.
The ODE will still treat it as if it were at the value of the constraint, 
and with a small step size any problems should be minimal,
but you may still have slightly out-of-range numbers in your solution.
"""

import numpy as np
from functools import wraps

def constrain(constraints):
    """
    Decorator which wraps a function to be passed to an ODE solver which constrains the solution space.
    
    Example:
    
    @constrain([0, 1])
    def f(t, y)
        dy_dt = # your ODE
        return dy/dt
        
    solver = scipy.integrate.odeint(f, y0)  # use any solver you like!
    solution = solver.solve()
        
    If solution goes below 0 or above 1, the function f will ignore values of dy_dt which would make it more extreme,
    and treat the previous solution as if it were at 0 or 1.
    
    :params constraints: Sequence of (low, high) constraints - use None for unconstrained.
    """
    if all(constraint is not None for constraint in constraints):
        assert constraints[0] < constraints[1]

    def wrap(f):

        @wraps(f)
        def wrapper(obj, t, y, param, *args, **kwargs):
            lower, upper = constraints
            if lower is None:
                lower = -np.inf
            if upper is None:
                upper = np.inf

            
            too_low = y <= lower
            too_high = y >= upper
            
            y = np.maximum(y, np.ones(np.shape(y))*lower)
            y = np.minimum(y, np.ones(np.shape(y))*upper)

            result = f(obj, t, y, param, *args, **kwargs)

            result[too_low] = np.maximum(result[too_low], np.ones(too_low.sum())*lower)
            result[too_high] = np.minimum(result[too_high], np.ones(too_high.sum())*upper)

            return result

        return wrapper

    return wrap