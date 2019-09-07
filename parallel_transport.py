import numpy
import sympy
from sympy.vector import CoordSys3D

def mtx_rotate_by_vector(b, theta):
    """Returns 3x3 matrix for rotating around some 3D vector."""
    # Source:
    # Appendix A of "Parallel Transport to Curve Framing"
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    rot = numpy.array([
        [c+(b[0]**2)*(1-c), b[0]*b[1]*(1-c)-s*b[2], b[2]*b[0]*(1-c)+s*b[1]],
        [b[0]*b[1]*(1-c)+s*b[2], c+(b[1]**2)*(1-c), b[2]*b[1]*(1-c)-s*b[0]],
        [b[0]*b[2]*(1-c)-s*b[1], b[1]*b[2]*(1-c)+s*b[0], c+(b[2]**2)*(1-c)],
    ])
    return rot

def gen_faces(v1a, v1b, v2a, v2b):
    """Returns faces (as arrays of vertices) connecting two pairs of
    vertices."""
    # Keep winding order consistent!
    f1 = numpy.array([v1b, v1a, v2a])
    f2 = numpy.array([v2b, v1b, v2a])
    return f1, f2

def approx_tangents(points):
    """Returns an array of approximate tangent vectors.  Assumes a
    closed path, and approximates point I using neighbors I-1 and I+1 -
    that is, treating the three points as a circle.
    
    Input:
    points -- Array of shape (N,3). points[0,:] is assumed to wrap around
              to points[-1,:].
    
    Output:
    tangents -- Array of same shape as 'points'. Each row is normalized.
    """
    d = numpy.roll(points, -1, axis=0) - numpy.roll(points, +1, axis=0)
    d = d/numpy.linalg.norm(d, axis=1)[:,numpy.newaxis]
    return d

def approx_arc_length(points):
    p2 = numpy.roll(points, -1, axis=0)
    return numpy.sum(numpy.linalg.norm(points - p2, axis=1))

def torsion(v, arg):
    """Returns an analytical SymPy expression for torsion of a 3D curve.
    
    Inputs:
    v -- SymPy expression returning a 3D vector
    arg -- SymPy symbol for v's variable
    """
    # https://en.wikipedia.org/wiki/Torsion_of_a_curve#Alternative_description
    dv1 = v.diff(arg)
    dv2 = dv1.diff(arg)
    dv3 = dv2.diff(arg)
    v1_x_v2 = dv1.cross(dv2)
    # This calls for the square of the norm in denominator - but that
    # is just dot product with itself:
    return v1_x_v2.dot(dv3) / (v1_x_v2.dot(v1_x_v2))

def torsion_integral(curve_expr, var, a, b):
    # The line integral from section 3.1 of "Parallel Transport to Curve
    # Framing".  This should work in theory, but with the functions I've
    # actually tried, evalf() is ridiculously slow.
    c = torsion(curve_expr, var)
    return sympy.Integral(c * (sympy.diff(curve_expr, var).magnitude()), (var, a, b))

def torsion_integral_approx(curve_expr, var, a, b, step):
    # A numerical approximation of the line integral from section 3.1 of
    # "Parallel Transport to Curve Framing"
    N = CoordSys3D('N')
    # Get a (callable) derivative function of the curve:
    curve_diff = curve_expr.diff(var)
    diff_fn = sympy.lambdify([var], N.origin.locate_new('na', curve_diff).express_coordinates(N))
    # And a torsion function:
    torsion_fn = sympy.lambdify([var], torsion(curve_expr, var))
    # Generate values of 'var' to use:
    vs = numpy.arange(a, b, step)
    # Evaluate derivative function & torsion function over these:
    d = numpy.linalg.norm(numpy.array(diff_fn(vs)).T, axis=1)
    torsions = torsion_fn(vs)
    # Turn this into basically a left Riemann sum (I'm lazy):
    return -(d * torsions * step).sum()
