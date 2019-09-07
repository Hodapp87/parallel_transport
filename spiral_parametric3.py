#!/usr/bin/env python3

import sys
import numpy
import stl.mesh

# TODO:
# - This still has some strange errors around high curvature.
# They are plainly visible in the cross-section.
# - Check rotation direction
# - Fix phase, which only works if 0 (due to how I work with y)
# Things don't seem to line up right.
# - Why is there still a gap when using Array modifier?
# Check beginning and ending vertices maybe
# - Organize this so that it generates both meshes when run

# This is all rather tightly-coupled.  Almost everything is specific
# to the isosurface I was trying to generate. walk_curve may be able
# to generalize to some other shapes.
class ExplicitSurfaceThing(object):
    def __init__(self, freq, phase, scale, inner, outer, rad, ext_phase):
        self.freq = freq
        self.phase = phase
        self.scale = scale
        self.inner = inner
        self.outer = outer
        self.rad = rad
        self.ext_phase = ext_phase
        
    def angle(self, z):
        return self.freq*z + self.phase

    def max_z(self):
        # This value is the largest |z| for which 'radical' >= 0
        # (thus, for x_cross to have a valid solution)
        return (numpy.arcsin(self.rad / self.inner) - self.phase) / self.freq

    def radical(self, z):
        return self.rad*self.rad - self.inner*self.inner * (numpy.sin(self.angle(z)))**2

    # Implicit curve function
    def F(self, x, z):
        return (self.outer*x - self.inner*numpy.cos(self.angle(z)))**2 + (self.inner*numpy.sin(self.angle(z)))**2 - self.rad**2

    # Partial 1st derivatives of F:
    def F_x(self, x, z):
        return 2 * self.outer * self.outer * x - 2 * self.inner * self.outer * numpy.cos(self.angle(z))

    def F_z(self, x, z):
        return 2 * self.freq * self.inner * self.outer * numpy.sin(self.angle(z))

    # Curvature:
    def K(self, x, z):
        a1 = self.outer**2
        a2 = x**2
        a3 = self.freq*z + self.phase
        a4 = numpy.cos(a3)
        a5 = self.inner**2
        a6 = a4**2
        a7 = self.freq**2
        a8 = numpy.sin(a3)**2
        a9 = self.outer**3
        a10 = self.inner**3
        return -((2*a7*a10*self.outer*x*a4 + 2*a7*a5*a1*a2)*a8 + (2*a7*self.inner*a9*x**3 + 2*a7*a10*self.outer*x)*a4 - 4*a7*a5*a1*a2) / ((a7*a5*a2*a8 + a5*a6 - 2*self.inner*self.outer*x*a4 + a1*a2) * numpy.sqrt(4*a7*a5*a1*a2*a8 + 4*a5*a1*a6 - 8*self.inner*a9*x*a4 + 4*a2*self.outer**4))

    def walk_curve(self, x0, z0, eps, thresh = 1e-3, gd_thresh = 1e-7):
        x, z = x0, z0
        eps2 = eps*eps
        verts = []
        iters = 0
        # Until we return to the same point at which we started...
        while True:
            iters += 1
            verts.append([x, 0, z])
            # ...walk around the curve by stepping perpendicular to the
            # gradient by 'eps'.  So, first find the gradient:
            dx = self.F_x(x, z)
            dz = self.F_z(x, z)
            # Normalize it:
            f = 1/numpy.sqrt(dx*dx + dz*dz)
            nx, nz = dx*f, dz*f
            # Find curvature at this point because it tells us a little
            # about how far we can safely move:
            K_val = abs(self.K(x, z))
            eps_corr = 2 * numpy.sqrt(2*eps/K_val - eps*eps)
            # Scale by 'eps' and use (-dz, dx) as perpendicular:
            px, pz = -nz*eps_corr, nx*eps_corr
            # Walk in that direction:
            x += px
            z += pz
            # Moving in that direction is only good locally, and we may
            # have deviated off the curve slightly.  The implicit function
            # tells us (sort of) how far away we are, and the gradient
            # tells us how to minimize that:
            #print("W: x={} z={} dx={} dz={} px={} pz={} K={} eps_corr={}".format(
            #    x, z, dx, dz, px, pz, K_val, eps_corr))
            F_val = self.F(x, z)
            count = 0
            while abs(F_val) > gd_thresh:
                count += 1
                dx = self.F_x(x, z)
                dz = self.F_z(x, z)
                f = 1/numpy.sqrt(dx*dx + dz*dz)
                nx, nz = dx*f, dz*f
                # If F is negative, we want to increase it (thus, follow
                # gradient).  If F is positive, we want to decrease it
                # (thus, opposite of gradient).
                F_val = self.F(x, z)
                x += -F_val*nx
                z += -F_val*nz
                # Yes, this is inefficient gradient-descent...
            diff = numpy.sqrt((x-x0)**2 + (z-z0)**2)
            #print("{} gradient-descent iters. diff = {}".format(count, diff))
            if iters > 100 and diff < thresh:
                #print("diff < eps, quitting")
                #verts.append([x, 0, z])
                break
        data = numpy.array(verts)
        return data

    def x_cross(self, z, sign):
        # Single cross-section point in XZ for y=0.  Set sign for positive
        # or negative solution.
        n1 = numpy.sqrt(self.radical(z))
        n2 = self.inner * numpy.cos(self.angle(z))
        if sign > 0:
            return (n2-n1) / self.outer
        else:
            return (n2+n1) / self.outer

    def turn(self, points, dz):
        # Note one full revolution is dz = 2*pi/freq
        # How far to turn in radians (determined by dz):
        rad = self.angle(dz)
        c, s = numpy.cos(rad), numpy.sin(rad)
        mtx = numpy.array([
            [ c,  s,  0],
            [-s,  c,  0],
            [ 0,  0,  1],
        ])
        return points.dot(mtx) + [0, 0, dz]

    def screw_360(self, z0_period_start, x_init, z_init, eps, dz, thresh, endcaps=False):
        #z0 = -10 * 2*numpy.pi/freq / 2
        z0 = z0_period_start * 2*numpy.pi/self.freq / 2
        z1 = z0 + 2*numpy.pi/self.freq
        #z1 = 5 * 2*numpy.pi/freq / 2
        #z0 = 0
        #z1 = 2*numpy.pi/freq
        init_xsec = self.walk_curve(x_init, z_init, eps, thresh)
        num_xsec_steps = init_xsec.shape[0]
        zs = numpy.append(numpy.arange(z0, z1, dz), z1)
        num_screw_steps = len(zs)
        vecs = num_xsec_steps * num_screw_steps * 2
        offset = 0
        if endcaps:
            offset = num_xsec_steps
            vecs += 2*num_xsec_steps
        print("Generating {} vertices...".format(vecs))
        data = numpy.zeros(vecs, dtype=stl.mesh.Mesh.dtype)
        v = data["vectors"]
        # First endcap:
        if endcaps:
            center = init_xsec.mean(0)
            for i in range(num_xsec_steps):
                v[i][0,:] = init_xsec[(i + 1) % num_xsec_steps,:]
                v[i][1,:] = init_xsec[i,:]
                v[i][2,:] = center
        # Body:
        verts = init_xsec
        for i,z in enumerate(zs):
            verts_last = verts
            verts = self.turn(init_xsec, z-z0)
            if i > 0:
                for j in range(num_xsec_steps):
                    # Vertex index:
                    vi = offset + (i-1)*num_xsec_steps*2 + j*2
                    v[vi][0,:] = verts[(j + 1) % num_xsec_steps,:]
                    v[vi][1,:] = verts[j,:]
                    v[vi][2,:] = verts_last[j,:]
                    #print("Write vertex {}".format(vi))
                    v[vi+1][0,:] = verts_last[(j + 1) % num_xsec_steps,:]
                    v[vi+1][1,:] = verts[(j + 1) % num_xsec_steps,:]
                    v[vi+1][2,:] = verts_last[j,:]
                    #print("Write vertex {} (2nd half)".format(vi+1))
        # Second endcap:
        if endcaps:
            center = verts.mean(0)
            for i in range(num_xsec_steps):
                vi = num_xsec_steps * num_screw_steps * 2 + num_xsec_steps + i
                v[vi][0,:] = center
                v[vi][1,:] = verts[i,:]
                v[vi][2,:] = verts[(i + 1) % num_xsec_steps,:]
        v[:, :, 2] += z0 + self.ext_phase / self.freq
        v[:, :, :] /= self.scale
        mesh = stl.mesh.Mesh(data, remove_empty_areas=False)
        print("Beginning z: {}".format(z0/self.scale))
        print("Ending z: {}".format(z1/self.scale))
        print("Period: {}".format((z1-z0)/self.scale))
        return mesh

surf1 = ExplicitSurfaceThing(
    freq = 20,
    phase = 0,
    scale = 1/16, # from libfive
    inner = 0.4 * 1/16,
    outer = 2.0 * 1/16,
    rad = 0.3 * 1/16,
    ext_phase = 0)

z_init = 0
x_init = surf1.x_cross(z_init, 1)
mesh1 = surf1.screw_360(-10, x_init, z_init, 0.000002, 0.001, 5e-4)
fname = "spiral_inner0_one_period.stl"
print("Writing {}...".format(fname))
mesh1.save(fname)

surf2 = ExplicitSurfaceThing(
    freq = 10,
    phase = 0,
    scale = 1/16, # from libfive
    inner = 0.9 * 1/16,
    outer = 2.0 * 1/16,
    rad = 0.3 * 1/16,
    ext_phase = numpy.pi/2)

z_init = 0
x_init = surf2.x_cross(z_init, 1)
mesh2 = surf2.screw_360(-5, x_init, z_init, 0.000002, 0.001, 5e-4)
fname = "spiral_outer90_one_period.stl"
print("Writing {}...".format(fname))
mesh2.save(fname)
