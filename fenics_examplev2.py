#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:23:56 2024

@author: dimitris
"""

import matplotlib.pyplot as plt

import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from dolfinx.fem.petsc import LinearProblem

from scipy.signal import fftconvolve
from scipy.interpolate import griddata

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 200
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 25, 25
a_D,b_D=-4,4
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([a_D, a_D]), np.array([b_D, b_D])],
                               [nx, ny], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))


# Create initial condition
def initial_condition(xx, a=5):
    # x=xx[0]
    # y=xx[1]
    # return np.exp(-a * (xx[0]**2 + xx[1]**2))+1
    # Returns 1/4 if (x, y) is within the square [-3, 3] x [-3, 3], otherwise 0
    return (1/4) * ((xx[0] >= -3) & (xx[0] <= 3) & (xx[1] >= -3) & (xx[1] <= 3))

def initial_condition2(xx, a=5):
    # x=xx[0]
    # y=xx[1]
    # return 0.1*xx[0]*(1-xx[0])*xx[1]*(1-xx[1])
    # return np.exp(-a * (xx[0]**2 + xx[1]**2))+1
    # Returns 1/4 if (x, y) is within the square [-3, 3] x [-3, 3], otherwise 0
    return (1/4) * ((xx[0] >= -3) & (xx[0] <= 3) & (xx[1] >= -3) & (xx[1] <= 3))

def initial_condition3(xx, a=5):
    
    return (1/4) * ((xx[0] >= -3) & (xx[0] <= 3) & (xx[1] >= -3) & (xx[1] <= 3))

    # x=xx[0]
    # y=xx[1]
    # return xx[0]*(1-xx[0])*xx[1]*(1-xx[1])
    # return np.exp(-a * (xx[0]**2 + xx[1]**2))+1
    # return 0.1*x[0]*(1-x[0])*x[1]*(1-x[1])

def exact_solution(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))+1


    # return 10*x[0]*(1-x[0])*x[1]*(1-x[1])
def W__(xx):
    x=xx[0]
    y=xx[1]
    return np.exp(-(x**2+y**2))/np.pi
    
def F_(x):
    return x**2



def compute_convo(u_h):

    aa = np.linspace(a_D, b_D, nx)
    bb = np.linspace(a_D, b_D, ny)
    A, B = np.meshgrid(aa, bb)
    
    Wconv=W__([A,B])*0
    points = np.zeros((3, bb.size))
    points[0] = aa
    points[1] = bb
    u_values = []
    p_values = []
    
    from dolfinx import geometry
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    
    colliding_cells2 = {}
    for i, point1 in enumerate(points[0]):
        for j, point2 in enumerate(points[1]):
            cell_candidates = geometry.compute_collisions_points(bb_tree, np.array([point1, point2, 0]))
            # Choose one of the cells that contains the point
            colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, np.array([point1, point2, 0]))
            colliding_cells2[i, j] = colliding_cells
            
    # for i, point1 in enumerate(points[0]):
    #     for j, point2 in enumerate(points[1]):
    #         Wconv[i,j]=W__([point1,point2])
    
    cells={}
    points_on_proc={}
    for i, point1 in enumerate(points[0]):
        for j, point2 in enumerate(points[1]):
            if len(colliding_cells2[i, j].links(0)) > 0:
                points_on_proc[i,j]=[point1,point2,0]
                cells[i,j]=colliding_cells2[i, j].links(0)[0]
                
    # points_on_proc = np.array(points_on_proc, dtype=np.float64)
    U=Wconv
    for i, point1 in enumerate(points[0]):
        for j, point2 in enumerate(points[1]):
            if i>0 and j>0 and i<(len(points[0])-1) and  j<(len(points[0])-1):
                U[i,j]=u_n.eval(points_on_proc[i,j], cells[i,j])
            
    Wconv=W__([A,B])
    return U,Wconv



def source_term(xx):
    x=xx[0]
    y=xx[1]
    lap_expr = (-200*x**2 - 200*y**2 + 20)*np.exp(-10*x**2 - 10*y**2)
    lab_expr = -(200*x**2 + 200*y**2 - 20)*np.exp(-10*x**2 - 10*y**2)
    lab_expr = (-20.0*x**2*(np.exp(5*x**2 + 5*y**2) + 1)**2 - 40.0*x**2*(np.exp(5*x**2 + 5*y**2) + 1) - 20.0*y**2*(np.exp(5*x**2 + 5*y**2) + 1)**2 - 40.0*y**2*(np.exp(5*x**2 + 5*y**2) + 1) + 4.0*(np.exp(5*x**2 + 5*y**2) + 1)**2)*np.exp(-15*x**2 - 15*y**2)
    return lap_expr


def interpolate_conv(xxx,fft_result):
    aa = np.linspace(a_D, b_D, nx)
    bb = np.linspace(a_D, b_D, ny)
    da = aa[1] - aa[0]
    db = bb[1] - bb[0]
    # Set up the coordinate grid for the convolution result
    xx, yy = np.indices(fft_result.shape)
    xx = xx * da-a_D  # Assume `da` is your grid spacing in the x-direction
    yy = yy * db-a_D  # Assume `db` is your grid spacing in the y-direction
    
    # Flatten the arrays for use with griddata
    point_ = np.vstack((xx.ravel(), yy.ravel())).T
    values = fft_result.ravel()
    
    
    interpolated_value_nearest = griddata(point_, values, xxx[0:2,:].T, method='nearest')

    return interpolated_value_nearest*da*db

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)




xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)

x = V.tabulate_dof_coordinates()
x_order = np.argsort(x[:,0])

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition2)

u_nl = fem.Function(V)
u_nl.name = "u_nl"
u_nl.interpolate(initial_condition)


u_nm = fem.Function(V)
u_nm.name = "u_nm"
u_nm.interpolate(initial_condition3)


f = fem.Function(V)
f.name = "source"
f.interpolate(source_term)
L2_diff=1

W_ = fem.functionspace(domain, ("Lagrange", 1 + 4))
W_ = fem.Function(W_)
W_.interpolate(lambda x: W__(x))

W_c = fem.functionspace(domain, ("Lagrange", 1 + 4))
W_c.name = "convo"
W_c = fem.Function(W_c)



VV_x = fem.functionspace(domain, ("Lagrange", 1 + 4))
VV_x.name = "VV_x"
VV_x = fem.Function(VV_x)
VV_x.interpolate(lambda x: x[0]**2+x[1]**2)
# W_c.interpolate(lambda x: interpolate_conv(x,fft_result))
# Function space for exact solution - need it to be higher than deg
V_exact = fem.functionspace(domain, ("Lagrange", 1 + 4))
u_exact = fem.Function(V_exact)
u_exact.interpolate(lambda x: exact_solution(x))

# import convolution_estimate
# xdmf.write_function(u_n, t)


for i in range(num_steps):
    L2_diff=1
    iteration=0
    u_n.x.array[u_n.x.array<0.0000]=0

    while np.sqrt(L2_diff)>1e-15 and iteration<300:
        iteration+=1

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        
        # a,b=-5,5
        # U,Wconv=compute_convo(u_n)
        # fft_result = fftconvolve(Wconv, U, mode='same')
        # W_c.fft_result=fftconvolve(Wconv, U, mode='same')
        # W_c.interpolate(lambda x: interpolate_conv(x,fft_result))

        #non smoothing
    #    a = (1/(nx*ny))*ufl.inner(ufl.grad(u),ufl.grad(v))* ufl.dx  + ufl.inner(u,v)* ufl.dx +dt*ufl.dot(u_n*ufl.grad((0.2*u_n)*u), ufl.grad(v)) * ufl.dx  + dt*ufl.dot(u*ufl.grad(0.1*u_n**2+VV_x), ufl.grad(v)) * ufl.dx
    #    L = -((u_n-u_nm)*v* ufl.dx +dt*ufl.inner( u_n* ufl.grad(0.1*u_n**2+VV_x) ,ufl.grad(v))* ufl.dx+(1/(nx*ny))*ufl.inner(ufl.grad(u_n),ufl.grad(v))* ufl.dx) #- dt*ufl.inner(f, v) * ufl.dx)
     
        #smoothing
        a = (1/(nx*ny))*ufl.inner(ufl.grad(u),ufl.grad(v))* ufl.dx  + ufl.inner(u,v)* ufl.dx +dt*ufl.dot(u_n*ufl.grad((0.2*u_n)*u), ufl.grad(v)) * ufl.dx  + dt*ufl.dot(u*ufl.grad(0.1*u_n**2+VV_x), ufl.grad(v)) * ufl.dx
        L = -((u_n-u_nm)*v* ufl.dx +dt*ufl.inner( u_n* ufl.grad(0.1*u_n**2+VV_x) ,ufl.grad(v))* ufl.dx+(1/(nx*ny))*ufl.inner(ufl.grad(u_n),ufl.grad(v))* ufl.dx) #- dt*ufl.inner(f, v) * ufl.dx)
     
        

    
        
        problem = LinearProblem(a, L,bcs=[bc], u=uh, petsc_options={"ksp_type": "gmres", "pc_type": "lu"})
        problem.solve()
        
        
        uh.x.scatter_forward()
        # uh.x.array[uh.x.array<0.0000]=0
        u_nl.x.array[:] =u_n.x.array    
        # Update solution at previous time step (u_n)
        u_n.x.array[:] = u_n.x.array +1.0*uh.x.array 
        u_n.x.array[u_n.x.array<0.0000]=0


        diff = u_nl - u_n
        # H1_diff = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx)), op=MPI.SUM)
        L2_diff = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM)
        # print("||u1-u0||_H^1:", abs(np.sqrt(H1_diff)))
        print("||u1-u0||_L^2:", abs(np.sqrt(L2_diff) ))

    
    print(f"iteration = {i}")
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(fft_result, extent=[-4, 4, -4, 4], origin='lower', cmap='viridis')
    # plt.title('FFT Convolution Result')
    # plt.colorbar()
    # plt.scatter([0.5], [0.5], color='red')  # Mark the point of interest        
    
    #midpoint method
    # u_n.x.array[:]=2*u_n.x.array-u_nm.x.array
    # u_n.x.array[u_n.x.array<0.0000]=0
    u_nm.x.array[:] =u_n.x.array[:]
    # print(uh.x.array[:])
    t += dt
    # u_n.x.array[u_n.x.array<0.0000]=0
    # Zero out values less than 0

    # H1 errors
    diff = u_n - u_exact
    H1_diff = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx)), op=MPI.SUM)
    H1_exact = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_exact), ufl.grad(u_exact)) * ufl.dx)), op=MPI.SUM)
    print("Relative H1 error of FEM solution:", abs(np.sqrt(H1_diff) / np.sqrt(H1_exact)))

    # L2 errors
    L2_diff = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), op=MPI.SUM)
    L2_exact = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx)), op=MPI.SUM)
    print("Relative L2 error of FEM solution:", abs(np.sqrt(L2_diff) ))
        
    # Write solution to file
    if i%10==0:
        xdmf.write_function(u_n, t)
# # 
xdmf.close()



