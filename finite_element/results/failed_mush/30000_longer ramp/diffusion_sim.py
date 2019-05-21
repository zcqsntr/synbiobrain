
import pyopencl as cl
import numpy as np
import os
import time
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from matplotlib import animation, rc
from IPython.display import HTML
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
from scipy.integrate import odeint
from matplotlib.animation import FFMpegWriter

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = (self.end - self.start)


class SynBioBrainGrid(object):

    """A Grid class that can handle two dimensional flat and surface grids."""

    def __init__(self, vertices, elements, dtype='float64'):
        """
        Initialize a grid.

        Parameters
        ----------
        vertices: (N, 3) Array
            Vertices are stored in an (N, 3) array of floating point numbers.
        elements: (N, 3) Array
            Elements are stored in an (N, 3) array of type 'np.int32'.
        dtype: string
          dtype is either 'float64' or 'float32'. Internally, all structures and the
          vertices are converted to the format specified by 'dtype'. This is useful
          for OpenCL computations on GPUs that do not properly support double precision
          ('float64') types

        """

        # set initial class properties
        self.mesh = {}
        self.mesh['vertices'] = vertices
        self.mesh['elements'] = elements
        self.dtype = dtype;
        self.jacobians = []
        self.iters = 0

        # calculate inverse_jacobian_transpose and integration elements in parrellel
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # get number of elements
        n = len(self.elements)

        # use relevant precision
        if self.dtype == 'float64':
            dtype = 'double'
            allVertices = np.array([self.corners(element_index) for element_index in range(n)], dtype = np.float64).reshape(n,3,3) # input array
            invJTs = np.empty((n*2*2), dtype = np.float64) #output array
            detJs = np.empty(n, dtype = np.float64)  # output array
        elif self.dtype == 'float32':
            dtype = 'float'
            allVertices = np.array([self.corners(element_index) for element_index in range(n)], dtype = np.float32).reshape(n,3,3) # input array
            invJTs = np.empty((n*2*2), dtype = np.float32) # output array
            detJs = np.empty(n, dtype = np.float32) # output array

        allVerticesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=allVertices) # input  buffer
        invJTsb = cl.Buffer(ctx, mf.WRITE_ONLY, invJTs.nbytes) # output buffer
        detJsb = cl.Buffer(ctx, mf.WRITE_ONLY, detJs.nbytes) # output buffer

        # build kernel
        kernel = """
            __kernel void jacobian(
            __global const {0} *allVerticesb, __global {0} *invJTsb, __global {0} *detJsb)
            {{
                int n = get_global_id(0);

                {0} jacobian[4];
                jacobian[0] = allVerticesb[9*n + 3] - allVerticesb[9*n];
                jacobian[1] = allVerticesb[9*n + 6] - allVerticesb[9*n];
                jacobian[2] = allVerticesb[9*n + 4] - allVerticesb[9*n + 1];
                jacobian[3] = allVerticesb[9*n + 7] - allVerticesb[9*n + 1];

                {0} detJ = jacobian[0] * jacobian[3] - jacobian[1] * jacobian[2];

                invJTsb[4*n] = jacobian[3]/detJ;
                invJTsb[4*n + 1] = -jacobian[2]/detJ;
                invJTsb[4*n + 2] = -jacobian[1]/detJ;
                invJTsb[4*n + 3] = jacobian[0]/detJ;
                detJsb[n] = detJ;
            }}
                """.format(dtype)

        prg = cl.Program(ctx, kernel).build()
        prg.jacobian(queue, (n,), None, allVerticesb, invJTsb, detJsb) #run kernel
        cl.enqueue_copy(queue, detJs, detJsb) # copy result from buffer
        cl.enqueue_copy(queue, invJTs, invJTsb) # copy result from buffer

        # set the remaining class variables
        self.inverse_jacobian_transposes = invJTs.reshape(n,2,2)
        self.integration_elements = detJs
        self.iterates = []
        self.nIters = 0
        self.callIters = 0



    @property
    def vertices(self):
        """Return the vertices."""
        return np.array(self.mesh['vertices'])

    @property
    def elements(self):
        """Return the elemenets."""
        return np.array(self.mesh['elements'])

    def corners(self, element_index):
        """Return the corners of a given element as (3, 3) array"""
        corner_ind = map(int, self.elements[element_index]) # get indexes of corners
        return np.array([self.vertices[c] for c in corner_ind]) # get vertices at the indeces

    def integration_element(self, element_index):
        """Return |det J| for a given element."""
        return self.integration_elements[element_index]


    def normal(self, element_index):
        """Return the exterior pointing normal of an element."""
        vertices = self.corners(element_index) # get vertices
        return np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0]) # return the normal to two side vectors

    def jacobian(self, element_index):
        '''Returns the (2x2) jacobian of a given element'''

        if self.jacobians == []: # if the jacobians havent been calculated yet
            # create context, queue and memory flags
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            mf = cl.mem_flags

            # get number of elements
            n = len(self.elements)

            # create arrays
            if self.dtype == 'float64':
                allVertices = np.array([self.corners(element_index) for element_index in range(n)], dtype = np.float64).reshape(n,3,3) # input array
                jacobians = np.zeros((n*2*2), dtype = np.float64) # output array
                dtype = 'double'

            elif self.dtype == 'float32':
                allVertices = np.array([self.corners(element_index) for element_index in range(n)], dtype = np.float32).reshape(n,3,3) # input array
                jacobians = np.zeros((n*2*2), dtype = np.float32) # output array
                dtype = 'float'

            # create buffers
            allVerticesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=allVertices) # input  buffer
            jacobiansb = cl.Buffer(ctx, mf.WRITE_ONLY, jacobians.nbytes) # output buffer

            # build kernel
            kernel = """
                __kernel void jacobian(
                __global const {0} *allVerticesb, __global {0} *jacobiansb)
                {{
                    int n = get_global_id(0);
                    jacobiansb[4*n] = allVerticesb[9*n + 3] - allVerticesb[9*n];
                    jacobiansb[4*n + 1] = allVerticesb[9*n + 6] - allVerticesb[9*n];
                    jacobiansb[4*n + 2] = allVerticesb[9*n + 4] - allVerticesb[9*n + 1];
                    jacobiansb[4*n + 3] = allVerticesb[9*n + 7] - allVerticesb[9*n + 1];

                }}
                    """.format(dtype)
            prg = cl.Program(ctx, kernel).build()

            prg.jacobian(queue, (n,), None, allVerticesb, jacobiansb) #run kernel
            cl.enqueue_copy(queue, jacobians, jacobiansb) # copy result from buffer
            self.jacobians = jacobians.reshape(n,2,2)

        return self.jacobians[element_index,:,:] #return J as a square matrix


    def inverse_jacobian_transpose(self, element_index): #
        """Returns the (2x2) inverse Jacobian transpose of an element."""
        return self.inverse_jacobian_transposes[element_index,:,:] #return J as a square matrix

    @classmethod
    def from_file(cls, file_name, dtype ='float64'):
        """Read a mesh from a vtk file."""

        with open(file_name) as file:
            line = file.readline()
            k = 0
            # read line by line
            while line:
                split = line.split() # split by whitespace

                if len(split) > 0 and split[0] == 'POINTS': # start reading in the points
                    number_of_points = int(split[1]) # number of points to be read
                    points = [] # to store all points

                    # read in all the points
                    for _ in range(number_of_points):
                        line = file.readline()
                        line = line.split()
                        points.append(list(map(float, line))) # add points as lists of integers

                elif len(split) > 0 and split[0] == 'CELLS': #read in elements
                    number_of_els = int(split[1]) # number of elements to be read
                    elements = [] # to store elements
                    for _ in range(number_of_els):
                        line = file.readline()
                        line = line.split()

                        not_three = 0
                        if line[0] == '3': # if it is a triangle
                            elements.append(list(map(float, line[1:4])))
                        else:
                            not_three += 1
                    print('not_three: ', not_three)

                line = file.readline()
        points = np.array(points)
        elements = np.array(elements)

        print("done from_file")
        print('nodes: ', len(points))
        print('elements: ', len(elements))

        return cls(points, elements, dtype)

    def get_boundary_nodes(self):
        '''
        CHANGE FOR SQUARE
        '''
        boundaries = np.array([[-2], [2]])
        boundary_nodes = []
        for index, node in enumerate(self.vertices):

            if np.any(np.abs(boundaries - node[0:2]) < 0.0000001): #will get all nodes on the square
                boundary_nodes.append(index)

            '''
            elif abs(node[0]) < 0.0000001 and  node[1] <= 0:
                boundary_nodes.append(index)
            elif abs(node[1]) < 0.0000001 and  node[0] <= 0:
                boundary_nodes.append(index)
            '''


        return np.array(boundary_nodes)


    def right_hand_side(self, f):
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # get number of elements
        n = len(self.elements)

        # number of nodes
        nodes = len(self.vertices)


        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        # create arrays
        if self.dtype == 'float64':
            nodal_f_values = np.array([f(vertice) for vertice in self.vertices], dtype = np.float64)
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            global_rhs = np.zeros((n,3), dtype = np.float64) # output array
            summed_rhs = np.zeros(nodes, dtype = np.float64)
            dtype = 'double'

        elif self.dtype == 'float32':
            nodal_f_values = np.array([f(vertice) for vertice in self.vertices], dtype = np.float32)# input array
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            global_rhs = np.zeros((n,3), dtype = np.float32) # output array
            summed_rhs = np.zeros(nodes, dtype = np.float32)
            dtype = 'float'

        # create buffers
        nodal_f_valuesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nodal_f_values) # input  buffer
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer
        global_rhsb = cl.Buffer(ctx, mf.WRITE_ONLY, global_rhs.nbytes) # output buffer

        # build kernel
        kernel = """
            __kernel void get_rhs(
            __global const {0} *nodal_f_valuesb,__global const {0} *inv_J_Tsb,__global const {0} *integration_elementsb,__global const int *element_mappingb, __global {0} *global_rhsb)
            {{

                int n = get_global_id(0);

                // get the mapping for this element
                int element_indeces[3]; //seeems to be working
                element_indeces[0] = element_mappingb[3*n];
                element_indeces[1] = element_mappingb[3*n + 1];
                element_indeces[2] = element_mappingb[3*n + 2];

                {0} reference_M[3][3] = {{{{1.0/12.0,1.0/24.0,1.0/24.0}},
                                            {{1.0/24.0, 1.0/12.0, 1.0/24.0}},
                                            {{1.0/24.0, 1.0/24.0, 1.0/12.0}}}};



                {0} inv_J_T[2][2];    // seems to be working
                inv_J_T[0][0] = inv_J_Tsb[4*n];
                inv_J_T[0][1] = inv_J_Tsb[4*n+1];
                inv_J_T[1][0] = inv_J_Tsb[4*n+2];
                inv_J_T[1][1] = inv_J_Tsb[4*n+3];

                {0} local_M[3][3];
                // transform reference dels into local dels
                for(int i = 0; i < 3; i ++){{
                    for(int j = 0; j<3;j++){{
                        local_M[i][j] = reference_M[i][j] * integration_elementsb[i];
                    }}
                }}

                // get local function values
                {0} local_f[3];
                local_f[0] = nodal_f_valuesb[element_indeces[0]];
                local_f[1] = nodal_f_valuesb[element_indeces[1]];
                local_f[2] = nodal_f_valuesb[element_indeces[2]];



                // get local rhs
                {0} local_rhs[3];
                local_rhs[0] = local_M[0][0] * local_f[0] + local_M[0][1] * local_f[1] + local_M[0][2] * local_f[2];
                local_rhs[1] = local_M[1][0] * local_f[0] + local_M[1][1] * local_f[1] + local_M[1][2] * local_f[2];
                local_rhs[2] = local_M[2][0] * local_f[0] + local_M[2][1] * local_f[1] + local_M[2][2] * local_f[2];


                for(int i = 0; i<3; i++){{
                    global_rhsb[3*n + i] = local_rhs[i];
                }}


            }}
                """.format(dtype)


        sum_kernel = """
        __kernel void sum_rhs(
            __global const {0} *global_yb,__global const int *element_mappingb, const int number_of_elements, __global {0} *summed_yb)
            {{
                for(int i = 0; i < number_of_elements; i++) {{

                    for (int j = 0; j < 3; j++) {{

                        summed_yb[element_mappingb[3*i + j]] += global_yb[3*i + j];
                    }}
                }}
            }}

        """.format(dtype)


        prg = cl.Program(ctx, kernel).build()
        prg.get_rhs(queue, (n,), None, nodal_f_valuesb, inv_J_Tsb, integration_elementsb, element_mappingb, global_rhsb) #run kernel

        summed_rhsb = cl.Buffer(ctx, mf.WRITE_ONLY, summed_rhs.nbytes) # output buffer


        prg2 = cl.Program(ctx, sum_kernel).build()
        prg2.sum_rhs(queue, (1,), None, global_rhsb, element_mappingb, np.int32(n), summed_rhsb) #run kernel

        cl.enqueue_copy(queue, summed_rhs, summed_rhsb) # copy result from buffer
        boundary_nodes = self.get_boundary_nodes()

        summed_rhs[boundary_nodes] = 0

        return summed_rhs

    def get_sol(self,f):

        rhs = self.right_hand_side(f)

        n = len(self.mesh['vertices'])
        LO = LinearOperator((n,n), matvec = self.matvec)

        sol, status = cg(LO, rhs,tol = 10**(-4), callback = self.call_me)

        self.rhs = rhs
        self.sol = sol

        if status != 0:
            print("grid exited with code: " + str(status))
        print("done get_sol")


        iters = self.iters

        return sol, iters


    def get_vertex_integration_elements(self):
        # get the integration elements mapped to vertices:
        element_mappings = []

        for i in range(len(self.elements)):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])


        vertex_int_els = np.zeros(len(self.vertices))
        for element in range(len(self.elements)): # for each element
            for j in range(3):
                vertex = element_mappings[element][j]
                vertex_int_els[vertex] += 1/6 * self.integration_element(element) # 1/6 from the integral

        return vertex_int_els

    def get_test_point(self, test_point):

        min_distance = np.linalg.norm(self.vertices[0] - test_point)
        closest_point = self.vertices[0]

        closest_index = 0

        for index, vertice in enumerate(self.vertices):
            dist = np.linalg.norm(vertice - test_point)

            if dist < min_distance:
                closest_point = vertice
                closest_index = index
                min_distance = dist
                if min_distance == 0: return closest_index
        return closest_index


    def call_me(self, xk):

        xk = np.array(xk)
        xk[0:128] = 0
        self.iterates.append(xk)
        self.callIters +=1

    def matvec(self, x, buffer = False):
        self.nIters += 1
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # get number of elements
        n = len(self.elements)

        # number of nodes
        nodes = len(self.vertices)

        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        # create arrays
        if self.dtype == 'float64':
            global_x = np.array(x, dtype = np.float64).reshape(nodes) # input array
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            global_y = np.zeros((n,3), dtype = np.float64) # output array
            summed_y = np.zeros(nodes, dtype = np.float64)
            dtype = 'double'

        elif self.dtype == 'float32':
            global_x = np.array(x, dtype = np.float32).reshape(nodes) # input array
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            global_y = np.zeros((n,3), dtype = np.float32) # output array
            summed_y = np.zeros(nodes, dtype = np.float32)
            dtype = 'float'

        # create buffers
        global_xb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=global_x) # input  buffer
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer
        global_yb = cl.Buffer(ctx, mf.WRITE_ONLY, global_y.nbytes) # output buffer

        # build kernel
        get_y_kernel = """
           __kernel void get_y(
            __global const {0} *global_xb,__global const {0} *inv_J_Tsb,__global const {0} *integration_elementsb,__global const int *element_mappingb, __global {0} *global_yb)
            {{
                int n = get_global_id(0);

                // get the mapping for this element
                int element_indeces[3]; //seeems to be working
                element_indeces[0] = element_mappingb[3*n];
                element_indeces[1] = element_mappingb[3*n + 1];
                element_indeces[2] = element_mappingb[3*n + 2];

                int reference_dels[3][2] = {{{{-1,-1}},
                                            {{1, 0}},
                                            {{0, 1}}}};

                {0} inv_J_T[2][2];    // seems to be working
                inv_J_T[0][0] = inv_J_Tsb[4*n];
                inv_J_T[0][1] = inv_J_Tsb[4*n+1];
                inv_J_T[1][0] = inv_J_Tsb[4*n+2];
                inv_J_T[1][1] = inv_J_Tsb[4*n+3];

                // transform reference dels into local dels
                {0} local_dels[3][2]; //seems to be working
                local_dels[0][0] = inv_J_T[0][0] * reference_dels[0][0] + inv_J_T[0][1] * reference_dels[0][1];
                local_dels[0][1] = inv_J_T[1][0] * reference_dels[0][0] + inv_J_T[1][1] * reference_dels[0][1];

                local_dels[1][0] = inv_J_T[0][0] * reference_dels[1][0] + inv_J_T[0][1] * reference_dels[1][1];
                local_dels[1][1] = inv_J_T[1][0] * reference_dels[1][0] + inv_J_T[1][1] * reference_dels[1][1];

                local_dels[2][0] = inv_J_T[0][0] * reference_dels[2][0] + inv_J_T[0][1] * reference_dels[2][1];
                local_dels[2][1] = inv_J_T[1][0] * reference_dels[2][0] + inv_J_T[1][1] * reference_dels[2][1];

                // get local A
                {0} local_A[3][3]; //seeems to work
                for(int i = 0; i<3; i++){{
                    for(int j = 0; j< 3; j++) {{
                        {0} dot_prod = local_dels[i][0]*local_dels[j][0] + local_dels[i][1]*local_dels[j][1];

                        local_A[i][j] = dot_prod * integration_elementsb[n];
                    }}
                }}

                // get local x
                {0} local_x[3];
                local_x[0] = global_xb[element_indeces[0]];
                local_x[1] = global_xb[element_indeces[1]];
                local_x[2] = global_xb[element_indeces[2]];

                // get local y
                {0} local_y[3];
                local_y[0] = local_A[0][0] * local_x[0] + local_A[0][1] * local_x[1] + local_A[0][2] * local_x[2];
                local_y[1] = local_A[1][0] * local_x[0] + local_A[1][1] * local_x[1] + local_A[1][2] * local_x[2];
                local_y[2] = local_A[2][0] * local_x[0] + local_A[2][1] * local_x[1] + local_A[2][2] * local_x[2];


                // put the local ys into global y

                for(int i = 0; i<3; i++){{
                    global_yb[3*n + i] = local_y[i];
                }}

            }}

             """.format(dtype)



        sum_kernel = """
        __kernel void sum_y(
            __global const {0} *global_yb,__global const int *element_mappingb, const int number_of_elements, __global {0} *summed_yb)
            {{
                for(int i = 0; i < number_of_elements; i++) {{

                    for (int j = 0; j < 3; j++) {{
                        summed_yb[element_mappingb[3*i + j]] += global_yb[3*i + j];
                    }}
                }}
            }}

        """.format(dtype)


        prg = cl.Program(ctx, get_y_kernel).build()
        prg.get_y(queue, (n,), None, global_xb, inv_J_Tsb, integration_elementsb, element_mappingb, global_yb) #run kernel

        summed_yb = cl.Buffer(ctx, mf.WRITE_ONLY, summed_y.nbytes) # output buffer


        prg2 = cl.Program(ctx, sum_kernel).build()
        prg2.sum_y(queue, (1,), None, global_yb, element_mappingb, np.int32(n), summed_yb) #run kernel


        cl.enqueue_copy(queue, summed_y, summed_yb) # copy result from buffer

        boundary_nodes = self.get_boundary_nodes()

        #summed_y[boundary_nodes] = 0

        return summed_y

    def get_global_A(self, D):
        #verified to get the same answer as matvec
        element_mappings = []

        for i in range(len(self.elements)):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        reference_dels = np.array([[-1, -1], [1, 0], [0, 1]]).T

        global_A = np.zeros((len(self.vertices), len(self.vertices)))

        for n in range(len(self.elements)):
            local_A = np.zeros((3, 3))
            inv_JT = self.inverse_jacobian_transpose(n)
            detJ = self.integration_elements[n]

            el_map = element_mappings[n]

            local_dels = np.matmul(inv_JT, reference_dels)

            for i in range(3):
                for j in range(3):
                    global_A[el_map[i], el_map[j]] +=  np.dot(local_dels[:, i], local_dels[:, j]) * detJ

        return global_A * D

    def synbio_brain(self, u0, dt, n_steps, input_vertices, output_vertices, ps, D, production_rate, is_automata = False):
        # M doesnt depend on U so can calculate it once outside the loop

        n = len(self.elements)
        nodes = len(self.vertices)
        print(n, nodes)
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        vertex_int_els = self.get_vertex_integration_elements()
        # create arrays
        if self.dtype == 'float64':
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array

            dtype = 'double'
            global_M = np.zeros((nodes,nodes), dtype = np.float64)
            all_u = np.zeros((n_steps+1, nodes), dtype = np.float64)

            dt = np.float64(dt)
            unsummed_Au = np.zeros((n,3), dtype = np.float64) # output array
            summed_Au = np.zeros(nodes, dtype = np.float64)
            current_u = np.array(u0, dtype = np.float64)

            vertex_int_els = np.array(vertex_int_els, dtype = np.float64)
            D = np.float64(D)
            #vertex_production_rates = np.float64(vertex_production_rates, dtype = np.float64)
            ps = np.array(ps, dtype = np.float64)
            production_rate = np.float64(production_rate)

            vertex_times_on = np.zeros(nodes, dtype = np.float64)
            vertex_times_off = np.zeros(nodes, dtype = np.float64)

        elif self.dtype == 'float32':

            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array
            dtype = 'float'
            global_M = np.zeros((nodes,nodes), dtype = np.float32)
            all_u = np.zeros(( n_steps + 1, nodes), dtype = np.float32)
            dt = np.float32(dt)
            unsummed_Au = np.zeros((n,3), dtype = np.float32) # output array

            summed_Au = np.zeros(nodes, dtype = np.float32)
            current_u = np.array(u0, dtype = np.float32)

            ps = np.array(ps, dtype = np.float32)
            #vertex_production_rates = np.float32(vertex_production_rates, dtype = np.float32)
            D = np.float32(D)
            production_rate = np.float32(production_rate)
            vertex_int_els = np.array(vertex_int_els, dtype = np.float32)

            vertex_times_on = np.zeros(nodes, dtype = np.float32)
            vertex_times_off = np.zeros(nodes, dtype = np.float32)


        input_vertices = np.array(input_vertices, dtype = np.int32)

        output_vertices = np.array(output_vertices, dtype = np.int32)

        #activated = np.zeros((nodes,1), dtype = np.int32)

        # set activated to be the input nodes initially
        activated = input_vertices
        all_activated = np.zeros(( n_steps + 1, nodes), dtype = np.int32)
        boundary_nodes = np.array(self.get_boundary_nodes(), dtype = np.int32)
        element_mappings = np.array(element_mappings, dtype = np.int32) # input array
        boundary_flags = np.zeros(len(self.vertices), dtype = np.int32)

        boundary_flags[boundary_nodes] = 1
        u0[boundary_nodes] = 0

        all_u[0,:] = current_u

        get_M_kernel = """
        __kernel void get_M(__global const {0} *integration_elementsb,
                __global const int *element_mappingb, const int number_of_nodes,
                __global {0} *global_Mb) {{

                int n = get_global_id(0);

                {0} reference_M[3][3] =  {{{{ 1.0/12.0, 1.0/24.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/12.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/24.0, 1.0/12.0 }}}};

                int element_indeces[3];
                for(int i = 0; i<3; i++) {{
                    element_indeces[i] = element_mappingb[n*3 + i];
                }}

                for(int i = 0; i < 3; i++) {{
                    for(int j = 0; j < 3; j++) {{

                        global_Mb[number_of_nodes * element_indeces[i] + element_indeces[j]] = reference_M[i][j] * integration_elementsb[n];
                    }}
                }}

        }}


        """.format(dtype)


        # build kernel
        get_Au_kernel = """
           __kernel void get_y(
            __global const {0} *global_xb,__global const {0} *inv_J_Tsb,__global const {0} *integration_elementsb,__global const int *element_mappingb, __global {0} *global_yb, const {0} D)
            {{

                int n = get_global_id(0);


                // get the mapping for this element
                int element_indeces[3]; //seeems to be working
                element_indeces[0] = element_mappingb[3*n];
                element_indeces[1] = element_mappingb[3*n + 1];
                element_indeces[2] = element_mappingb[3*n + 2];

                int reference_dels[3][2] = {{{{-1,-1}},
                                            {{1, 0}},
                                            {{0, 1}}}};

                {0} inv_J_T[2][2];    // seems to be working
                inv_J_T[0][0] = inv_J_Tsb[4*n];
                inv_J_T[0][1] = inv_J_Tsb[4*n+1];
                inv_J_T[1][0] = inv_J_Tsb[4*n+2];
                inv_J_T[1][1] = inv_J_Tsb[4*n+3];

                // transform reference dels into local dels
                {0} local_dels[3][2]; //seems to be working
                local_dels[0][0] = inv_J_T[0][0] * reference_dels[0][0] + inv_J_T[0][1] * reference_dels[0][1];
                local_dels[0][1] = inv_J_T[1][0] * reference_dels[0][0] + inv_J_T[1][1] * reference_dels[0][1];

                local_dels[1][0] = inv_J_T[0][0] * reference_dels[1][0] + inv_J_T[0][1] * reference_dels[1][1];
                local_dels[1][1] = inv_J_T[1][0] * reference_dels[1][0] + inv_J_T[1][1] * reference_dels[1][1];

                local_dels[2][0] = inv_J_T[0][0] * reference_dels[2][0] + inv_J_T[0][1] * reference_dels[2][1];
                local_dels[2][1] = inv_J_T[1][0] * reference_dels[2][0] + inv_J_T[1][1] * reference_dels[2][1];


                // get local A
                {0} local_A[3][3]; //seeems to work
                for(int i = 0; i<3; i++){{
                    for(int j = 0; j< 3; j++) {{
                        {0} dot_prod = local_dels[i][0]*local_dels[j][0] + local_dels[i][1]*local_dels[j][1];

                        local_A[i][j] =  D * dot_prod * integration_elementsb[n];
                    }}
                }}

                // get local x
                {0} local_x[3];
                local_x[0] = global_xb[element_indeces[0]];
                local_x[1] = global_xb[element_indeces[1]];
                local_x[2] = global_xb[element_indeces[2]];

                // get local y
                {0} local_y[3];
                local_y[0] = local_A[0][0] * local_x[0] + local_A[0][1] * local_x[1] + local_A[0][2] * local_x[2];
                local_y[1] = local_A[1][0] * local_x[0] + local_A[1][1] * local_x[1] + local_A[1][2] * local_x[2];
                local_y[2] = local_A[2][0] * local_x[0] + local_A[2][1] * local_x[1] + local_A[2][2] * local_x[2];

                // put the local ys into global y

                for(int i = 0; i<3; i++){{
                    global_yb[3*n + i] = local_y[i];
                }}

            }}

             """.format(dtype)

        add_next_u_kernel = """__kernel void add_next_u(
        __global {0} *Aub,__global const {0} *Mb, __global const int *boundary_flagsb, const {0} dt, const int number_of_nodes, const int t, __global {0} *all_ub,  __global {0} *current_ub, __global {0} * vertex_integration_elementsb, __global int * activated_nodesb, __global {0} * times_onb, __global {0} * times_offb, {0} production_rate)
        {{

            /*
            calculates du and adds the net u to time_series
            activated_nodesb: one hot vector of the activated nodes, only output vertices get changed, if input vertices set to one will stay on
            */

            // calculate un_1 and add to buffer of all us

            int i = get_global_id(0);

            // calculate the expression level from ramping up and down
            {0} exp_level;

            if (activated_nodesb[i] == 1) {{ // if cell is on
                {0} time_passed = t - times_onb[i];
                exp_level = time_passed * 0.001;  


                if(exp_level > 1.0) {{
                    exp_level = 1.0;
                }}


            }} else if(activated_nodesb[i] == 0) {{ // if off might be ramping down
                {0} time_passed = t - times_offb[i];
                exp_level = 1.0 - time_passed * 0.1;

                if (exp_level < 0.0) {{
                    exp_level = 0.0;
                }}



            }}

            {0} production = 1.0 /Mb[i] * activated_nodesb[i] * vertex_integration_elementsb[i] * exp_level * production_rate;
            {0} diffusion = - 1.0/Mb[i] * Aub[i];
            {0} degradation;

            if (current_ub[i] > 0.0005) {{
                degradation = -0.0005;
            }} else {{
                degradation = -current_ub[i];
            }}

            {0} du = dt * (production + diffusion);

            if (boundary_flagsb[i] == 1){{
                du = 0;
            }}

            current_ub[i] += du;

            all_ub[number_of_nodes*(t+1) + i] = current_ub[i];
            Aub[i] = 0;



        }}
            """.format(dtype)


        # adds boundary conditions on the value of u, value of du BCs set in add_next_u
        boundary_kernel = """__kernel void boundary(
        __global {0} *Aub,__global const int *boundary_nodes)
        {{

            int n = get_global_id(0);
            int b = boundary_nodes[n];

            Aub[b] = 0;

        }}
            """.format(dtype)


        add_activated_kernel = """__kernel void add_activated(
        __global {0} *current_ub, __global int *activatedb, __global int * all_activatedb, __global int * input_verticesb, __global int * output_verticesb, const int number_of_nodes, const int t, __global {0} * psb){{

            /*
            tests the output vertices and returns a one hot vecotr fo the ones that are activated
            */
            int n = get_global_id(0);

            {0} u = current_ub[n];
            // gett activated output nodes, change between mushroom and bacnd pass using different ps

            if (output_verticesb[n] == 1) {{ // if this is an output vertex

                if (activatedb[n] == 0 && psb[0] < u && u < psb[1]) {{ // if vertex not activated
                    activatedb[n] = 1;


                }} else if(activatedb[n] == 1 && (u < psb[2] || u > psb[3])) {{ // if vertex is activated
                    activatedb[n] = 0;

                }}
            }}

            /*
            if(t > 5000){{
                if(input_verticesb[n] == 1){{
                    activatedb[n] = 0;
                }}
            }}


            if(n==0){{
                float sum = 0;
                for (int i = 0; i< 10178; i++){{
                    sum += activatedb[i];
                }}
                printf("%d ", sum);
            }}
            */

            all_activatedb[number_of_nodes*(t+1) + n] = activatedb[n];
        }}
            """.format(dtype)



        sum_kernel = """
        __kernel void sum(
            __global const {0} *global_yb,__global const int *element_mappingb, const int number_of_elements, __global {0} *summed_yb)
            {{
                for(int i = 0; i < number_of_elements; i++) {{

                    for (int j = 0; j < 3; j++) {{

                        summed_yb[element_mappingb[3*i + j]] += global_yb[3*i + j];
                    }}
                }}
            }}

        """.format(dtype)



        global_Mb = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=global_M)

        # Au buffers
        # create buffers
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer

        prg = cl.Program(ctx, get_M_kernel).build()

        prg.get_M(queue, (len(self.elements),), None,integration_elementsb, element_mappingb, np.int32(len(self.vertices)), global_Mb)

        cl.enqueue_copy(queue, global_M, global_Mb) # copy result from buffer, copying at the beggining wont slow down much and i dont have much time
        print('size: ',global_M.size)
        # make M lumpy, np is fast
        M = np.sum(global_M,1)
        print(M)
        global_A = self.get_global_A(D)
        inv_A = np.linalg.inv(global_A.reshape((len(self.vertices), len(self.vertices))))
        inv_M = np.linalg.inv(global_M.reshape((len(self.vertices), len(self.vertices))))

        # create buffers
        Mb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M) # input buffer
        boundary_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=boundary_nodes) # input buffer
        boundary_flagsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=boundary_flags)
        input_verticesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_vertices)
        output_verticesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=output_vertices)
        unsummed_Aub = cl.Buffer(ctx, mf.READ_WRITE, unsummed_Au.nbytes) # output buffer
        summed_Aub = cl.Buffer(ctx, mf.READ_WRITE, summed_Au.nbytes)
        all_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_u) # output buffer
        current_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=current_u) # input buffer
        activatedb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=activated)
        all_activatedb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_activated)
        psb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps)
        vertex_int_elsb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_int_els)
        #vertex_production_ratesb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_production_rates)


        vertex_times_onb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_times_on)
        vertex_times_offb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_times_off)

        Au_prg = cl.Program(ctx, get_Au_kernel).build()
        sum_prg = cl.Program(ctx, sum_kernel).build()
        add_next_u_prg = cl.Program(ctx, add_next_u_kernel).build()
        boundary_prg = cl.Program(ctx, boundary_kernel).build()
        add_activated_prg = cl.Program(ctx, add_activated_kernel).build()
        all_u = all_u.reshape(-1)

        for t in range(n_steps):
            if t%100 == 0:
                print(t)
            # get Au buffer for the time step
            Au_prg.get_y(queue, (n,),None,current_ub, inv_J_Tsb, integration_elementsb, element_mappingb, unsummed_Aub, D)
            sum_prg.sum(queue, (1,), None, unsummed_Aub, element_mappingb, np.int32(n), summed_Aub)
            # sum the Au
            boundary_prg.boundary(queue, (len(boundary_nodes),), None, summed_Aub, boundary_nodesb)
            # calculate next u and add to the buffer of all us

            add_next_u_prg.add_next_u(queue, (nodes,), None, summed_Aub, Mb, boundary_flagsb, dt, np.int32(nodes), np.int32(t), all_ub, current_ub, vertex_int_elsb, activatedb, vertex_times_onb, vertex_times_offb,  production_rate)#run kernel
            add_activated_prg.add_activated(queue, (nodes,), None, current_ub, activatedb, all_activatedb, input_verticesb, output_verticesb, np.int32(nodes), np.int32(t), psb)


        cl.enqueue_copy(queue, all_u, all_ub) # copy result from buffer

        all_u = all_u.reshape(n_steps+1,nodes)
        cl.enqueue_copy(queue, all_activated, all_activatedb)
        all_activated = all_activated.reshape(n_steps+1,nodes)

        global_Mb.release()
        inv_J_Tsb.release()
        integration_elementsb.release()
        element_mappingb.release()
        Mb.release()
        boundary_nodesb.release()
        boundary_flagsb.release()
        input_verticesb.release()
        output_verticesb.release()
        unsummed_Aub.release()
        summed_Aub.release()
        all_ub.release()
        current_ub.release()
        activatedb.release()
        all_activatedb.release()
        psb.release()

        return all_u, all_activated


class Grid(object):
    """A Grid class that can handle two dimensional flat and surface grids."""

    def __init__(self, vertices, elements, dtype='float64'):
        """
        Initialize a grid.

        Parameters
        ----------
        vertices: (N, 3) Array
            Vertices are stored in an (N, 3) array of floating point numbers.
        elements: (N, 3) Array
            Elements are stored in an (N, 3) array of type 'np.int32'.
        dtype: string
          dtype is either 'float64' or 'float32'. Internally, all structures and the
          vertices are converted to the format specified by 'dtype'. This is useful
          for OpenCL computations on GPUs that do not properly support double precision
          ('float64') types

        """

        # set initial class properties
        self.mesh = {}
        self.mesh['vertices'] = vertices
        self.mesh['elements'] = elements
        self.dtype = dtype;
        self.jacobians = []
        self.iters = 0

        # calculate inverse_jacobian_transpose and integration elements in parrellel
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # get number of elements
        n = len(self.elements)

        # use relevant precision
        if self.dtype == 'float64':
            dtype = 'double'
            allVertices = np.array([self.corners(element_index) for element_index in range(n)], dtype = np.float64).reshape(n,3,3) # input array
            invJTs = np.empty((n*2*2), dtype = np.float64) #output array
            detJs = np.empty(n, dtype = np.float64)  # output array
        elif self.dtype == 'float32':
            dtype = 'float'
            allVertices = np.array([self.corners(element_index) for element_index in range(n)], dtype = np.float32).reshape(n,3,3) # input array
            invJTs = np.empty((n*2*2), dtype = np.float32) # output array
            detJs = np.empty(n, dtype = np.float32) # output array

        allVerticesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=allVertices) # input  buffer
        invJTsb = cl.Buffer(ctx, mf.WRITE_ONLY, invJTs.nbytes) # output buffer
        detJsb = cl.Buffer(ctx, mf.WRITE_ONLY, detJs.nbytes) # output buffer

        # build kernel
        kernel = """
            __kernel void jacobian(
            __global const {0} *allVerticesb, __global {0} *invJTsb, __global {0} *detJsb)
            {{
                int n = get_global_id(0);

                {0} jacobian[4];
                jacobian[0] = allVerticesb[9*n + 3] - allVerticesb[9*n];
                jacobian[1] = allVerticesb[9*n + 6] - allVerticesb[9*n];
                jacobian[2] = allVerticesb[9*n + 4] - allVerticesb[9*n + 1];
                jacobian[3] = allVerticesb[9*n + 7] - allVerticesb[9*n + 1];

                {0} detJ = jacobian[0] * jacobian[3] - jacobian[1] * jacobian[2];

                invJTsb[4*n] = jacobian[3]/detJ;
                invJTsb[4*n + 1] = -jacobian[2]/detJ;
                invJTsb[4*n + 2] = -jacobian[1]/detJ;
                invJTsb[4*n + 3] = jacobian[0]/detJ;
                detJsb[n] = detJ;
            }}
                """.format(dtype)

        prg = cl.Program(ctx, kernel).build()
        prg.jacobian(queue, (n,), None, allVerticesb, invJTsb, detJsb) #run kernel
        cl.enqueue_copy(queue, detJs, detJsb) # copy result from buffer
        cl.enqueue_copy(queue, invJTs, invJTsb) # copy result from buffer

        # set the remaining class variables
        self.inverse_jacobian_transposes = invJTs.reshape(n,2,2)
        self.integration_elements = detJs
        self.iterates = []
        self.nIters = 0
        self.callIters = 0



    @property
    def vertices(self):
        """Return the vertices."""
        return np.array(self.mesh['vertices'])

    @property
    def elements(self):
        """Return the elemenets."""
        return np.array(self.mesh['elements'])

    def corners(self, element_index):
        """Return the corners of a given element as (3, 3) array"""
        corner_ind = map(int, self.elements[element_index]) # get indexes of corners
        return np.array([self.vertices[c] for c in corner_ind]) # get vertices at the indeces

    def integration_element(self, element_index):
        """Return |det J| for a given element."""
        return self.integration_elements[element_index]


    def normal(self, element_index):
        """Return the exterior pointing normal of an element."""
        vertices = self.corners(element_index) # get vertices
        return np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0]) # return the normal to two side vectors

    def jacobian(self, element_index):
        '''Returns the (2x2) jacobian of a given element'''

        if self.jacobians == []: # if the jacobians havent been calculated yet
            # create context, queue and memory flags
            ctx = cl.create_some_context()
            queue = cl.CommandQueue(ctx)
            mf = cl.mem_flags

            # get number of elements
            n = len(self.elements)

            # create arrays
            if self.dtype == 'float64':
                allVertices = np.array([self.corners(element_index) for element_index in range(n)], dtype = np.float64).reshape(n,3,3) # input array
                jacobians = np.zeros((n*2*2), dtype = np.float64) # output array
                dtype = 'double'

            elif self.dtype == 'float32':
                allVertices = np.array([self.corners(element_index) for element_index in range(n)], dtype = np.float32).reshape(n,3,3) # input array
                jacobians = np.zeros((n*2*2), dtype = np.float32) # output array
                dtype = 'float'

            # create buffers
            allVerticesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=allVertices) # input  buffer
            jacobiansb = cl.Buffer(ctx, mf.WRITE_ONLY, jacobians.nbytes) # output buffer

            # build kernel
            kernel = """
                __kernel void jacobian(
                __global const {0} *allVerticesb, __global {0} *jacobiansb)
                {{
                    int n = get_global_id(0);
                    jacobiansb[4*n] = allVerticesb[9*n + 3] - allVerticesb[9*n];
                    jacobiansb[4*n + 1] = allVerticesb[9*n + 6] - allVerticesb[9*n];
                    jacobiansb[4*n + 2] = allVerticesb[9*n + 4] - allVerticesb[9*n + 1];
                    jacobiansb[4*n + 3] = allVerticesb[9*n + 7] - allVerticesb[9*n + 1];

                }}
                    """.format(dtype)
            prg = cl.Program(ctx, kernel).build()

            prg.jacobian(queue, (n,), None, allVerticesb, jacobiansb) #run kernel
            cl.enqueue_copy(queue, jacobians, jacobiansb) # copy result from buffer
            self.jacobians = jacobians.reshape(n,2,2)

        return self.jacobians[element_index,:,:] #return J as a square matrix


    def inverse_jacobian_transpose(self, element_index): #
        """Returns the (2x2) inverse Jacobian transpose of an element."""
        return self.inverse_jacobian_transposes[element_index,:,:] #return J as a square matrix

    @classmethod
    def from_file(cls, file_name, dtype ='float64'):
        """Read a mesh from a vtk file."""

        with open(file_name) as file:
            line = file.readline()
            k = 0
            # read line by line
            while line:
                split = line.split() # split by whitespace

                if len(split) > 0 and split[0] == 'POINTS': # start reading in the points
                    number_of_points = int(split[1]) # number of points to be read
                    points = [] # to store all points

                    # read in all the points
                    for _ in range(number_of_points):
                        line = file.readline()
                        line = line.split()
                        points.append(list(map(float, line))) # add points as lists of integers

                elif len(split) > 0 and split[0] == 'CELLS': #read in elements
                    number_of_els = int(split[1]) # number of elements to be read
                    elements = [] # to store elements
                    for _ in range(number_of_els):
                        line = file.readline()
                        line = line.split()

                        not_three = 0
                        if line[0] == '3': # if it is a triangle
                            elements.append(list(map(float, line[1:4])))
                        else:
                            not_three += 1
                    print('not_three: ', not_three)

                line = file.readline()
        points = np.array(points)
        elements = np.array(elements)
        print("done from_file")

        return cls(points, elements, dtype)

    def get_boundary_nodes(self):
        '''
        CHANGE FOR SQUARE
        '''
        boundaries = np.array([[-1], [1]])
        boundary_nodes = []
        for index, node in enumerate(self.vertices):

            if np.any(np.abs(boundaries - node[0:2]) < 0.0000001): #will get all nodes on the square
                boundary_nodes.append(index)

            '''
            elif abs(node[0]) < 0.0000001 and  node[1] <= 0:
                boundary_nodes.append(index)
            elif abs(node[1]) < 0.0000001 and  node[0] <= 0:
                boundary_nodes.append(index)
            '''


        return np.array(boundary_nodes)


    def right_hand_side(self, f):
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # get number of elements
        n = len(self.elements)

        # number of nodes
        nodes = len(self.vertices)


        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        # create arrays
        if self.dtype == 'float64':
            nodal_f_values = np.array([f(vertice) for vertice in self.vertices], dtype = np.float64)
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            global_rhs = np.zeros((n,3), dtype = np.float64) # output array
            summed_rhs = np.zeros(nodes, dtype = np.float64)
            dtype = 'double'

        elif self.dtype == 'float32':
            nodal_f_values = np.array([f(vertice) for vertice in self.vertices], dtype = np.float32)# input array
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            global_rhs = np.zeros((n,3), dtype = np.float32) # output array
            summed_rhs = np.zeros(nodes, dtype = np.float32)
            dtype = 'float'

        # create buffers
        nodal_f_valuesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nodal_f_values) # input  buffer
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer
        global_rhsb = cl.Buffer(ctx, mf.WRITE_ONLY, global_rhs.nbytes) # output buffer

        # build kernel
        kernel = """
            __kernel void get_rhs(
            __global const {0} *nodal_f_valuesb,__global const {0} *inv_J_Tsb,__global const {0} *integration_elementsb,__global const int *element_mappingb, __global {0} *global_rhsb)
            {{

                int n = get_global_id(0);

                // get the mapping for this element
                int element_indeces[3]; //seeems to be working
                element_indeces[0] = element_mappingb[3*n];
                element_indeces[1] = element_mappingb[3*n + 1];
                element_indeces[2] = element_mappingb[3*n + 2];

                {0} reference_M[3][3] = {{{{1.0/12.0,1.0/24.0,1.0/24.0}},
                                            {{1.0/24.0, 1.0/12.0, 1.0/24.0}},
                                            {{1.0/24.0, 1.0/24.0, 1.0/12.0}}}};



                {0} inv_J_T[2][2];    // seems to be working
                inv_J_T[0][0] = inv_J_Tsb[4*n];
                inv_J_T[0][1] = inv_J_Tsb[4*n+1];
                inv_J_T[1][0] = inv_J_Tsb[4*n+2];
                inv_J_T[1][1] = inv_J_Tsb[4*n+3];

                {0} local_M[3][3];
                // transform reference dels into local dels
                for(int i = 0; i < 3; i ++){{
                    for(int j = 0; j<3;j++){{
                        local_M[i][j] = reference_M[i][j] * integration_elementsb[i];
                    }}
                }}

                // get local function values
                {0} local_f[3];
                local_f[0] = nodal_f_valuesb[element_indeces[0]];
                local_f[1] = nodal_f_valuesb[element_indeces[1]];
                local_f[2] = nodal_f_valuesb[element_indeces[2]];






                // get local rhs
                {0} local_rhs[3];
                local_rhs[0] = local_M[0][0] * local_f[0] + local_M[0][1] * local_f[1] + local_M[0][2] * local_f[2];
                local_rhs[1] = local_M[1][0] * local_f[0] + local_M[1][1] * local_f[1] + local_M[1][2] * local_f[2];
                local_rhs[2] = local_M[2][0] * local_f[0] + local_M[2][1] * local_f[1] + local_M[2][2] * local_f[2];


                for(int i = 0; i<3; i++){{
                    global_rhsb[3*n + i] = local_rhs[i];
                }}


            }}
                """.format(dtype)


        sum_kernel = """
        __kernel void sum_rhs(
            __global const {0} *global_yb,__global const int *element_mappingb, const int number_of_elements, __global {0} *summed_yb)
            {{
                for(int i = 0; i < number_of_elements; i++) {{

                    for (int j = 0; j < 3; j++) {{

                        summed_yb[element_mappingb[3*i + j]] += global_yb[3*i + j];
                    }}
                }}
            }}

        """.format(dtype)


        prg = cl.Program(ctx, kernel).build()
        prg.get_rhs(queue, (n,), None, nodal_f_valuesb, inv_J_Tsb, integration_elementsb, element_mappingb, global_rhsb) #run kernel

        summed_rhsb = cl.Buffer(ctx, mf.WRITE_ONLY, summed_rhs.nbytes) # output buffer


        prg2 = cl.Program(ctx, sum_kernel).build()
        prg2.sum_rhs(queue, (1,), None, global_rhsb, element_mappingb, np.int32(n), summed_rhsb) #run kernel

        cl.enqueue_copy(queue, summed_rhs, summed_rhsb) # copy result from buffer
        boundary_nodes = self.get_boundary_nodes()

        summed_rhs[boundary_nodes] = 0

        return summed_rhs

    def get_sol(self,f):

        rhs = self.right_hand_side(f)

        n = len(self.mesh['vertices'])
        LO = LinearOperator((n,n), matvec = self.matvec)

        sol, status = cg(LO, rhs,tol = 10**(-4), callback = self.call_me)

        self.rhs = rhs
        self.sol = sol

        if status != 0:
            print("grid exited with code: " + str(status))
        print("done get_sol")


        iters = self.iters

        return sol, iters


    def get_vertex_integration_elements(self):
        # get the integration elements mapped to vertices:
        element_mappings = []

        for i in range(len(self.elements)):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])


        vertex_int_els = np.zeros(len(self.vertices))
        for element in range(len(self.elements)): # for each element
            for j in range(3):
                vertex = element_mappings[element][j]
                vertex_int_els[vertex] += 1/6 * self.integration_element(element) # 1/6 from the integral

        return vertex_int_els

    def get_test_point(self, test_point):

        min_distance = np.linalg.norm(self.vertices[0] - test_point)
        closest_point = self.vertices[0]

        closest_index = 0

        for index, vertice in enumerate(self.vertices):
            dist = np.linalg.norm(vertice - test_point)

            if dist < min_distance:
                closest_point = vertice
                closest_index = index
                min_distance = dist
                if min_distance == 0: return closest_index
        return closest_index


    def call_me(self, xk):

        xk = np.array(xk)
        xk[0:128] = 0
        self.iterates.append(xk)
        self.callIters +=1

    def matvec(self, x, buffer = False):
        self.nIters += 1
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        # get number of elements
        n = len(self.elements)

        # number of nodes
        nodes = len(self.vertices)

        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        # create arrays
        if self.dtype == 'float64':
            global_x = np.array(x, dtype = np.float64).reshape(nodes) # input array
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            global_y = np.zeros((n,3), dtype = np.float64) # output array
            summed_y = np.zeros(nodes, dtype = np.float64)
            dtype = 'double'

        elif self.dtype == 'float32':
            global_x = np.array(x, dtype = np.float32).reshape(nodes) # input array
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            global_y = np.zeros((n,3), dtype = np.float32) # output array
            summed_y = np.zeros(nodes, dtype = np.float32)
            dtype = 'float'

        # create buffers
        global_xb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=global_x) # input  buffer
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer
        global_yb = cl.Buffer(ctx, mf.WRITE_ONLY, global_y.nbytes) # output buffer

        # build kernel
        get_y_kernel = """
           __kernel void get_y(
            __global const {0} *global_xb,__global const {0} *inv_J_Tsb,__global const {0} *integration_elementsb,__global const int *element_mappingb, __global {0} *global_yb)
            {{
                int n = get_global_id(0);

                // get the mapping for this element
                int element_indeces[3]; //seeems to be working
                element_indeces[0] = element_mappingb[3*n];
                element_indeces[1] = element_mappingb[3*n + 1];
                element_indeces[2] = element_mappingb[3*n + 2];

                int reference_dels[3][2] = {{{{-1,-1}},
                                            {{1, 0}},
                                            {{0, 1}}}};

                {0} inv_J_T[2][2];    // seems to be working
                inv_J_T[0][0] = inv_J_Tsb[4*n];
                inv_J_T[0][1] = inv_J_Tsb[4*n+1];
                inv_J_T[1][0] = inv_J_Tsb[4*n+2];
                inv_J_T[1][1] = inv_J_Tsb[4*n+3];

                // transform reference dels into local dels
                {0} local_dels[3][2]; //seems to be working
                local_dels[0][0] = inv_J_T[0][0] * reference_dels[0][0] + inv_J_T[0][1] * reference_dels[0][1];
                local_dels[0][1] = inv_J_T[1][0] * reference_dels[0][0] + inv_J_T[1][1] * reference_dels[0][1];

                local_dels[1][0] = inv_J_T[0][0] * reference_dels[1][0] + inv_J_T[0][1] * reference_dels[1][1];
                local_dels[1][1] = inv_J_T[1][0] * reference_dels[1][0] + inv_J_T[1][1] * reference_dels[1][1];

                local_dels[2][0] = inv_J_T[0][0] * reference_dels[2][0] + inv_J_T[0][1] * reference_dels[2][1];
                local_dels[2][1] = inv_J_T[1][0] * reference_dels[2][0] + inv_J_T[1][1] * reference_dels[2][1];

                // get local A
                {0} local_A[3][3]; //seeems to work
                for(int i = 0; i<3; i++){{
                    for(int j = 0; j< 3; j++) {{
                        {0} dot_prod = local_dels[i][0]*local_dels[j][0] + local_dels[i][1]*local_dels[j][1];

                        local_A[i][j] = dot_prod * integration_elementsb[n];
                    }}
                }}

                // get local x
                {0} local_x[3];
                local_x[0] = global_xb[element_indeces[0]];
                local_x[1] = global_xb[element_indeces[1]];
                local_x[2] = global_xb[element_indeces[2]];

                // get local y
                {0} local_y[3];
                local_y[0] = local_A[0][0] * local_x[0] + local_A[0][1] * local_x[1] + local_A[0][2] * local_x[2];
                local_y[1] = local_A[1][0] * local_x[0] + local_A[1][1] * local_x[1] + local_A[1][2] * local_x[2];
                local_y[2] = local_A[2][0] * local_x[0] + local_A[2][1] * local_x[1] + local_A[2][2] * local_x[2];


                // put the local ys into global y

                for(int i = 0; i<3; i++){{
                    global_yb[3*n + i] = local_y[i];
                }}

            }}

             """.format(dtype)



        sum_kernel = """
        __kernel void sum_y(
            __global const {0} *global_yb,__global const int *element_mappingb, const int number_of_elements, __global {0} *summed_yb)
            {{
                for(int i = 0; i < number_of_elements; i++) {{

                    for (int j = 0; j < 3; j++) {{
                        summed_yb[element_mappingb[3*i + j]] += global_yb[3*i + j];
                    }}
                }}
            }}

        """.format(dtype)


        prg = cl.Program(ctx, get_y_kernel).build()
        prg.get_y(queue, (n,), None, global_xb, inv_J_Tsb, integration_elementsb, element_mappingb, global_yb) #run kernel

        summed_yb = cl.Buffer(ctx, mf.WRITE_ONLY, summed_y.nbytes) # output buffer


        prg2 = cl.Program(ctx, sum_kernel).build()
        prg2.sum_y(queue, (1,), None, global_yb, element_mappingb, np.int32(n), summed_yb) #run kernel


        cl.enqueue_copy(queue, summed_y, summed_yb) # copy result from buffer

        boundary_nodes = self.get_boundary_nodes()

        #summed_y[boundary_nodes] = 0

        return summed_y

    def synbio_brain_cheating(self, u0, dt, n_steps, input_vertices, output_vertices, ps):
        # M doesnt depend on U so can calculate it once outside the loop
        reference_M = 1/24 * np.array([[2,1,1],
                                        [1,2,1],
                                        [1,1,2]])
        nodes = len(self.vertices)
        n = len(self.elements)

        # us stores AHL concentration
        us = np.zeros((n_steps + 1, nodes))
        us[0,:] = u0
        # get all the local Ms and map them to the global M

        #activated stores the flouresing nodes
        activated_ts = np.zeros((n_steps + 1, nodes))

        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])


        # create arrays
        if self.dtype == 'float64':
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array
            dtype = 'double'
            global_M = np.zeros((nodes,nodes), dtype = np.float64)
            input_vertices = np.array(input_vertices, dtype = np.float64)




        elif self.dtype == 'float32':
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array
            dtype = 'float'
            global_M = np.zeros((nodes,nodes), dtype = np.float32)
            input_vertices = np.array(input_vertices, dtype = np.float32)


        boundary_nodes = np.array(self.get_boundary_nodes(), dtype = np.int32)
        element_mappings = np.array(element_mappings, dtype = np.int32)

        get_M_kernel = """
        __kernel void get_M(__global const {0} *integration_elementsb,
                __global const int *element_mappingb, const int number_of_nodes,
                __global {0} *global_Mb) {{

                int n = get_global_id(0);

                {0} reference_M[3][3] =  {{{{ 1.0/12.0, 1.0/24.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/12.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/24.0, 1.0/12.0 }}}};

                int element_indeces[3];

                for(int i = 0; i<3; i++){{
                    element_indeces[i] = element_mappingb[n*3 + i];
                }}

                for(int i = 0; i < 3; i++){{
                    for(int j = 0; j < 3; j++){{

                        global_Mb[number_of_nodes * element_indeces[i] + element_indeces[j]] = reference_M[i][j] * integration_elementsb[n];
                    }}
                }}

        }}


        """.format(dtype)

        # create buffers
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer
        global_Mb = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=global_M)

        prg = cl.Program(ctx, get_M_kernel).build()

        prg.get_M(queue, (len(self.elements),), None,integration_elementsb, element_mappingb, np.int32(len(self.vertices)), global_Mb)

        cl.enqueue_copy(queue, global_M, global_Mb) # copy result from buffer

        # make M lumpy, np is fast
        M = np.sum(global_M,1)

        u = u0
        production_rate = 1

        for t in range(n_steps):
            print(t)
            Au = self.matvec(u)
            du = -dt * 1/M * Au + dt*input_vertices * production_rate
            u += du
            activated = (np.logical_and(u< 2*threshold, u > threshold)) * output_vertices

            u[boundary_nodes] = 0
            activated_ts[t+1, :] = activated
            us[t+1, :] = u

        return us, activated_ts

    def get_global_A(self, D):
        #verified to get the same answer as matvec
        element_mappings = []

        for i in range(len(self.elements)):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        reference_dels = np.array([[-1, -1], [1, 0], [0, 1]]).T

        global_A = np.zeros((len(self.vertices), len(self.vertices)))

        for n in range(len(self.elements)):
            local_A = np.zeros((3, 3))
            inv_JT = self.inverse_jacobian_transpose(n)
            detJ = self.integration_elements[n]

            el_map = element_mappings[n]

            local_dels = np.matmul(inv_JT, reference_dels)

            for i in range(3):
                for j in range(3):
                    global_A[el_map[i], el_map[j]] +=  np.dot(local_dels[:, i], local_dels[:, j]) * detJ

        return global_A * D

    def synbio_brain(self, u0, dt, n_steps, input_vertices, output_vertices, ps, D, production_rate, is_automata = False):
        # M doesnt depend on U so can calculate it once outside the loop
        print('--------------------------------')
        sys.exit()
        n = len(self.elements)
        nodes = len(self.vertices)
        print(n, nodes)
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        vertex_int_els = self.get_vertex_integration_elements()
        # create arrays
        if self.dtype == 'float64':
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array

            dtype = 'double'
            global_M = np.zeros((nodes,nodes), dtype = np.float64)
            all_u = np.zeros((n_steps+1, nodes), dtype = np.float64)

            dt = np.float64(dt)
            unsummed_Au = np.zeros((n,3), dtype = np.float64) # output array
            summed_Au = np.zeros(nodes, dtype = np.float64)
            current_u = np.array(u0, dtype = np.float64)
            activated = np.zeros((nodes,1), dtype = np.float64)
            all_activated = np.zeros(( n_steps + 1, nodes), dtype = np.float64)
            input_vertices = np.array(input_vertices, dtype = np.float64)
            output_vertices = np.array(output_vertices, dtype = np.float64)

            vertex_int_els = np.array(vertex_int_els, dtype = np.float64)
            D = np.float64(D)
            #vertex_production_rates = np.float64(vertex_production_rates, dtype = np.float64)
            ps = np.array(ps, dtype = np.float64)
            production_rate = np.float64(production_rate)

            vertex_times_on = np.zeros(nodes, dtype = np.float64)
            vertex_times_off = np.zeros(nodes, dtype = np.float64)

        elif self.dtype == 'float32':

            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array
            dtype = 'float'
            global_M = np.zeros((nodes,nodes), dtype = np.float32)
            all_u = np.zeros(( n_steps + 1, nodes), dtype = np.float32)
            dt = np.float32(dt)
            unsummed_Au = np.zeros((n,3), dtype = np.float32) # output array

            summed_Au = np.zeros(nodes, dtype = np.float32)
            current_u = np.array(u0, dtype = np.float32)
            activated = np.zeros((nodes,1), dtype = np.float32)

            input_vertices = np.array(input_vertices, dtype = np.float32)
            all_activated = np.zeros((n_steps+1, nodes), dtype = np.float32)
            output_vertices = np.array(output_vertices, dtype = np.float32)
            ps = np.array(ps, dtype = np.float32)
            #vertex_production_rates = np.float32(vertex_production_rates, dtype = np.float32)
            D = np.float32(D)
            production_rate = np.float32(production_rate)
            vertex_int_els = np.array(vertex_int_els, dtype = np.float32)

            vertex_times_on = np.zeros(nodes, dtype = np.float32)
            vertex_times_off = np.zeros(nodes, dtype = np.float32)

        boundary_nodes = np.array(self.get_boundary_nodes(), dtype = np.int32)
        element_mappings = np.array(element_mappings, dtype = np.int32) # input array
        boundary_flags = np.zeros(len(self.vertices), dtype = np.int32)
        cell_states = np.ones(nodes, dtype = np.int32)
        boundary_flags[boundary_nodes] = 1
        u0[boundary_nodes] = 0

        all_u[0,:] = current_u
        print(all_u.dtype)
        print(current_u.dtype)

        get_M_kernel = """
        __kernel void get_M(__global const {0} *integration_elementsb,
                __global const int *element_mappingb, const int number_of_nodes,
                __global {0} *global_Mb) {{

                int n = get_global_id(0);

                {0} reference_M[3][3] =  {{{{ 1.0/12.0, 1.0/24.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/12.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/24.0, 1.0/12.0 }}}};

                int element_indeces[3];
                for(int i = 0; i<3; i++) {{
                    element_indeces[i] = element_mappingb[n*3 + i];
                }}

                for(int i = 0; i < 3; i++) {{
                    for(int j = 0; j < 3; j++) {{

                        global_Mb[number_of_nodes * element_indeces[i] + element_indeces[j]] = reference_M[i][j] * integration_elementsb[n];
                    }}
                }}

        }}


        """.format(dtype)


        # build kernel
        get_Au_kernel = """
           __kernel void get_y(
            __global const {0} *global_xb,__global const {0} *inv_J_Tsb,__global const {0} *integration_elementsb,__global const int *element_mappingb, __global {0} *global_yb, const {0} D)
            {{

                int n = get_global_id(0);
                printf("%f ", global_xb[n])

                // get the mapping for this element
                int element_indeces[3]; //seeems to be working
                element_indeces[0] = element_mappingb[3*n];
                element_indeces[1] = element_mappingb[3*n + 1];
                element_indeces[2] = element_mappingb[3*n + 2];

                int reference_dels[3][2] = {{{{-1,-1}},
                                            {{1, 0}},
                                            {{0, 1}}}};

                {0} inv_J_T[2][2];    // seems to be working
                inv_J_T[0][0] = inv_J_Tsb[4*n];
                inv_J_T[0][1] = inv_J_Tsb[4*n+1];
                inv_J_T[1][0] = inv_J_Tsb[4*n+2];
                inv_J_T[1][1] = inv_J_Tsb[4*n+3];

                // transform reference dels into local dels
                {0} local_dels[3][2]; //seems to be working
                local_dels[0][0] = inv_J_T[0][0] * reference_dels[0][0] + inv_J_T[0][1] * reference_dels[0][1];
                local_dels[0][1] = inv_J_T[1][0] * reference_dels[0][0] + inv_J_T[1][1] * reference_dels[0][1];

                local_dels[1][0] = inv_J_T[0][0] * reference_dels[1][0] + inv_J_T[0][1] * reference_dels[1][1];
                local_dels[1][1] = inv_J_T[1][0] * reference_dels[1][0] + inv_J_T[1][1] * reference_dels[1][1];

                local_dels[2][0] = inv_J_T[0][0] * reference_dels[2][0] + inv_J_T[0][1] * reference_dels[2][1];
                local_dels[2][1] = inv_J_T[1][0] * reference_dels[2][0] + inv_J_T[1][1] * reference_dels[2][1];


                // get local A
                {0} local_A[3][3]; //seeems to work
                for(int i = 0; i<3; i++){{
                    for(int j = 0; j< 3; j++) {{
                        {0} dot_prod = local_dels[i][0]*local_dels[j][0] + local_dels[i][1]*local_dels[j][1];

                        local_A[i][j] =  D * dot_prod * integration_elementsb[n];
                    }}
                }}

                // get local x
                {0} local_x[3];
                local_x[0] = global_xb[element_indeces[0]];
                local_x[1] = global_xb[element_indeces[1]];
                local_x[2] = global_xb[element_indeces[2]];

                // get local y
                {0} local_y[3];
                local_y[0] = local_A[0][0] * local_x[0] + local_A[0][1] * local_x[1] + local_A[0][2] * local_x[2];
                local_y[1] = local_A[1][0] * local_x[0] + local_A[1][1] * local_x[1] + local_A[1][2] * local_x[2];
                local_y[2] = local_A[2][0] * local_x[0] + local_A[2][1] * local_x[1] + local_A[2][2] * local_x[2];

                // put the local ys into global y

                for(int i = 0; i<3; i++){{
                    global_yb[3*n + i] = local_y[i];
                }}

            }}

             """.format(dtype)

        add_next_u_kernel = """__kernel void add_next_u(
        __global {0} *Aub,__global const {0} *Mb, __global const int *boundary_flagsb, const {0} dt, const int number_of_nodes, const int t, __global {0} *all_ub,  __global {0} *current_ub, __global {0} *input_verticesb, __global {0} * vertex_integration_elementsb, __global int * statesb, __global {0} * times_onb, __global {0} * times_offb, {0} production_rate)
        {{

            // calculate un_1 and add to buffer of all us

            int i = get_global_id(0);

            // calculate the expression level from ramping up and down
            {0} exp_level;

            if (statesb[i] == 1) {{ // if cell is on
                {0} time_passed = t - times_onb[i];
                exp_level = time_passed * 0.1;  // ramp = 0.1

                if(exp_level > 1.0) {{
                    exp_level = 1.0;
                }}

            }} else if(statesb[i] == 0) {{ // if off might be ramping down
                {0} time_passed = t - times_offb[i];
                exp_level = 1.0 - time_passed * 0.01;

                if (exp_level < 0.0) {{
                    exp_level = 0.0;
                }}
            }}

            {0} production = 1.0 /Mb[i] * input_verticesb[i] * vertex_integration_elementsb[i] * exp_level * production_rate;
            {0} diffusion = - 1.0/Mb[i] * Aub[i];
            {0} du = dt * (production + diffusion);

            if (boundary_flagsb[i] == 1){{
                //du = 0;
            }}

            current_ub[i] += du;

            all_ub[number_of_nodes*(t+1) + i] = current_ub[i];
            Aub[i] = 0;


        }}
            """.format(dtype)


        # adds boundary conditions on the value of u, value of du BCs set in add_next_u
        boundary_kernel = """__kernel void boundary(
        __global {0} *Aub,__global const int *boundary_nodes)
        {{

            int n = get_global_id(0);
            int b = boundary_nodes[n];

            Aub[b] = 0;

        }}
            """.format(dtype)


        add_activated_kernel = """__kernel void add_activated(
        __global {0} *current_ub, __global {0} *a\

            // gett activated output nodes, change between mushroom and bacnd pass using different ps

            if (output_verticesb[n] == 1) {{ // if this is an output vertex

                if (activatedb[n] == 0 && psb[0] < u && u < psb[1]) {{ // if vertex not activated
                    activatedb[n] = 1;

                }} else if(activatedb[n] == 1 && (u < psb[2] || u > psb[3])) {{ // if vertex is activated
                    activatedb[n] = 0;
                }}
            }}

            all_activatedb[number_of_nodes*(t+1) + n] = activatedb[n];
        }}
            """.format(dtype)



        sum_kernel = """
        __kernel void sum(
            __global const {0} *global_yb,__global const int *element_mappingb, const int number_of_elements, __global {0} *summed_yb)
            {{
                for(int i = 0; i < number_of_elements; i++) {{

                    for (int j = 0; j < 3; j++) {{

                        summed_yb[element_mappingb[3*i + j]] += global_yb[3*i + j];
                    }}
                }}
            }}

        """.format(dtype)

        global_Mb = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=global_M)

        # Au buffers
        # create buffers
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer

        prg = cl.Program(ctx, get_M_kernel).build()

        prg.get_M(queue, (len(self.elements),), None,integration_elementsb, element_mappingb, np.int32(len(self.vertices)), global_Mb)

        cl.enqueue_copy(queue, global_M, global_Mb) # copy result from buffer, copying at the beggining wont slow down much and i dont have much time
        print('size: ',global_M.size)
        # make M lumpy, np is fast
        M = np.sum(global_M,1)
        print(M)
        global_A = self.get_global_A(D)
        inv_A = np.linalg.inv(global_A.reshape((len(self.vertices), len(self.vertices))))
        inv_M = np.linalg.inv(global_M.reshape((len(self.vertices), len(self.vertices))))

        # create buffers
        Mb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M) # input buffer
        boundary_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=boundary_nodes) # input buffer
        boundary_flagsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=boundary_flags)
        input_verticesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_vertices)
        output_verticesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=output_vertices)
        unsummed_Aub = cl.Buffer(ctx, mf.READ_WRITE, unsummed_Au.nbytes) # output buffer
        summed_Aub = cl.Buffer(ctx, mf.READ_WRITE, summed_Au.nbytes)
        all_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_u) # output buffer
        current_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=current_u) # input buffer
        activatedb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=activated)
        all_activatedb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_activated)
        psb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps)
        vertex_int_elsb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_int_els)
        #vertex_production_ratesb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_production_rates)


        vertex_times_onb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_times_on)
        vertex_times_offb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_times_off)
        cell_statesb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cell_states)

        Au_prg = cl.Program(ctx, get_Au_kernel).build()
        sum_prg = cl.Program(ctx, sum_kernel).build()
        add_next_u_prg = cl.Program(ctx, add_next_u_kernel).build()
        boundary_prg = cl.Program(ctx, boundary_kernel).build()
        add_activated_prg = cl.Program(ctx, add_activated_kernel).build()

        print('-----------------------------------', all_u.shape)
        all_u = all_u.reshape(-1)
        print(all_u.shape)

        for t in range(n_steps):
            if t%100 == 0:
                print(t)
            # get Au buffer for the time step
            Au_prg.get_y(queue, (n,),None,current_ub, inv_J_Tsb, integration_elementsb, element_mappingb, unsummed_Aub, D)
            sum_prg.sum(queue, (1,), None, unsummed_Aub, element_mappingb, np.int32(n), summed_Aub)
            # sum the Au
            #boundary_prg.boundary(queue, (len(boundary_nodes),), None, summed_Aub, boundary_nodesb)
            # calculate next u and add to the buffer of all us
            add_next_u_prg.add_next_u(queue, (nodes,), None, summed_Aub, Mb, boundary_flagsb, dt, np.int32(nodes), np.int32(t), all_ub, current_ub, input_verticesb, vertex_int_elsb, cell_statesb, vertex_times_onb, vertex_times_offb,  production_rate)#run kernel

            add_activated_prg.add_activated(queue, (nodes,), None, current_ub, activatedb, all_activatedb, output_verticesb, np.int32(nodes), np.int32(t), psb)

        cl.enqueue_copy(queue, all_u, all_ub) # copy result from buffer

        all_u = all_u.reshape(n_steps+1,nodes)
        cl.enqueue_copy(queue, all_activated, all_activatedb)
        all_activated = all_activated.reshape(n_steps+1,nodes)

        global_Mb.release()
        inv_J_Tsb.release()
        integration_elementsb.release()
        element_mappingb.release()
        Mb.release()
        boundary_nodesb.release()
        boundary_flagsb.release()
        input_verticesb.release()
        output_verticesb.release()
        unsummed_Aub.release()
        summed_Aub.release()
        all_ub.release()
        current_ub.release()
        activatedb.release()
        all_activatedb.release()
        psb.release()

        return all_u, all_activated


    def time_series_cheating(self, u0, dt, n_steps):
        # M doesnt depend on U so can calculate it once outside the loop
        reference_M = 1/24 * np.array([[2,1,1],
                                        [1,2,1],
                                        [1,1,2]])
        nodes = len(self.vertices)
        n = len(self.elements)

        us = np.zeros((n_steps + 1, nodes))
        us[0,:] = u0
        # get all the local Ms and map them to the global M

        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        # create arrays
        if self.dtype == 'float64':
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array
            dtype = 'double'
            global_M = np.zeros((nodes,nodes), dtype = np.float64)


        elif self.dtype == 'float32':
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array

            dtype = 'float'
            global_M = np.zeros((nodes,nodes), dtype = np.float32)

        element_mappings = np.array(element_mappings, dtype = np.int32) # input array


        get_M_kernel = """
        __kernel void get_M(__global const {0} *integration_elementsb,
                __global const int *element_mappingb, const int number_of_nodes,
                __global {0} *global_Mb) {{

                int n = get_global_id(0);

                {0} reference_M[3][3] =  {{{{ 1.0/12.0, 1.0/24.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/12.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/24.0, 1.0/12.0 }}}};

                int element_indeces[3];

                for(int i = 0; i<3; i++){{
                    element_indeces[i] = element_mappingb[n*3 + i];
                }}

                for(int i = 0; i < 3; i++){{
                    for(int j = 0; j < 3; j++){{

                        global_Mb[number_of_nodes * element_indeces[i] + element_indeces[j]] = reference_M[i][j] * integration_elementsb[n];
                    }}
                }}

        }}


        """.format(dtype)

        # create buffers
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer
        global_Mb = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=global_M)



        prg = cl.Program(ctx, get_M_kernel).build()

        prg.get_M(queue, (len(self.elements),), None,integration_elementsb, element_mappingb, np.int32(len(self.vertices)), global_Mb)

        cl.enqueue_copy(queue, global_M, global_Mb) # copy result from buffer

        # make M lumpy, np is fast
        M = np.sum(global_M,1)

        boundary_nodes = self.get_boundary_nodes()

        u = u0
        production_rate = 0.2

        for t in range(n_steps):

            Au = self.matvec(u)
            du = -dt * 1/M * Au
            u += du
            u[boundary_nodes] = 0
            us[t+1, :] = u

        return us

    def time_series(self, u0, dt, n_steps):
        # M doesnt depend on U so can calculate it once outside the loop

        n = len(self.elements)
        nodes = len(self.vertices)
        print(n, nodes)
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        element_mappings = []

        for i in range(n):
            element_mappings.append([int(ind) for ind in self.mesh['elements'][i]])

        # create arrays
        if self.dtype == 'float64':
            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float64) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float64) # input array
            element_mappings = np.array(element_mappings, dtype = np.int32) # input array
            dtype = 'double'
            global_M = np.zeros((nodes,nodes), dtype = np.float64)
            all_u = np.zeros((n_steps+1, nodes), dtype = np.float64)
            dt = np.double(dt)
            unsummed_Au = np.zeros((n,3), dtype = np.float64) # output array
            summed_Au = np.zeros(nodes, dtype = np.float64)
            boundary_nodes = np.array(self.get_boundary_nodes(), dtype = np.int32)
            current_u = np.array(u0, dtype = np.float64)



        elif self.dtype == 'float32':

            inv_J_Ts = np.array([self.inverse_jacobian_transpose(i) for i in range(n)], dtype = np.float32) # input array
            integration_elements = np.array(self.integration_elements, dtype = np.float32) # input array

            dtype = 'float'
            global_M = np.zeros((nodes,nodes), dtype = np.float32)
            all_u = np.zeros(( n_steps + 1, nodes), dtype = np.float32)
            dt = np.float(dt)
            unsummed_Au = np.zeros((n,3), dtype = np.float32) # output array

            summed_Au = np.zeros(nodes, dtype = np.float32)
            current_u = np.array(u0, dtype = np.float32)



        element_mappings = np.array(element_mappings, dtype = np.int32) # input array
        boundary_nodes = np.array(self.get_boundary_nodes(), dtype = np.int32)


        u0[boundary_nodes] = 0


        all_u[0,:] = u0



        get_M_kernel = """
        __kernel void get_M(__global const {0} *integration_elementsb,
                __global const int *element_mappingb, const int number_of_nodes,
                __global {0} *global_Mb) {{

                int n = get_global_id(0);

                {0} reference_M[3][3] =  {{{{ 1.0/12.0, 1.0/24.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/12.0, 1.0/24.0 }},

                                        {{ 1.0/24.0, 1.0/24.0, 1.0/12.0 }}}};

                int element_indeces[3];
                for(int i = 0; i<3; i++){{
                    element_indeces[i] = element_mappingb[n*3 + i];
                }}

                for(int i = 0; i < 3; i++){{
                    for(int j = 0; j < 3; j++){{

                        global_Mb[number_of_nodes * element_indeces[i] + element_indeces[j]] = reference_M[i][j] * integration_elementsb[n];
                    }}
                }}

        }}


        """.format(dtype)


        # build kernel
        get_Au_kernel = """
           __kernel void get_y(
            __global const {0} *global_xb,__global const {0} *inv_J_Tsb,__global const {0} *integration_elementsb,__global const int *element_mappingb, __global {0} *global_yb)
            {{

                int n = get_global_id(0);



                // get the mapping for this element
                int element_indeces[3]; //seeems to be working
                element_indeces[0] = element_mappingb[3*n];
                element_indeces[1] = element_mappingb[3*n + 1];
                element_indeces[2] = element_mappingb[3*n + 2];

                int reference_dels[3][2] = {{{{-1,-1}},
                                            {{1, 0}},
                                            {{0, 1}}}};

                {0} inv_J_T[2][2];    // seems to be working
                inv_J_T[0][0] = inv_J_Tsb[4*n];
                inv_J_T[0][1] = inv_J_Tsb[4*n+1];
                inv_J_T[1][0] = inv_J_Tsb[4*n+2];
                inv_J_T[1][1] = inv_J_Tsb[4*n+3];

                // transform reference dels into local dels
                {0} local_dels[3][2]; //seems to be working
                local_dels[0][0] = inv_J_T[0][0] * reference_dels[0][0] + inv_J_T[0][1] * reference_dels[0][1];
                local_dels[0][1] = inv_J_T[1][0] * reference_dels[0][0] + inv_J_T[1][1] * reference_dels[0][1];

                local_dels[1][0] = inv_J_T[0][0] * reference_dels[1][0] + inv_J_T[0][1] * reference_dels[1][1];
                local_dels[1][1] = inv_J_T[1][0] * reference_dels[1][0] + inv_J_T[1][1] * reference_dels[1][1];

                local_dels[2][0] = inv_J_T[0][0] * reference_dels[2][0] + inv_J_T[0][1] * reference_dels[2][1];
                local_dels[2][1] = inv_J_T[1][0] * reference_dels[2][0] + inv_J_T[1][1] * reference_dels[2][1];


                // get local A
                {0} local_A[3][3]; //seeems to work
                for(int i = 0; i<3; i++){{
                    for(int j = 0; j< 3; j++) {{
                        {0} dot_prod = local_dels[i][0]*local_dels[j][0] + local_dels[i][1]*local_dels[j][1];

                        local_A[i][j] = dot_prod * integration_elementsb[n];
                    }}
                }}


                // get local x
                {0} local_x[3];
                local_x[0] = global_xb[element_indeces[0]];
                local_x[1] = global_xb[element_indeces[1]];
                local_x[2] = global_xb[element_indeces[2]];

                // get local y
                {0} local_y[3];
                local_y[0] = local_A[0][0] * local_x[0] + local_A[0][1] * local_x[1] + local_A[0][2] * local_x[2];
                local_y[1] = local_A[1][0] * local_x[0] + local_A[1][1] * local_x[1] + local_A[1][2] * local_x[2];
                local_y[2] = local_A[2][0] * local_x[0] + local_A[2][1] * local_x[1] + local_A[2][2] * local_x[2];


                // put the local ys into global y

                for(int i = 0; i<3; i++){{
                    global_yb[3*n + i] = local_y[i];
                }}

            }}

             """.format(dtype)

        sum_kernel = """
        __kernel void sum(
            __global const {0} *global_yb,__global const int *element_mappingb, const int number_of_elements, __global {0} *summed_yb)
            {{
                for(int i = 0; i < number_of_elements; i++) {{

                    for (int j = 0; j < 3; j++) {{

                        summed_yb[element_mappingb[3*i + j]] += global_yb[3*i + j];
                    }}
                }}
            }}

        """.format(dtype)


        add_next_u_kernel = """__kernel void add_next_u(
        __global {0} *Aub,__global const {0} *Mb,const {0} dt,const int number_of_nodes, const int t, __global {0} *all_ub,  __global {0} *current_ub)
        {{
            // calculate un_1 and add to buffer of all us

            int i = get_global_id(0);

            current_ub[i] -= dt * 1.0/Mb[i] * Aub[i];


            all_ub[number_of_nodes*(t+1) + i] = current_ub[i];
            Aub[i] = 0;

        }}
            """.format(dtype)

        # currently will only work for 0 boundary conditions
        boundary_kernel = """__kernel void boundary(
        __global {0} *Aub,__global const int *boundary_nodes)
        {{

            int n = get_global_id(0);
            int b = boundary_nodes[n];


            Aub[b] = 0;

        }}
            """.format(dtype)


        global_Mb = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=global_M)

        # Au buffers
        # create buffers
        inv_J_Tsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_J_Ts) # input buffer
        integration_elementsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=integration_elements) # input buffer
        element_mappingb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=element_mappings) # input buffer

        prg = cl.Program(ctx, get_M_kernel).build()

        prg.get_M(queue, (len(self.elements),), None,integration_elementsb, element_mappingb, np.int32(len(self.vertices)), global_Mb)

        cl.enqueue_copy(queue, global_M, global_Mb) # copy result from buffer, cppying at the beggining wont slow down much and i dont have much time

        # make M lumpy, np is fast
        M = np.sum(global_M,1)

        # create buffers
        Mb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M) # input buffer
        boundary_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=boundary_nodes) # input buffer

        unsummed_Aub = cl.Buffer(ctx, mf.READ_WRITE, unsummed_Au.nbytes) # output buffer
        summed_Aub = cl.Buffer(ctx, mf.READ_WRITE, summed_Au.nbytes)
        all_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_u) # output buffer
        current_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u0) # input buffer



        Au_prg = cl.Program(ctx, get_Au_kernel).build()
        sum_prg = cl.Program(ctx, sum_kernel).build()
        add_next_u_prg = cl.Program(ctx, add_next_u_kernel).build()
        boundary_prg = cl.Program(ctx, boundary_kernel).build()

        all_u = all_u.reshape(-1)



        for t in range(n_steps):
            print(t)
            # get Au buffer for the time step
            Au_prg.get_y(queue, (n,),None,current_ub, inv_J_Tsb, integration_elementsb, element_mappingb, unsummed_Aub)

            sum_prg.sum(queue, (1,), None, unsummed_Aub, element_mappingb, np.int32(n), summed_Aub)
            # sum the Au
            boundary_prg.boundary(queue, (len(boundary_nodes),), None, summed_Aub, boundary_nodesb)
            # calculate next u and add to the buffer of all us
            add_next_u_prg.add_next_u(queue, (nodes,), None, summed_Aub, Mb,  dt, np.int32(nodes), np.int32(t), all_ub, current_ub)#run kernel







        cl.enqueue_copy(queue, all_u, all_ub) # copy result from buffer

        all_u = all_u.reshape(n_steps+1,nodes)

        return all_u



class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self._w, self._h)
            self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._frame_sink().write(self.fig.canvas.tostring_argb())
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err))


def get_node_positions(node_dim, grid_corners):
    '''
    gets node positions assuming they are equally spaced
    '''
    grid_size = grid_corners[:, 1] - grid_corners[:, 0]

    spacing = grid_size/(node_dim + 1)
    # use matrix indexing
    row = np.array(np.arange(grid_corners[0, 0], grid_corners[0, 1], spacing[0])[1:])
    col = np.array(np.arange(grid_corners[1, 0], grid_corners[1, 1], spacing[1])[1:])

    positions = [(r,c) for c in col[::-1] for r in row] #backwards through y first to convert between matrix indexing and cartesian

    return positions


def assign_vertices(vertices, node_positions, node_radius):
    '''
    assigns mesh vertices to be part of input or output nodes and returns the
    number of vertices inside each node
    '''
    nodes = []
    node_counts = []
    one_hot_nodes = np.zeros(len(vertices))
    for position in node_positions:
        count = 0
        this_node = []
        for i, point in enumerate(vertices):
            if np.linalg.norm(point[:2] - position) < node_radius:
                count += 1
                this_node.append(i)
                one_hot_nodes[i] = 1
        nodes.append(this_node)
        node_counts.append(count)

    return nodes, one_hot_nodes, node_counts

def get_vertex_production_rates(colony_production_rate, input_vertices, input_counts, n_vertices):
    '''
    calculates and returns the per vertex production rate for each of the input nodes
    '''

    node_production_rates = colony_production_rate/np.array(input_counts) # get the per vertex production rate for each node


    one_hot_production = np.zeros(n_vertices)


    for n, node in enumerate(input_vertices):
        for vertex in node:
            one_hot_production[vertex] = node_production_rates[n]

    return one_hot_production


def get_activated_from_AHL(grid, AHL, ps, node_dim, node_radius, grid_corners, input_indeces):

    node_positions = get_node_positions(node_dim, grid_corners)
    output_indeces = np.array(range(node_dim[0] * node_dim[1]))
    output_indeces = np.delete(output_indeces, input_indeces)
    output_node_positions = [node_positions[i] for i in output_indeces]

    output_vertices, one_hot_out, output_counts = assign_vertices(grid.vertices, output_node_positions, node_radius)

    # get the activated output vertices
    print(np.max(AHL))
    activated = np.zeros(len(one_hot_out))

    for i in range(len(activated)):
        if one_hot_out[i] == 1 and ps[0] < AHL[i] < ps[1]:
            activated[i] = 1

    return activated

def logistic_growth_rate(pop, t, max_growth_rate, carrying_capacity):
    return max_growth_rate  * pop *(1-pop/carrying_capacity)

def get_population_timeseries(initial_pop, max_growth_rate, carrying_capacity, tmax, dt):

    """
    gets the times series of population density for a node
    """


    time_points = np.arange(0, tmax, dt)
    time_series = odeint(logistic_growth_rate, initial_pop, time_points, args = (max_growth_rate, carrying_capacity))

    return time_series

def get_elements_in_nodes(node_vertices, elements):
    '''
    If all of an elements vertices are in a node it is a node element, else it is a boundary element with lower diffusion
    '''
    for i in range(len(elements)):
        element_mappings.append([int(ind) for ind in elements[i]])
    return
