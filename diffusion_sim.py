
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


class SynBioBrainFD(object):

    def __init__(self, grid_corners, nx, ny, dtype='float64'):
        """
        Initialize a grid.

        Parameters
        ---------------------
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


        '''

        THESE MUST BE SET SO THE dx = dy
        '''
        self.grid_corners = np.array(grid_corners)
        self.nx = nx
        self.ny = ny
        self.dx = (grid_corners[0,1] - grid_corners[0,0]) / (nx-1)
        self.dy = (grid_corners[1,1] - grid_corners[1,0]) / (ny-1)

        if not (self.dx == self.dy):
            raise Exception('dx should equal dy')

        self.dtype = dtype
        self.iters = 0

        # calculate inverse_jacobian_transpose and integration elements in parrellel



        # use relevant precision
        if self.dtype == 'float64':
            dtype = 'double'

        elif self.dtype == 'float32':
            dtype = 'float'

        self.iterates = []
        self.nIters = 0
        self.callIters = 0
        self.n_vertices = self.nx * self.ny

        self.build_synbio_brain_kernels(dtype)
        self.vertex_positions = np.array([self.get_vertex_position(i) for i in range(self.n_vertices)])

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

    def get_vertex_position(self,node_number):
        node_coordinates = (node_number//self.nx, node_number % self.nx)

        node_position = (self.grid_corners[1, 0] + node_coordinates[0] * self.dy, self.grid_corners[0, 0] + node_coordinates[1] * self.dx)

        return node_position


    def get_node_positions(self, node_dim, grid_corners):
        '''
        gets node positions assuming they are equally spaced
        '''
        grid_size = grid_corners[:, 1] - grid_corners[:, 0]

        spacing = grid_size/(node_dim)
        # use matrix indexing
        row = np.array(np.arange(grid_corners[0, 0] - spacing[0]/2, grid_corners[0, 1]+ spacing[0]/2, spacing[0])[1:])
        col = np.array(np.arange(grid_corners[1, 0]- spacing[1]/2, grid_corners[1, 1] + spacing[1]/2, spacing[1])[1:])

        positions = [(r,c) for c in col[::-1] for r in row] #backwards through y first to convert between matrix indexing and cartesian

        return positions

    def assign_vertices(self, vertices, node_positions, node_radius):
        '''
        assigns mesh vertices to be part of input or output nodes and returns the
        number of vertices inside each node
        '''
        vertices = np.array(vertices)[:,:2]
        node_positions = np.array(node_positions)

        one_hot_vertices = np.zeros(len(vertices))
        #print('asdfsdafsadfasdf',node_positions[:, None].shape)
        differences = vertices-node_positions[:, None]

        vertices = np.where(np.linalg.norm(differences, axis = 2) < node_radius)[1]
        one_hot_vertices[vertices] = 1

        one_hot_vertices = np.array(one_hot_vertices, dtype = np.int32)

        return vertices, one_hot_vertices

    def get_barrier_vertices(self, vertices, grid_corners, node_dim, nx, ny):
        '''
        assigns mesh vertices to be part of input or output nodes and returns the
        number of vertices inside each node
        '''
        vertices = np.array(vertices)[:,:2]
        grid_width = grid_corners[0][1] - grid_corners[0][0]
        grid_height = grid_corners[1][1] - grid_corners[1][0]

        node_width = grid_width / node_dim[0]
        node_height = grid_height / node_dim[0]

        vertex_width = grid_width / nx
        vertex_height = grid_height / ny

        one_hot_vertices = np.zeros(len(vertices))

        x_barrier = np.where(np.abs(vertices[:,0]%node_width) < vertex_width*2 )[0]
        y_barrier = np.where(np.abs(vertices[:,1]%node_height) < vertex_height*2 )[0]

        barrier_vertices = np.union1d(x_barrier, y_barrier)

        one_hot_vertices[barrier_vertices] = 1

        return barrier_vertices, one_hot_vertices

    def get_boundary_nodes(self):

        boundaries = np.array(self.grid_corners)
        boundary_nodes = []
        for i in range(self.n_vertices):
            position = self.get_vertex_position(i)

            if np.any(np.abs(boundaries.T - np.array(position)) < self.dx/5): #will get all nodes on the square
                boundary_nodes.append(i)

        return np.array(boundary_nodes)

    def build_synbio_brain_kernels(self, dtype):
        # build kernel
        self.get_Au_kernel = """
           __kernel void get_y(
            __global const {0} *xb, __global {0} *yb, __global const int * non_boundary_nodesb, __global const int * barrier_nodesb, const int nx, const {0} dx, const {0} D)
            {{
                // THIS ONLY RUNS FOR NODES THAT ARENT BOUNDARY NODES
                int i = get_global_id(0); // non boundary node number

                // map boundary node number to global node numbering
                int n = non_boundary_nodesb[i];

                {0} d;

                if (barrier_nodesb[n]) {{ // all nodes are output_nodes
                    d = D;
                }} else {{
                    d = D;
                }}

                yb[n] = d*(xb[n-1] - 4 * xb[n] + xb[n+1] + xb[n+nx] + xb[n-nx]) / (dx*dx);

            }}

             """.format(dtype)




        self.add_next_u_kernel = """__kernel void add_next_u(
        __global {0} *Aub, __global const int *boundary_flagsb, const {0} dt, const int number_of_nodes, const int t, __global {0} *all_ub,  __global {0} *current_ub,  __global int * activated_nodesb, __global {0} * times_onb, __global {0} * times_offb, {0} production_rate)
        {{


            /*
            calculates du and adds the next u to time_series
            activated_nodesb: one hot vector of the activated nodes, only output vertices get changed, if input vertices set to one will stay on
            */


            // calculate un_1 and add to buffer of all us

            int i = get_global_id(0);

            // calculate the expression level from ramping up and down
            {0} exp_level;

            if (activated_nodesb[i]  == 1) {{ // if cell is on


                {0} time_passed = t - times_onb[i];
                exp_level = time_passed * 0.001;

                if(exp_level > 1.0) {{
                    exp_level = 1.0;
                }}


            }} else if(activated_nodesb[i]== 0) {{ // if off might be ramping down

                {0} time_passed = t - times_offb[i];
                exp_level = 1.0 - time_passed * 0.1;

                if (exp_level < 0.0) {{
                    exp_level = 0.0;
                }}
            }}

            exp_level = 1.0;


            {0} production =  activated_nodesb[i] * exp_level * production_rate;
            {0} diffusion = Aub[i];
            {0} degradation;

            /*
            if (current_ub[i] > 0.001) {{
                //degradation = -0.01/60.0;
                //degradation = -0.000001*current_ub[i];
                //degradation = -0.02;
                //degradation = -0.000005;

                degradation = -0.000005;
            }} else {{
                degradation = 0;
            }}
            */


            {0} u = current_ub[i];
            {0} du = dt * (production + diffusion);

            if(-du > u){{
                du = -u;
            }}


            current_ub[i] += du;

            all_ub[number_of_nodes*(t) + i] = current_ub[i];
            Aub[i] = 0;

        }}
            """.format(dtype)

        # adds boundary conditions on the value of u, value of du BCs set in add_next_u
        self.boundary_kernel_zero = """__kernel void boundary(
        __global {0} *Aub,__global const int *boundary_nodes, __global const {0} *xb, const int N, const int nx, const {0} dx, const {0} D)
        {{

            int n = get_global_id(0);
            int b = boundary_nodes[n];

            Aub[b] = 0;

        }}
            """.format(dtype)

        self.boundary_kernel_insulating = """__kernel void boundary(
        __global {0} *Aub,__global const int *boundary_nodes, __global const {0} *xb, const int N, const int nx, const {0} dx, const {0} D)
        {{

            int n = get_global_id(0);
            int b = boundary_nodes[n];

            if(b == 0){{
                // top left
                Aub[b] = Aub[nx + 1];

            }} else if (b == nx - 1) {{

                // top right
                Aub[b] = Aub[2*nx-2];


            }} else if (b == N - nx) {{

                // bottom left
                Aub[b] = Aub[N - nx*2 - 1];


            }} else if (b == N - 1) {{
                // bottom right

                Aub[b] = Aub[N-nx-2];

            }} else if (b <= nx - 1) {{
                // top

                Aub[b] = Aub[b+nx];

            }} else if (b >= N - nx) {{
                // bottom

                Aub[b] = Aub[b-nx];


            }} else if (b%nx == 0){{ // if same sign a left node
                // left

                Aub[b] = Aub[b + 1];
                //Aub[b] = 0;


            }} else if (b%nx == nx-1) {{ // if opposite sign a right node
                // right
                Aub[b] = Aub[b - 1];

            }} else {{
                // throw error as all boundary nodes sould be done
                printf("boundary node not done: %d ", b);
            }}


        }}
            """.format(dtype)

        self.boundary_kernel_periodic = """__kernel void boundary(
        __global {0} *Aub,__global const int *boundary_nodes, __global const {0} *xb, const int N, const int nx, const {0} dx, const {0} D)
        {{

            int n = get_global_id(0);
            int b = boundary_nodes[n];

            // all boundary nodes will have the smaller diffusion coefficient

            {0} d = D;

            if(b == 0){{
                // top left
                Aub[b] = d*(-4 * xb[b] + xb[b+1] + xb[b+nx] + xb[N-(nx-b)+1] + xb[b+nx-1]) / (dx*dx);

            }} else if (b == nx - 1) {{

                // top right
                Aub[b] = d*(-4 * xb[b] + xb[b-1] + xb[b+nx] + xb[N-(nx-b)+1] + xb[b-nx+1]) / (dx*dx);

            }} else if (b == N - nx) {{
                // bottom left

                Aub[b] = d*(-4 * xb[b] + xb[b+1] + xb[b-nx] + xb[nx-(N-b)-1] + xb[b+nx-1]) / (dx*dx);

            }} else if (b == N - 1) {{
                // bottom right
                Aub[b] = d*(-4 * xb[b] + xb[b-1] + xb[b-nx] + xb[nx-(N-b)-1] + xb[b-nx+1]) / (dx*dx);

            }} else if (b <= nx - 1) {{
                // top
                Aub[b] = d*(-4 * xb[b] + xb[b-1] + xb[b+1] + xb[b+nx] + xb[N-(nx-b)+1]) / (dx*dx);

            }} else if (b >= N - nx) {{
                // bottom

                Aub[b] = d*(-4 * xb[b] + xb[b-1] + xb[b+1] + xb[b-nx] + xb[nx - (N-b) -1]) / (dx*dx);

            }} else if (b%nx == 0){{ // if same sign a left node
                // left
                Aub[b] = d*(-4 * xb[b] + xb[b-nx] + xb[b+nx] + xb[b+1] + xb[b + nx - 1]) / (dx*dx);

            }} else if (b%nx == nx-1) {{ // if opposite sign a right node
                // right

                Aub[b] = d*(-4 * xb[b] + xb[b-nx] + xb[b+nx] + xb[b-1] + xb[b - (nx-1)]) / (dx*dx);

            }} else {{
                // throw error as all boundary nodes sould be done
                printf("boundary node not done: %d ", b);
            }}

        }}
            """.format(dtype)

        self.add_activated_kernel = """__kernel void add_activated(
        __global {0} *current_ub, __global int *activatedb, __global int * all_activatedb, __global int * input_verticesb, __global int * output_verticesb, const int number_of_nodes, const int t, __global {0} * psb){{

            /*
            tests the output vertices and returns a one hot vecotr fo the ones that are activated
            */
            int n = get_global_id(0);

            {0} u = current_ub[n];
            // gett activated output nodes, change between mushroom and bacnd pass using different ps

            if (output_verticesb[n] == 1 && input_verticesb[n] != 1) {{ // if this is an output vertex

                if (activatedb[n] == 0 && psb[0] < u && u < psb[1]) {{ // if vertex not activated
                    activatedb[n] = 1;

                }} else if(activatedb[n] == 1 && (u < psb[2] || u > psb[3])) {{ // if vertex is activated
                    activatedb[n] = 0;

                }}
            }}

            if(t > 3000000){{

                if(input_verticesb[n] == 1){{

                    activatedb[n] = 0;
                    input_verticesb[n] = 0;
                }}
            }}

            /*
            if(n==0){{
                float sum = 0;
                for (int i = 0; i< 10178; i++){{
                    sum += activatedb[i];
                }}
                printf("%d ", sum);
            }}
            */

            all_activatedb[number_of_nodes*(t) + n] = activatedb[n];
        }}
            """.format(dtype)

    def convert_params(self, params):

        dt, n_timesteps, n_loops, ps, D, production_rate, boundary_cond, node_radius = params
        # create arrays
        if self.dtype == 'float64':
            dtype = 'double'
            D = np.float64(D)
            self.dx = np.float64(self.dx)
            production_rate = np.float64(production_rate)
            dt = np.float64(dt)
        elif self.dtype == 'float32':
            dtype = 'float'
            D = np.float32(D)
            self.dx = np.float32(self.dx)
            production_rate = np.float32(production_rate)
            dt = np.float32(dt)


        params = dt, n_timesteps, n_loops, ps, D, production_rate, boundary_cond, node_radius

        return params

    def create_arrays(self, u0, grid_corners, node_dim, input_indeces, output_indeces, params):
        dt, n_steps, n_loops, ps, D, production_rate, boundary_cond, node_radius = params
        # create context, queue and memory flags

        all_node_positions = self.get_node_positions(node_dim, grid_corners)
        input_node_positions = [all_node_positions[i] for i in input_indeces]

        output_node_positions = [all_node_positions[i] for i in output_indeces]

        output_vertices, one_hot_out = self.assign_vertices(self.vertex_positions, output_node_positions, node_radius)

        input_vertices, one_hot_in = self.assign_vertices(self.vertex_positions, input_node_positions, node_radius)
        barrier_vertices, one_hot_barrier = self.get_barrier_vertices(self.vertex_positions, grid_corners, node_dim, self.nx, self.ny)

        print('nx, ny', self.nx, self.ny)
        print('n_nodes: ', self.n_vertices)

        # create arrays
        if self.dtype == 'float64':
            all_u = np.zeros((n_steps, self.n_vertices), dtype = np.float64)
            Au = np.zeros(self.n_vertices, dtype = np.float64)
            current_u = np.array(u0, dtype = np.float64)

            #vertex_production_rates = np.float64(vertex_production_rates, dtype = np.float64)
            ps = np.array(ps, dtype = np.float64)
            node_times_on = np.zeros(self.n_vertices, dtype = np.float64)
            node_times_off = np.zeros(self.n_vertices, dtype = np.float64)

        elif self.dtype == 'float32':
            all_u = np.zeros((n_steps, self.n_vertices), dtype = np.float32)
            Au = np.zeros(self.n_vertices, dtype = np.float32)
            current_u = np.array(u0, dtype = np.float32)
            ps = np.array(ps, dtype = np.float32)
            #vertex_production_rates = np.float32(vertex_production_rates, dtype = np.float32)
            node_times_on = np.zeros(self.n_vertices, dtype = np.float32)
            node_times_off = np.zeros(self.n_vertices, dtype = np.float32)

        #activated = np.zeros((nodes,1), dtype = np.int32)
        # set activated to be the input nodes initially
        activated = np.array(one_hot_in, dtype = np.int32)
        all_activated = np.zeros(( n_steps, self.n_vertices), dtype = np.int32)
        boundary_nodes = np.array(self.get_boundary_nodes(), dtype = np.int32)
        self.n_boundary_vertices = np.int32(len(boundary_nodes))
        boundary_flags = np.zeros(self.n_vertices, dtype = np.int32)
        all_nodes = np.arange(self.nx*self.ny)

        non_boundary_nodes = np.array(np.delete(all_nodes, boundary_nodes), dtype = np.int32)

        boundary_flags[boundary_nodes] = 1
        current_u[boundary_nodes] = 0

        all_u[0,:] = current_u

        return [all_u, current_u, Au, ps, node_times_on, node_times_off, all_activated, activated, boundary_nodes, non_boundary_nodes, boundary_flags, one_hot_in, one_hot_out, one_hot_barrier]

    def create_programs(self, boundary_cond):
        Au_prg = cl.Program(self.ctx, self.get_Au_kernel).build()

        if boundary_cond == 'zero':
            boundary_prg = cl.Program(self.ctx, self.boundary_kernel_zero).build()
        elif boundary_cond == 'periodic':
            boundary_prg = cl.Program(self.ctx, self.boundary_kernel_periodic).build()
        elif boundary_cond == 'insulating':
            boundary_prg = cl.Program(self.ctx, self.boundary_kernel_insulating).build()
        else:
            raise Exception('invalid boundary conditions')


        add_next_u_prg = cl.Program(self.ctx, self.add_next_u_kernel).build()

        add_activated_prg = cl.Program(self.ctx, self.add_activated_kernel).build()

        return [Au_prg, boundary_prg, add_next_u_prg, add_activated_prg]

    def create_buffers(self, arrays):
        # Au buffers
        # create buffer


        all_u, current_u, Au, ps, node_times_on, node_times_off, all_activated, activated, boundary_nodes, non_boundary_nodes, boundary_flags, one_hot_in, one_hot_out, one_hot_barrier = arrays

        non_boundary_nodesb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=non_boundary_nodes)
        boundary_flagsb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=boundary_flags)
        input_nodesb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=one_hot_in)

        barrier_nodesb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=one_hot_barrier)
        boundary_nodesb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=boundary_nodes)

        all_activatedb = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=all_activated)

        Aub = cl.Buffer(self.ctx, self.mf.READ_WRITE, Au.nbytes)
        all_ub = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=all_u) # output buffer
        current_ub = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=current_u) # input buffer
        activatedb = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=activated)
        output_nodesb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=one_hot_out)


        psb = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=ps)

        node_times_onb = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=node_times_on)
        node_times_offb = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=node_times_off)

        return [all_ub, current_ub, Aub, psb, node_times_onb, node_times_offb, all_activatedb, activatedb, boundary_nodesb, non_boundary_nodesb, boundary_flagsb, input_nodesb, output_nodesb, barrier_nodesb]

    def run_sim(self, arrays, buffers, programs, n_loops, n_timesteps, params):

        Au_prg, boundary_prg, add_next_u_prg, add_activated_prg = programs
        all_u, current_u, Au, ps, node_times_on, node_times_off, all_activated, activated, boundary_nodes, non_boundary_nodes, boundary_flags, one_hot_in, one_hot_out, one_hot_barrier = arrays
        all_ub, current_ub, Aub, psb, node_times_onb, node_times_offb, all_activatedb, activatedb, boundary_nodesb, non_boundary_nodesb, boundary_flagsb, one_hot_inb, one_hot_outb, one_hot_barrierb = buffers
        dt, n_timesteps, n_loops, ps, D, production_rate, boundary_cond, node_radius = params

        overall_activated = []
        overall_Us = []
        for i in range(n_loops): # each loop allows copying from GPU to RAM
            print('loop:', i)
            for t in range(n_timesteps): # each loop runs in GPU memory
                if t %1000 == 0:
                    print(t)
                    print('--------------------------------------------------------------------')

                # get Au buffer for the time step
                Au_prg.get_y(self.queue, (self.n_vertices - self.n_boundary_vertices,),None, current_ub, Aub, non_boundary_nodesb,one_hot_barrierb, np.int32(self.nx), self.dx,  D)

                boundary_prg.boundary(self.queue, (self.n_boundary_vertices,), None, Aub, boundary_nodesb, current_ub, np.int32(self.n_vertices), np.int32(self.nx), self.dx, D)
                # calculate next u and add to the buffer of all us

                add_next_u_prg.add_next_u(self.queue, (self.n_vertices,), None, Aub, boundary_flagsb, dt, np.int32(self.n_vertices), np.int32(t), all_ub, current_ub, activatedb, node_times_onb, node_times_offb,  production_rate)
                #add_next_u_prg.add_next_u(queue, (self.n_nodes,), None, Aub, boundary_flagsb, dt, np.int32(self.n_nodes), np.int32(t), activatedb, production_rate)#run kernel
                add_activated_prg.add_activated(self.queue, (self.n_vertices,), None, current_ub, activatedb, all_activatedb, one_hot_inb, one_hot_outb, np.int32(self.n_vertices), np.int32(t), psb)
                #cl.enqueue_copy(queue, current_u, current_ub)
                #cl.enqueue_copy(queue, all_u, all_ub) # copy result from buffer
                #print('sum: ', sum(current_u))

            cl.enqueue_copy(self.queue, all_u, all_ub) # copy result from buffer

            all_u = all_u.reshape(n_timesteps,self.n_vertices)
            print('\n u', sum(all_u[0]))
            print(sum(all_u[-1]))
            cl.enqueue_copy(self.queue, all_activated, all_activatedb)
            all_activated = all_activated.reshape(n_timesteps,self.n_vertices)
            print('a' ,sum(all_activated[0]))
            print(sum(all_activated[-1]))

            #input_nodes = np.zeros(self.n_nodes)

            #input_nodes = np.array(input_nodes, dtype = np.int32)
            overall_Us.append(all_u[:n_timesteps])
            overall_activated.append(all_activated[:n_timesteps])
            all_u = np.zeros(( n_timesteps , self.n_vertices), dtype = np.float32)

            #all_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_u)
            all_activated = np.zeros(( n_timesteps, self.n_vertices), dtype = np.int32)

        #all_activatedb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_activated)
        #input_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_nodes)
        overall_activated = np.array(overall_activated).reshape(n_loops*(n_timesteps),self.nx*self.nx)
        overall_Us = np.array(overall_Us).reshape(n_loops*(n_timesteps),self.nx*self.nx)

        return overall_Us, overall_activated

    def synbio_brain(self, u0, grid_corners, node_dim, input_indeces, output_indeces, params):

        params = self.convert_params(params)
        dt, n_timesteps, n_loops, ps, D, production_rate, boundary_cond, node_radius = params

        arrays = self.create_arrays(u0, grid_corners, node_dim, input_indeces, output_indeces, params)# REFACTOR THIS INTO CLASS

        buffers = self.create_buffers(arrays)

        boundary_cond = params[6]
        programs = self.create_programs(boundary_cond)

        overall_Us, overall_activated = self.run_sim(arrays, buffers, programs, n_loops, n_timesteps, params)

        return overall_Us, overall_activated


    def synbio_brain_checkerboard(self, u0, input_nodes, output_nodes, barrier_nodes, params):
        # M doesnt depend on U so can calculate it once outside the loop
        dt, n_steps, n_loops, ps, D, production_rate, boundary_cond = params
        # create context, queue and memory flags
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags
        print(self.nx, self.ny)
        print('n_nodes: ', self.n_nodes)

        # create arrays
        if self.dtype == 'float64':
            dtype = 'double'

            all_u = np.zeros((n_steps, self.n_nodes), dtype = np.float64)

            dt = np.float64(dt)

            Au = np.zeros(self.n_nodes, dtype = np.float64)
            current_u = np.array(u0, dtype = np.float64)
            D = np.float64(D)
            #vertex_production_rates = np.float64(vertex_production_rates, dtype = np.float64)
            ps = np.array(ps, dtype = np.float64)
            production_rate = np.float64(production_rate)

            node_times_on = np.zeros(self.n_nodes, dtype = np.float64)
            node_times_off = np.zeros(self.n_nodes, dtype = np.float64)
            self.dx = np.float64(self.dx)

        elif self.dtype == 'float32':

            dtype = 'float'
            if checkerboard:
                all_u = np.zeros((2, n_steps, self.n_nodes), dtype = np.float32)
                checkerboard_mask1 = np.ones(len(input_nodes), dtype = np.int32)
                checkerboard_mask2 = np.ones(len(input_nodes), dtype = np.int32)

            else:
                all_u = np.zeros((n_steps, self.n_nodes), dtype = np.float32)
                checkerboard_mask = np.ones(len(input_nodes), dtype = np.int32)
            dt = np.float32(dt)

            Au = np.zeros(self.n_nodes, dtype = np.float32)
            current_u = np.array(u0, dtype = np.float32)

            ps = np.array(ps, dtype = np.float32)
            #vertex_production_rates = np.float32(vertex_production_rates, dtype = np.float32)
            D = np.float32(D)
            production_rate = np.float32(production_rate)
            self.dx = np.float32(self.dx)

            node_times_on = np.zeros(self.n_nodes, dtype = np.float32)
            node_times_off = np.zeros(self.n_nodes, dtype = np.float32)

        input_nodes = np.array(input_nodes, dtype = np.int32)
        output_nodes = np.array(output_nodes, dtype = np.int32)
        barrier_nodes = np.array(barrier_nodes, dtype = np.int32)
        #activated = np.zeros((nodes,1), dtype = np.int32)

        # set activated to be the input nodes initially
        activated = np.array(input_nodes, dtype = np.int32)
        all_activated = np.zeros(( n_steps, self.n_nodes), dtype = np.int32)

        boundary_nodes = np.array(self.get_boundary_nodes(), dtype = np.int32)
        print(boundary_nodes)
        n_boundary_nodes = np.int32(len(boundary_nodes))
        boundary_flags = np.zeros(self.n_nodes, dtype = np.int32)
        all_nodes = np.arange(self.nx*self.ny)

        non_boundary_nodes = np.array(np.delete(all_nodes, boundary_nodes), dtype = np.int32)

        boundary_flags[boundary_nodes] = 1
        current_u[boundary_nodes] = 0

        all_u[0,:] = current_u

        # Au buffers
        # create buffer
        non_boundary_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=non_boundary_nodes)
        boundary_flagsb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=boundary_flags)
        input_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_nodes)

        barrier_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=barrier_nodes)
        boundary_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=boundary_nodes)

        all_activatedb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_activated)

        if checkerboard:

            Au1b = cl.Buffer(ctx, mf.READ_WRITE, Au.nbytes)
            all_u1b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_u) # output buffer
            current_u1b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=current_u) # input buffer
            activated1b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=activated)
            output_nodes1b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=output_nodes)

            Au2b = cl.Buffer(ctx, mf.READ_WRITE, Au.nbytes)
            all_u2b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_u) # output buffer
            current_u2b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=current_u) # input buffer
            activated2b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=activated)
            output_nodes2b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=output_nodes)
        else:
            Aub = cl.Buffer(ctx, mf.READ_WRITE, Au.nbytes)
            all_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_u) # output buffer
            current_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=current_u) # input buffer
            activatedb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=activated)
            output_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=output_nodes)


        psb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ps)

        if checkerboard:
            checkerboard_mask1b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=checkerboard_mask1)
            checkerboard_mask2b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=checkerboard_mask2)
        else:
            checkerboard_maskb = checkerboard_mask1b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=checkerboard_mask)
        #vertex_production_ratesb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vertex_production_rates)

        node_times_onb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=node_times_on)
        node_times_offb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=node_times_off)
        Au_prg = cl.Program(ctx, get_Au_kernel).build()

        if boundary_cond == 'zero':
            boundary_prg = cl.Program(ctx, boundary_kernel_zero).build()
        elif boundary_cond == 'periodic':
            boundary_prg = cl.Program(ctx, boundary_kernel_periodic).build()
        elif boundary_cond == 'insulating':
            boundary_prg = cl.Program(ctx, boundary_kernel_insulating).build()
            print(boundary_cond)
        else:
            raise Exception('invalid boundary conditions')


        add_next_u_prg = cl.Program(ctx, add_next_u_kernel).build()

        add_activated_prg = cl.Program(ctx, add_activated_kernel).build()
        print(Au.shape)

        overall_activated = []

        for i in range(n_loops):
            print('loop:', i)
            for t in range(n_steps):
                if t %1000 == 0:
                    print(t)
                    print('--------------------------------------------------------------------')


                if checkerboard:
                    Au_prg.get_y(queue, (self.n_nodes - n_boundary_nodes,),None, current_u1b, Au1b, non_boundary_nodesb,barrier_nodesb, np.int32(self.nx), self.dx,  D)
                    Au_prg.get_y(queue, (self.n_nodes - n_boundary_nodes,),None, current_u2b, Au2b, non_boundary_nodesb,barrier_nodesb, np.int32(self.nx), self.dx,  D)

                    boundary_prg.boundary(queue, (len(boundary_nodes),), None, Au1b, boundary_nodesb, np.int32(self.n_nodes), np.int32(self.nx))
                    boundary_prg.boundary(queue, (len(boundary_nodes),), None, Au2b, boundary_nodesb, np.int32(self.n_nodes), np.int32(self.nx))

                    add_next_u_prg.add_next_u(queue, (self.n_nodes,), None, Au1b, boundary_flagsb, dt, np.int32(self.n_nodes), np.int32(t), all_u1b, current_u1b, activated1b, node_times_onb, node_times_offb, production_rate)
                    add_next_u_prg.add_next_u(queue, (self.n_nodes,), None, Au2b, boundary_flagsb, dt, np.int32(self.n_nodes), np.int32(t), all_u2b, current_u2b, activated2b, node_times_onb, node_times_offb, production_rate)

                    add_activated_prg.add_activated(queue, (self.n_nodes,), None, current_u1b, activated1b, all_activatedb, input_nodesb, output_nodesb, np.int32(self.n_nodes), np.int32(t), psb)
                    add_activated_prg.add_activated(queue, (self.n_nodes,), None, current_u2b, activated2b, all_activatedb, input_nodesb, output_nodesb, np.int32(self.n_nodes), np.int32(t), psb)
                else:

                    # get Au buffer for the time step
                    Au_prg.get_y(queue, (self.n_nodes - n_boundary_nodes,),None, current_ub, Aub, non_boundary_nodesb,barrier_nodesb, np.int32(self.nx), self.dx,  D)

                    boundary_prg.boundary(queue, (len(boundary_nodes),), None, Aub, boundary_nodesb, current_ub, np.int32(self.n_nodes), np.int32(self.nx), self.dx, D)
                    # calculate next u and add to the buffer of all us

                    add_next_u_prg.add_next_u(queue, (self.n_nodes,), None, Aub, boundary_flagsb, dt, np.int32(self.n_nodes), np.int32(t), all_ub, current_ub, activatedb, checkerboard_maskb, node_times_onb, node_times_offb,  production_rate)
                    #add_next_u_prg.add_next_u(queue, (self.n_nodes,), None, Aub, boundary_flagsb, dt, np.int32(self.n_nodes), np.int32(t), activatedb, production_rate)#run kernel
                    add_activated_prg.add_activated(queue, (self.n_nodes,), None, current_ub, activatedb, all_activatedb, input_nodesb, output_nodesb, np.int32(self.n_nodes), np.int32(t), psb)
                    #cl.enqueue_copy(queue, current_u, current_ub)
                    #cl.enqueue_copy(queue, all_u, all_ub) # copy result from buffer
                    #print('sum: ', sum(current_u))

            cl.enqueue_copy(queue, all_u, all_ub) # copy result from buffer

            all_u = all_u.reshape(n_steps,self.n_nodes)
            print('\n u', sum(all_u[0]))
            print(sum(all_u[-1]))
            cl.enqueue_copy(queue, all_activated, all_activatedb)
            all_activated = all_activated.reshape(n_steps,self.n_nodes)
            print('a' ,sum(all_activated[0]))
            print(sum(all_activated[-1]))

            #input_nodes = np.zeros(self.n_nodes)

            #input_nodes = np.array(input_nodes, dtype = np.int32)

            overall_Us.append(all_u[:n_steps])
            overall_activated.append(all_activated[:n_steps])
            all_u = np.zeros(( n_steps , self.n_nodes), dtype = np.float32)

            #all_ub = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_u)
            all_activated = np.zeros(( n_steps, self.n_nodes), dtype = np.int32)

            #all_activatedb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=all_activated)
            #input_nodesb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_nodes)

        if checkerboard:
            return np.array(overall_U1s).reshape(n_loops*(n_steps),self.nx*self.nx), np.array(overall_U2s).reshape(n_loops*(n_steps),self.nx*self.nx), np.array(overall_activated).reshape(n_loops*(n_steps),self.nx*self.nx)
        else:

            return np.array(overall_Us).reshape(n_loops*(n_steps),self.nx*self.nx), np.array(overall_activated).reshape(n_loops*(n_steps),self.nx*self.nx)




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


def count_cohesive_nodes_FD(activated_ts, vertex_positions, node_dim, node_radius, grid_corners):
    # get the vertices which we need to check for activation
    all_node_positions = get_node_positions(node_dim, grid_corners)
    output_vertices, one_hot_out = assign_vertices(vertex_positions, all_node_positions, node_radius)

    cohesive_ts = []
    print(np.sum(one_hot_out))

    for i in range(0, len(activated_ts), 1):

        # count the percentage of vertices within each node that are activated
        activated = activated_ts[i]
        if i%1000 == 0: print(i)
        n_cohesive = 0
        for node in range(node_dim[0] * node_dim[1]):
            #print('shae:', np.array(all_node_positions[node:node+1]).shape)
            vertices, one_hot_vertices = assign_vertices(vertex_positions, [all_node_positions[node]], node_radius)
            one_hot_vertices = np.array(one_hot_vertices, dtype = np.int32)
            #print('one_hot_out:', output_vertices)
            #print(np.where(activated == 1)[0][-200:])
            #print(np.where(one_hot_vertices == 1))
            #print(one_hot_vertices.shape)
            percentage_activated = np.sum(activated*one_hot_vertices) / np.sum(one_hot_vertices)
            #print(percentage_activated)

            if percentage_activated == 0 or percentage_activated == 1:
                n_cohesive += 1
        cohesive_ts.append(n_cohesive)

    return cohesive_ts
