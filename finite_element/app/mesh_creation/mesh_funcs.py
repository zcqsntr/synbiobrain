from diffusion_sim import *


def create_node_fields(mesh_file, node_dim, node_radius, grid_corners, VIn, VOut, new_mesh_name = False):
    centres = get_node_positions(node_dim, grid_corners)

    mesh = open(mesh_file, 'a')

    for i, centre in enumerate(centres):
        print(i)
        mesh.write("Field[{0}] = Ball;\n"\
            "Field[{0}].Radius = {1};\n"\
            "Field[{0}].XCenter = {2};\n"\
            "Field[{0}].YCenter = {3};\n"\
            "Field[{0}].ZCenter = 0;\n"\
            "Field[{0}].VIn = {4};\n"\
            "Field[{0}].VOut = {5};\n"\
            .format(i, node_radius, centre[0], centre[1], VIn, VOut))

    field_list = ', '.join(map(str, range(i+1)))

    mesh.write("Field[{0}] = Min;\n"\
            "Field[{0}].FieldsList = {{{1}}};\n"\
            "Background Field = {0};\n"
            .format(i+1, field_list))

    mesh.close()


def create_circles(mesh_file, node_dim, node_radius, grid_corners, new_mesh_name = False):
    centres = get_node_positions(node_dim, grid_corners)


    with open(new_mesh_name, 'w') as mesh, open(mesh_file, 'r') as orig:


        for line in orig:
            mesh.write(line)


        for i, centre in enumerate(centres):

            #make points that define the circle
            top = np.append(centre + np.array([node_radius, 0]), [0, 1.0]) #top
            right = np.append(centre + np.array([0, node_radius]), [0, 1.0]) #right
            bottom = np.append(centre + np.array([-node_radius, 0]), [0, 1.0]) #bottom
            left = np.append(centre + np.array([0, -node_radius]), [0, 1.0]) #left
            centre = np.append(centre, [0, 1.0])

            points = [top, right, bottom, left, centre]

            for j in range(len(points)):
                points[j] = ', '.join(map(str, points[j]))


            #write points and circles to file, 5 points per circle plus 4 points for square corners
            mesh.write("Point({0}) = {{{1}}};\n"\
                    "Point({2}) = {{{3}}};\n"\
                    "Point({4}) = {{{5}}};\n"\
                    "Point({6}) = {{{7}}};\n"\
                    "Point({8}) = {{{9}}};\n"
                    .format(5*i + 5, points[0], 5*i + 6, points[1], 5*i + 7, points[2], 5*i + 8, points[3], 5*i + 9, points[4]))

            #write circle arcs 4 arcs per circle, four lines in square
            mesh.write("Circle({0}) = {{{1}, {2}, {3}}};\n"\
                    "Circle({4}) = {{{3}, {2}, {5}}};\n"\
                    "Circle({6}) = {{{5}, {2}, {7}}};\n"\
                    "Circle({8}) = {{{7}, {2}, {1}}};\n"
                    .format(4*i + 5, 5*i + 5, 5*i + 9,  5*i + 6, 4*i + 6, 5*i + 7, 4*i + 7, 5*i + 8, 4*i + 8))


def create_circles_2(mesh_file, node_dim, node_radius, grid_corners, new_mesh_name = False):
    centres = get_node_positions(node_dim, grid_corners)


    with open(new_mesh_name, 'w') as mesh, open(mesh_file, 'r') as orig:


        for line in orig:
            mesh.write(line)


        for i, centre in enumerate(centres):
            print(node_radius)
            centre = ', '.join(map(str, centre)) + ', 0, {0}, 0, 2*Pi'.format(node_radius)
            mesh.write("Circle({0}) = {{{1}}}; \n".format(i+5, centre))


def add_surfaces(mesh_file, node_dim, node_radius, grid_corners, new_mesh_name = False):

        with open(new_mesh_name, 'w') as mesh, open(mesh_file, 'r') as orig:


            for line in orig:
                mesh.write(line)



def clear_fields(mesh_file):

    mesh = open(mesh_file, "r")
    lines = mesh.readlines()
    mesh.close()

    mesh = open(mesh_file, "w")

    for line in lines:

        if line[0:5] == "Field":
            break
        mesh.write(line)

    mesh.close()
