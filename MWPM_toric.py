import os
import sys
import getopt
import time
import subprocess
import argparse

import numpy as np

from src.toric_model import *
from src.util import Action


def usage():
    print('usage:  sdafklasfjlksdaklfjsa')

# non periodic manhattan distance between qubit at coord1 and coord2
def distance(coord1, coord2):
    x_distance = np.abs(coord1[0] - coord2[0])
    y_distance = np.abs(coord1[1] - coord2[1])
    return x_distance + y_distance

# non periodic one dimensional distance between rows or columns at coord1 and coord2
def get_non_periodic_distance(coord1, coord2):
    distance = np.abs(coord1 - coord2)
    return distance

# periodic one dimensional distance between rows or columns at coord1 and coord2
def get_periodic_distance_for_one_axis(coord1, coord2, system_size):
    distance_to_border_coord1 = system_size-1-coord2
    distance_to_border_coord2 = system_size-1-coord1
    distance1 = coord1 + 1 + distance_to_border_coord1
    distance2 = coord2 + 1 + distance_to_border_coord2
    # the result is the same as writing system_size - get_non_periodic_distance
    return min([distance1, distance2])

# shortest manhattan distance among all ways around the torus
def get_shortest_distance(coord1, coord2, system_size):
    x_distance = min([get_periodic_distance_for_one_axis(coord1[0], coord2[0], system_size), get_non_periodic_distance(coord1[0], coord2[0])])
    y_distance = min([get_periodic_distance_for_one_axis(coord1[1], coord2[1], system_size), get_non_periodic_distance(coord1[1], coord2[1])])
    distance = x_distance + y_distance
    return distance

# generates an array of distinct node pairs and stores in edges[]
def generate_node_indices(edges, nbr_of_edges, nbr_of_nodes):
    # list of single-valued arrays of decreasing length and increasing value
    start_nodes = [np.repeat(i, nbr_of_nodes-i-1) for i in range(nbr_of_nodes)]
    # combine the list of arrays into a 1d array
    edges[:,0] = np.concatenate(start_nodes)

    # list of integer sequence arrays of decreasing length. Each sequence ends at nbr_of_nodes-1
    end_nodes = [np.arange(i+1, nbr_of_nodes) for i in range(nbr_of_nodes)]
    # combine the list of arrays into a 1d array
    edges[:,1] = np.concatenate(end_nodes)

    return edges


def get_distances(MWPM_edges, edges_no_periodic, edges, nbr_of_nodes):
    # Special case with only 2 nodes
    if (nbr_of_nodes == 2):
        MWPM_edge_indices = 0
        periodic_distances = edges[0][2]
        non_periodic_distances = edges_no_periodic[0][2]
    else:
        # wtf. I think this calculates the (edge?) indices there edges_no_periodic is equal to MWPM_edges
        MWPM_edge_indices = np.where((edges_no_periodic[:, 0:2] == MWPM_edges[:, None]).all(-1))[1]
        # Uses indices to get periodic and non periodic distances
        periodic_distances = edges[MWPM_edge_indices, 2]
        non_periodic_distances = edges_no_periodic[MWPM_edge_indices, 2]

    return non_periodic_distances, periodic_distances

# Generates error chain given probability
def generate_syndrome(toric_code, p):
    toric_code.generate_random_error(p)
    return toric_code

# Generates error chain given number of errors
def generate_syndrome_smart(toric_code, n):
    toric_code.generate_n_random_errors(n)
    return toric_code


def generate_MWPM(matrix, system_size):
    # array of coordniates of defects on the syndrom matrix
    defect_coords = np.array(np.nonzero(matrix)).T
    
    # number of defects
    nbr_of_nodes = len(defect_coords)

    # each node is paired with each other node, so there are n(n-1)/2 edges
    nbr_of_edges= int(nbr_of_nodes*(nbr_of_nodes-1)/2)

    # ndarrays indexing the edges. zeroth/first column is index of start/end nodes
    # second column is distance between nodes, i.e. weight of edge
    edges = np.zeros((nbr_of_edges, 3))
    edges_no_periodic = np.zeros((nbr_of_edges, 3))

    # generates node indices
    edges = generate_node_indices(edges, nbr_of_edges, nbr_of_nodes)
    edges_no_periodic = generate_node_indices(edges_no_periodic, nbr_of_edges, nbr_of_nodes)

    # 2d list of shortest distances between each pair of nodes
    shortest_distances = [get_shortest_distance(coord1, coord2, system_size)
            for i, coord1 in enumerate(defect_coords[:-1]) for coord2 in defect_coords[i+1:, :]]

    # 2d list of non periodic shortest distances between each pair of nodes
    shortest_non_periodic_distances= [distance(coord1, coord2)
            for i, coord1 in enumerate(defect_coords[:-1]) for coord2 in defect_coords[i+1:, :]]

    # store 2d lists in edge lists
    edges[:, 2] = shortest_distances
    edges_no_periodic[:, 2] = shortest_non_periodic_distances

    # builds arguments for the bloosom5 program
    processId = os.getpid()
    PATH = str(processId) + 'edges.TXT'
    OUTPUT_PATH = str(processId) +'output.TXT'

    # Save txt file with data for blossom5 to read
    header_str = "{} {}".format(nbr_of_nodes, nbr_of_edges)
    np.savetxt(PATH, edges, fmt='%i', header=header_str, comments='')

    # Run the blossom5 program as if from the terminal. The devnull part discards any prints from blossom5
    blossomname = './src/blossom5-v2.05.src/blossom5'
    if os.name == 'windows':
        blossomname += '.exe'
    subprocess.call([blossomname, '-e', PATH, '-w', OUTPUT_PATH, '-V'], stdout=open(os.devnull, 'wb'))

    # Read the output from blossom5 and delete the output file
    MWPM_edges = np.loadtxt(OUTPUT_PATH, skiprows=1, dtype=int)
    MWPM_edges = MWPM_edges.reshape((int(nbr_of_nodes/2), 2))

    # remove files generated by blossom5
    os.remove(PATH)
    os.remove(OUTPUT_PATH)

    return MWPM_edges, edges, edges_no_periodic, defect_coords

def eliminate_defect_pair(toric_code, start_coord, end_coord, diff_coord, matrix_index, system_size):
    coord = start_coord
    # calculates manhattan distance between defects other way around torus 
    size_diff = (np.array([system_size, system_size])-np.abs(diff_coord))
    # NOTERA! INNAN STOD DET "5" ISTALLET FOR "system_size" NEDAN
    size_diff[size_diff == system_size] = 0

    # calculates shortest vertical and horizontal distance between defects
    nbr_of_vertical_steps = min(size_diff[0], np.abs(diff_coord[0]))
    nbr_of_horizontal_steps = min(size_diff[1], np.abs(diff_coord[1]))

    # matrix_index is the layer of the defect pair
    # this translates the layer into x or z operators
    action_index = (not matrix_index)*2 + 1

    # iterate through vertical steps
    for i in range(nbr_of_vertical_steps):
        # choose direction around torus
        direction = size_diff[0]>np.abs(diff_coord[0])
        # calculate qubit coordinates to apply correction
        coord[0] = (coord[0] + direction-(not matrix_index))%system_size
        # store qubit coordinates
        pos = [matrix_index, coord[0], coord[1]]
        # update coordinate of defect
        coord[0] = (coord[0] - (not direction)+(not matrix_index))%system_size

        # apply correction on torus
        action = Action(pos, action_index)
        toric_code.step(action)
        toric_code.syndrom('state')

    # iterate through horizontal steps
    for i in range(nbr_of_horizontal_steps):
        # calculate coordinates of qubit to apply correction
        # and update coordinate of defect
        if (diff_coord[1] < 0):
            direction = size_diff[1]<np.abs(diff_coord[1])
            coord[1] = (coord[1] + (direction)-(not matrix_index))%system_size
            pos = [int(not matrix_index), coord[0], coord[1]]
            coord[1] = (coord[1] - (not direction))+(not matrix_index)%system_size
        else:
            direction = size_diff[1]>np.abs(diff_coord[1])
            coord[1] = (coord[1] + (direction)-(not matrix_index))%system_size
            pos = [int(not matrix_index), coord[0], coord[1]]
            coord[1] = (coord[1] - (not direction)+(not matrix_index))%system_size

        # apply correction on torus
        action = Action(pos, action_index)
        toric_code.step(action)
        toric_code.syndrom('state')
    
    # return corrected torus
    return toric_code

# given torus and defect pairs, this applies a correction connecting the pairs
def generate_solution(MWPM_edges, defect_coords, toric_code, matrix_index, system_size):
    # coordinates of pairs to connect
    start_coords = defect_coords[MWPM_edges[:, 0].T, :]
    end_coords = defect_coords[MWPM_edges[:, 1].T, :]
    # distances of pairs to connect
    diff_coords = end_coords-start_coords

    # if there are defects on the torus
    if (np.sum(np.sum(toric_code.current_state[matrix_index])) > 0):
        # iterate through all the mwpm defect pairs
        for start_coord, end_coord, diff_coord in zip(start_coords, end_coords, diff_coords):
            # eliminate the current pair
            toric_code = eliminate_defect_pair(toric_code, start_coord, end_coord, diff_coord, matrix_index, system_size)
    return toric_code

def main(args):
    #TODO: add support for arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('test2', help='test adsfadd')
    # parser.parse_args()
    # print(args.test2)
    # try:
    #     opts, args = getopt.getopt(argv, "hn:d:p:", ["help"])
    # except getopt.GetoptError as err:
    #     print(err)
    #     usage()
    #     sys.exit(2)
    # print(opts)
    # print(args)

    # for o, a in opts:
    #     print(o)
    #     if o in ("-h", "--help"):
    #         usage()
    #     elif o == "-d":
    #         system_size = int(a)
    #         print("d = {}".format(system_size))
    #     elif o == "-n":
    #         nbr_of_iterations = int(float(a))
    #         print(nbr_of_iterations)


    p_errors = [0.07]
    system_size = int(5)
    nbr_of_iterations = int(1e2)

    print(p_errors)
    ground_state_kept_list = []

    # iterate through p_errors
    for p in p_errors:
        print(p)
        ground_states = 0
        # iterate some number of times
        for _ in range(nbr_of_iterations):
            # create a torus and generate error according to p
            toric_code = Toric_code(system_size)
            nbr_of_vertex_nodes = 0
            nbr_of_plaquette_nodes = 0

            toric_code = generate_syndrome(toric_code, p)
            n = 3

            MWPM_edges_vertex = []
            edges_vertex = []
            edges_no_periodic_vertex = []
            defect_coords_vertex = []

            MWPM_edges_plaquette = []
            edges_plaquette = []
            edges_no_periodic_plaquette = []
            defect_coords_plaquette = []

            # if there are defects in a layer, connect pairs of defects with mwpm
            if np.sum(np.sum(toric_code.current_state[0])) > 0:
                MWPM_edges_vertex, edges_vertex, edges_no_periodic_vertex, defect_coords_vertex = generate_MWPM(toric_code.current_state[0], system_size)
            if np.sum(np.sum(toric_code.current_state[1])) > 0:
                MWPM_edges_plaquette, edges_plaquette, edges_no_periodic_plaquette, defect_coords_plaquette = generate_MWPM(toric_code.current_state[1], system_size)

            # connect defect pairs given by mwpm
            if len(MWPM_edges_vertex) > 0:
                toric_code = generate_solution(MWPM_edges_vertex, defect_coords_vertex, toric_code, 0, system_size)
            if len(MWPM_edges_plaquette) > 0:
                toric_code = generate_solution(MWPM_edges_plaquette, defect_coords_plaquette, toric_code, 1, system_size)

            # check for logical errors
            toric_code.eval_ground_state()
            ground_states += toric_code.ground_state

        ground_state_kept_list.append(ground_states/nbr_of_iterations)

    # store results
    #timestamp = time.ctime()
    try:
        os.mkdir('results')
    except FileExistsError:
        pass

    PATH_ground2 = 'results/p_succes_MWPM_d={}.TXT'.format(system_size)

    np.savetxt(PATH_ground2, ground_state_kept_list, fmt='%e', comments='')
    print(ground_state_kept_list)

if __name__ == "__main__":
    main(sys.argv[1:])
