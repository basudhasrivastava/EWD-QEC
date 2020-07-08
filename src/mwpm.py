import os
import sys
import getopt
import time
import subprocess
import argparse
import copy

import numpy as np
import random as rand

from src.toric_model import *
from src.planar_model import *
from src.util import Action


class MWPM():
    def __init__(self, code):
        assert type(code) in (Toric_code, Planar_code), 'code has to be either Planar_code or Toric_code'
        self.code = code
        self.is_planar = (type(code) == Planar_code)
        self.solution = np.zeros_like(code.qubit_matrix)

    # calculates shortest distance between two defects
    def get_shortest_distance(self, defect1, defect2):
        # Planar_code only cares about direct path
        if self.is_planar:
            return manhattan_path(defect1, defect2).sum()
        # Toric_code has to take periodic path into account
        else:
            # Direct path
            non_periodic_path = manhattan_path(defect1, defect2)
            # Periodic path
            periodic_path = self.code.system_size - non_periodic_path
            # Minimum distance in each direction
            shortest_path = np.min(non_periodic_path, periodic_path, axis=1)
            # Total minimum distance
            return shortest_path.sum()


    def generate_random_pairing(self, layer):
        # generates node indices
        edges, _ = self.generate_edges(layer)
        chosen_edges = np.empty(shape=[0,3])

        #removes all connections between ancilla bits
        edges = edges[~np.all(edges[:,0:2] >= (np.amax(edges[:,0:2])+1)/2, axis=1)]

        #select random edges and remove the rest
        while edges.shape[0] > 0:
            rownum = rand.randint(0,edges.shape[0]-1)
            row = edges[rownum,:]
            chosen_edges = np.concatenate((chosen_edges, [row]), axis = 0)
            r0 = row[0]
            r1 = row[1]
            edges = edges[~np.any(edges[:,0:2] == r0, axis=1)]
            edges = edges[~np.any(edges[:,0:2] == r1, axis=1)]

        return chosen_edges.astype(int)



    #KLAR
    def get_layer(self, layer):
        # get defects in layer
        # planar code has different names for different layers
        if self.is_planar:
            if layer == 0:
                defects = self.code.vertex_defects
            elif layer == 1:
                defects = self.code.plaquette_defects
        else:
            defects = self.code.current_state[layer]

        return defects

    # generates an array of distinct node pairs and stores in edges[]
    def generate_edges(self, layer):
        # get defects in layer
        defect_coords = np.array(np.nonzero(self.get_layer(layer))).T

        # number of defects
        nbr_defects = defect_coords.shape[0]

        # planar has different names for different layers
        if self.is_planar:
            # twice as many nodes as defects since every defect gets an ancillary node
            nbr_nodes = 2 * nbr_defects

        else:
            # for toric the number of nodes is the same as the number of defects
            nbr_nodes = nbr_defects

        nbr_edges = int(nbr_defects * (nbr_defects - 1) / 2)

        # list of single-valued arrays of decreasing length and increasing value
        start_nodes, end_nodes = connect_all(nbr_defects)

        distances = [self.get_shortest_distance(defect_coords[start_nodes[edge]], defect_coords[end_nodes[edge]])
                for edge in range(nbr_edges)]

        if self.is_planar:
            # Generate and interconnect nodes for virtual defects
            nbr_ancilla_nodes = nbr_defects
            nbr_ancilla_edges = (nbr_ancilla_nodes * (nbr_ancilla_nodes - 1)) // 2
            ancilla_start, ancilla_end = connect_all(nbr_ancilla_nodes, nbr_defects)

            # put weight 0 on edges between ancilla defects
            ancilla_distances = [0] * nbr_ancilla_edges

            # connect every real defect to an ancillary defect
            border_start = [0] * nbr_defects
            border_end = [0] * nbr_defects
            border_distances = [0] * nbr_defects
            for start in range(nbr_defects):
                border_start[start] = start
                border_end[start] = start + nbr_defects
                # calculate shortest distance to any border
                distance = min(self.code.system_size - defect_coords[start, layer] - 1, defect_coords[start, layer] + 1)
                border_distances[start] = distance

            # append lists of start nodes, end nodes and distances
            start_nodes += ancilla_start     + border_start
            end_nodes   += ancilla_end       + border_end
            distances   += ancilla_distances + border_distances

            # correct edge count
            nbr_edges += nbr_defects + nbr_ancilla_edges

        # store the lists of node connections and distances in array
        edges = np.zeros((nbr_edges, 3))
        edges[:, 0] = np.array(start_nodes)
        edges[:, 1] = np.array(end_nodes)
        edges[:, 2] = np.array(distances)

        return edges, nbr_nodes


    # generates edges so that odd or even parity results in different equivalence classes
    def generate_edges_constrained(self, layer, parity):
        # parity == 1 adds an extra ancilla node on both sides

        # array of coordinates of defects on the defect matrix
        if layer == 0:
            defect_coords = np.array(np.nonzero(self.code.vertex_defects)).T
        elif layer == 1:
            defect_coords = np.array(np.nonzero(self.code.plaquette_defects)).T

        # number of defects
        nbr_defects = defect_coords.shape[0]
        # every defects gets an ancillary node
        nbr_nodes = 2 * nbr_defects
        # initially, number of edges is edges between real defects
        nbr_edges = int(nbr_defects * (nbr_defects - 1) / 2)

        # list of single-valued arrays of decreasing length and increasing value
        start_nodes, end_nodes = connect_all(nbr_defects)

        # list of distances between real defects
        distances = [self.get_shortest_distance(defect_coords[start_nodes[edge]], defect_coords[end_nodes[edge]])
                for edge in range(nbr_edges)]

        # left/top or right/bottom border closest to each real defect
        nearest_border = ((defect_coords[:, layer] + 1) * 2 > self.code.system_size).astype(int)

        # number of ancilla nodes on left/top and right/bottom side
        nbr_ancilla_nodes_0 = np.bincount(nearest_border)[0]
        nbr_ancilla_nodes_1 = nbr_defects - nbr_ancilla_nodes_0

        # if parity is 1, add nodes on both sides of plane
        if parity == 1:
            if nbr_ancilla_nodes_0 > 0:
                nbr_ancilla_nodes_0 += 1
                nbr_nodes += 1
            if nbr_ancilla_nodes_1 > 0:
                nbr_ancilla_nodes_1 += 1
                nbr_nodes += 1

        # count number of edges between ancilla nodes on both sides
        nbr_ancilla_edges_0 = (nbr_ancilla_nodes_0 * (nbr_ancilla_nodes_0 - 1)) // 2
        nbr_ancilla_edges_1 = (nbr_ancilla_nodes_1 * (nbr_ancilla_nodes_1 - 1)) // 2

        # Connect all ancilla defects on the top/left
        ancilla_0_start, ancilla_0_end = connect_all(nbr_ancilla_nodes_0, nbr_defects)
        # put weight 0 on edges between ancilla defects
        ancilla_0_distances = [0] * nbr_ancilla_edges_0

        # Connect all ancilla defects on the bottom/right
        ancilla_1_start, ancilla_1_end = connect_all(nbr_ancilla_nodes_1, nbr_defects + nbr_ancilla_nodes_0)
        # put weight 0 on edges between ancilla defects
        ancilla_1_distances = [0] * nbr_ancilla_edges_1

        # connect all real defects with their corresponding ancilla defects
        border_start     = [0] * nbr_defects
        border_end       = [0] * nbr_defects
        border_distances = [0] * nbr_defects
        ancilla_counts = [0, 0]
        for start, border in enumerate(nearest_border):
            # border edges start with a real defects
            border_start[start] = start
            # connects every real defect with an ancilla defect on the closest border
            border_end[start] = nbr_defects + border * nbr_ancilla_nodes_0 + ancilla_counts[border]
            # weight of border edges is distance from defect to closest border
            distance = self.code.system_size - defect_coords[start, layer] - 1 if border else defect_coords[start, layer] + 1
            border_distances[start] = distance
            ancilla_counts[border] += 1

        # append the lists of nodes with all the ancilla connections
        start_nodes += ancilla_0_start     + ancilla_1_start     + border_start
        end_nodes   += ancilla_0_end       + ancilla_1_end       + border_end
        distances   += ancilla_0_distances + ancilla_1_distances + border_distances
        # the number of edges is increased with all edges connecting to ancilla defects
        nbr_edges += nbr_defects + nbr_ancilla_edges_0 + nbr_ancilla_edges_1

        # store the lists of node connections and distances in array
        edges = np.zeros((nbr_edges, 3))
        edges[:, 0] = np.array(start_nodes)
        edges[:, 1] = np.array(end_nodes)
        edges[:, 2] = np.array(distances)

        return edges, nbr_nodes


    # takes coordinates of two defects and connects them along a minimum path
    def eliminate_defect_pair(self, start_coord, end_coord, layer):
        diff_coord = end_coord - start_coord
        system_size = self.code.system_size

        # if planar, only manhattan path is interesting
        if self.is_planar:
            nbr_vertical_steps = np.abs(diff_coord[0])
            nbr_horizontal_steps = np.abs(diff_coord[1])

        # if torus, both paths around torus are interesting
        else:
            # calculates manhattan distance between defects other way around torus
            size_diff = (system_size - np.abs(diff_coord)) % system_size

            # calculates shortest vertical and horizontal distance between defects
            nbr_vertical_steps = min(size_diff[0], np.abs(diff_coord[0]))
            nbr_horizontal_steps = min(size_diff[1], np.abs(diff_coord[1]))

        # translates the layer into x or z operators
        operator = (not layer) * 2 + 1

        correction = np.zeros_like(self.solution)

        top, bot = sorted([start_coord[0], end_coord[0]])
        left, right = sorted([start_coord[1], end_coord[1]])

        if self.is_planar:
            # vertical. '+ not layer' offsets z qubit vertical position in relation to z defect
            vert = [i + (not layer) for i in range(top, bot)]
            correction[layer, vert, start_coord[1]] = operator
            # horizontal. '+ layer' offsets x qubit horizontal position in relation to x defect
            horiz = [i + layer for i in range(left, right)]
            correction[int(not layer), end_coord[0], horiz] = operator

        else:
            # for toric the offsets are different and are only needed for x qubits
            # vertical. check if periodic distance is shorter
            if (bot - top) * 2 > system_size:
                # list of qubits connecting defects to periodic borders
                vert = [i for i in range(0, top + layer)]
                vert += [i for i in range(bot + layer, system_size)]
            else:
                # list of qubit directly connecting defect pair
                vert = [i + layer for i in range(top, bot)]
            correction[layer, vert, start_coord[1]] = operator

            # horizontal. check if periodic distance is shorter
            if (right - left) * 2 > system_size:
                # list of qubits connecting defects to periodic borders
                horiz = [i + layer for i in range(0, left)]
                horiz += [i + layer for i in range(right, system_size)]
            else:
                # list of qubit directly connecting defect pair
                horiz = [i + layer for i in range(left, right)]
            correction[int(not layer), end_coord[0], horiz] = operator

        # Apply correction
        self.solution ^= correction


    # connects a defect to its' closest border
    def eliminate_border_defect(self, coord, layer):
        # translates the layer into x or z operators
        operator = (not layer) * 2 + 1

        correction = np.zeros_like(self.solution)

        # layer = 0 -> Z defects -> connect verticaly
        if layer == 0:
            # find closest border
            if (coord[0] + 1) * 2 < self.code.system_size:
                correction[0, :coord[0] + 1, coord[1]] = operator
            else:
                correction[0, coord[0] + 1:, coord[1]] = operator

        # layer = 1 -> X defects -> connect horizontally
        else:
            # find closest border
            if (coord[1] + 1) * 2 < self.code.system_size:
                correction[0, coord[0], :coord[1] + 1] = operator
            else:
                correction[0, coord[0], coord[1] + 1:] = operator

        self.solution ^= correction


    # generates graph of defects in layer, runs blossom5 and generates a correction chain
    def generate_solution(self, layer, parity=None, random_pairing = False):
        assert (parity in (None, 0, 1)), 'parity has to be None, 0 or 1'

        if parity is None:
            # generates edges optimally
            edges, nbr_nodes = self.generate_edges(layer)
        else:
            # generates edges in a
            edges, nbr_nodes = self.generate_edges_constrained(layer, parity)

        if random_pairing == False: MWPM_edges = self.generate_MWPM(layer, edges, nbr_nodes)
        else: MWPM_edges = self.generate_random_pairing(layer)

        # get defects in layer
        defects = self.get_layer(layer)

        # array of coordinates of defects on the syndrom matrix
        defect_coords = np.array(np.nonzero(defects)).T



        if self.is_planar:
            # number of defects
            nbr_defects = defect_coords.shape[0]

            # select edges connecting pairs of defects
            defect_edges = MWPM_edges[(MWPM_edges[:, 0] < nbr_defects) & (MWPM_edges[:, 1] < nbr_defects)]

            # select edges connecting defects to borders
            border_edges = MWPM_edges[(MWPM_edges[:, 0] < nbr_defects) & (MWPM_edges[:, 1] >= nbr_defects)]

            # find coordinates of border defects
            # border edges always start with the 'real' defect
            border_coords = defect_coords[border_edges[:, 0].T, :]

            # eliminate border connected defects
            for coord in border_coords:
                self.eliminate_border_defect(coord, layer)

        else:
            defect_edges = MWPM_edges

        # coordinates of pairs to connect
        start_coords = defect_coords[defect_edges[:, 0].T, :]
        end_coords = defect_coords[defect_edges[:, 1].T, :]

        # iterate through all the mwpm defect pairs
        for start_coord, end_coord in zip(start_coords, end_coords):
            # eliminate the current pair
            self.eliminate_defect_pair(start_coord, end_coord, layer)


    # generates an mwpm solution in a given defect layer
    def generate_MWPM(self, layer, edges, nbr_nodes):
        nbr_edges = edges.shape[0]

        # builds arguments for the bloosom5 program
        processId = os.getpid()
        PATH = str(processId) + 'edges.TXT'
        OUTPUT_PATH = str(processId) +'output.TXT'

        # Save txt file with data for blossom5 to read
        header_str = "{} {}".format(nbr_nodes, nbr_edges)
        np.savetxt(PATH, edges, fmt='%i', header=header_str, comments='')

        # If on windows, the executable file ends in '.exe'
        blossomname = './src/blossom5-v2.05.src/blossom5'
        if os.name == 'windows':
            blossomname += '.exe'
        # Run the blossom5 program as if from the terminal. The devnull part discards any prints from blossom5
        subprocess.call([blossomname, '-e', PATH, '-w', OUTPUT_PATH, '-V'], stdout=open(os.devnull, 'wb'))

        # Read the output from blossom5 and delete the output file
        MWPM_edges = np.loadtxt(OUTPUT_PATH, skiprows=1, dtype=int)
        MWPM_edges = MWPM_edges.reshape((int(nbr_nodes/2), 2))

        # remove files generated by blossom5
        os.remove(PATH)
        os.remove(OUTPUT_PATH)

        return MWPM_edges


    # Solves syndrom according to optimal mwpm
    def solve(self, random_pairing = False):
        for layer in range(2):
            if np.count_nonzero(self.get_layer(layer)) > 0:
                self.generate_solution(layer, random_pairing = random_pairing)

        # applies generated correction chain
        self.code.qubit_matrix ^= self.solution
        self.code.syndrom()


    def generate_classes(self):
        pass

# non periodic manhattan distance between defect at coord1 and coord2
def manhattan_path(defect1, defect2):
    x_distance = np.abs(defect1[0] - defect2[0])
    y_distance = np.abs(defect1[1] - defect2[1])
    return np.array([x_distance, y_distance])


# creates a list of edges connecting nbr_nodes nodes with indices starting at index_offset
def connect_all(nbr_nodes, index_offset=0):
    # list of lists where every node's index is repeated for every connection it 'starts'
    start = []
    # list of lists with the indices of every 'end' node corresponding to the 'start' nodes above
    end = []
    for i in range(nbr_nodes):
        start += [i + index_offset] * (nbr_nodes - i - 1)
        end += [j + index_offset for j in range(i + 1, nbr_nodes)]

    return start, end


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


    p_errors = [0.15]
    system_size = 5
    nbr_of_iterations = 1

    #print(p_errors)
    ground_state_kept_list = []

    # iterate through p_errors
    for p in p_errors:
        print(p)
        ground_states = 0
        # iterate some number of times
        for _ in range(nbr_of_iterations):
            # create a torus and generate error according to p
            code = Planar_code(system_size)
            mwpm = MWPM(code)
            nbr_of_vertex_nodes = 0
            nbr_of_plaquette_nodes = 0

            #code.generate_random_error(p)

            code.qubit_matrix = np.array([[[0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0],
                                           [0, 1, 1, 0, 0],
                                           [0, 1, 1, 0, 0],
                                           [0, 0, 0, 0, 0]],
                                          [[0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]]])
            code.syndrom()

            code.plot('pre')

            MWPM_edges_vertex = []
            edges_vertex = []
            edges_no_periodic_vertex = []
            defect_coords_vertex = []

            MWPM_edges_plaquette = []
            edges_plaquette = []
            edges_no_periodic_plaquette = []
            defect_coords_plaquette = []

            # connect defect pairs given by mwpm
            mwpm.solve(random_pairing = True)

            code.plot('post')

            # check for logical errors
            #code.eval_ground_state()
            #ground_states += code.ground_state

        #ground_state_kept_list.append(ground_states/nbr_of_iterations)

    # store results
    #timestamp = time.ctime()
    try:
        os.mkdir('results')
    except FileExistsError:
        pass

    PATH_ground2 = 'results/p_succes_MWPM_d={}.TXT'.format(system_size)

    np.savetxt(PATH_ground2, ground_state_kept_list, fmt='%e', comments='')




if __name__ == "__main__":
    main(sys.argv[1:])
