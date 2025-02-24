import numpy as np
from enum import Enum

class BodyIdentifier(Enum):
    INDEX_FINGER_DIP_right = 0
    INDEX_FINGER_MCP_right = 1
    INDEX_FINGER_PIP_right = 2
    INDEX_FINGER_TIP_right = 3
    MIDDLE_FINGER_DIP_right = 4
    MIDDLE_FINGER_MCP_right = 5
    MIDDLE_FINGER_PIP_right = 6
    MIDDLE_FINGER_TIP_right = 7
    PINKY_DIP_right = 8
    PINKY_MCP_right = 9
    PINKY_PIP_right = 10
    PINKY_TIP_right = 11
    RING_FINGER_DIP_right = 12
    RING_FINGER_MCP_right = 13
    RING_FINGER_PIP_right = 14
    RING_FINGER_TIP_right = 15
    THUMB_CMC_right = 16
    THUMB_IP_right = 17
    THUMB_MCP_right = 18
    THUMB_TIP_right = 19
    WRIST_right = 20
    INDEX_FINGER_DIP_left = 21
    INDEX_FINGER_MCP_left = 22
    INDEX_FINGER_PIP_left = 23
    INDEX_FINGER_TIP_left = 24
    MIDDLE_FINGER_DIP_left = 25
    MIDDLE_FINGER_MCP_left = 26
    MIDDLE_FINGER_PIP_left = 27
    MIDDLE_FINGER_TIP_left = 28
    PINKY_DIP_left = 29
    PINKY_MCP_left = 30
    PINKY_PIP_left = 31
    PINKY_TIP_left = 32
    RING_FINGER_DIP_left = 33
    RING_FINGER_MCP_left = 34
    RING_FINGER_PIP_left = 35
    RING_FINGER_TIP_left = 36
    THUMB_CMC_left = 37
    THUMB_IP_left = 38
    THUMB_MCP_left = 39
    THUMB_TIP_left = 40
    WRIST_left = 41
    RIGHT_SHOULDER = 42
    LEFT_SHOULDER = 43
    LEFT_ELBOW = 44
    RIGHT_ELBOW = 45

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'mediapipe':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(20, 18), (18, 16), (20, 16), (16, 22), (16, 14), (14, 12),
                              (19, 17), (17, 15), (19, 15), (15, 21), (15, 13), (13, 11),
                              (12, 11), (12, 24), (24, 23), (23, 11),
                              (10, 9), 
                              (0, 4), (4, 5), (5, 6), (6, 8),
                              (0, 1), (1, 2), (2, 3), (3, 7)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 10

        elif layout == "mediapipe_two_hand":
            self.num_node = 46
            self_link = [(i, i) for i in range(self.num_node)]            
            neighbor_1base = [(BodyIdentifier.WRIST_left.value, BodyIdentifier.THUMB_CMC_left.value), 
                              (BodyIdentifier.THUMB_CMC_left.value, BodyIdentifier.THUMB_MCP_left.value),
                              (BodyIdentifier.THUMB_MCP_left.value, BodyIdentifier.THUMB_IP_left.value),
                              (BodyIdentifier.THUMB_IP_left.value, BodyIdentifier.THUMB_TIP_left.value),

                              (BodyIdentifier.WRIST_left.value, BodyIdentifier.INDEX_FINGER_MCP_left.value),
                              (BodyIdentifier.INDEX_FINGER_MCP_left.value, BodyIdentifier.INDEX_FINGER_PIP_left.value),
                              (BodyIdentifier.INDEX_FINGER_PIP_left.value, BodyIdentifier.INDEX_FINGER_DIP_left.value),
                              (BodyIdentifier.INDEX_FINGER_DIP_left.value, BodyIdentifier.INDEX_FINGER_TIP_left.value),

                              (BodyIdentifier.INDEX_FINGER_MCP_left.value, BodyIdentifier.MIDDLE_FINGER_MCP_left.value),
                              (BodyIdentifier.MIDDLE_FINGER_MCP_left.value, BodyIdentifier.MIDDLE_FINGER_PIP_left.value),
                              (BodyIdentifier.MIDDLE_FINGER_PIP_left.value, BodyIdentifier.MIDDLE_FINGER_DIP_left.value),
                              (BodyIdentifier.MIDDLE_FINGER_DIP_left.value, BodyIdentifier.MIDDLE_FINGER_TIP_left.value),

                              (BodyIdentifier.MIDDLE_FINGER_MCP_left.value, BodyIdentifier.RING_FINGER_MCP_left.value),
                              (BodyIdentifier.RING_FINGER_MCP_left.value, BodyIdentifier.RING_FINGER_PIP_left.value),
                              (BodyIdentifier.RING_FINGER_PIP_left.value, BodyIdentifier.RING_FINGER_DIP_left.value),
                              (BodyIdentifier.RING_FINGER_DIP_left.value, BodyIdentifier.RING_FINGER_TIP_left.value),

                              (BodyIdentifier.WRIST_left.value, BodyIdentifier.PINKY_MCP_left.value),
                              (BodyIdentifier.PINKY_MCP_left.value, BodyIdentifier.PINKY_PIP_left.value),
                              (BodyIdentifier.PINKY_PIP_left.value, BodyIdentifier.PINKY_DIP_left.value),
                              (BodyIdentifier.PINKY_DIP_left.value, BodyIdentifier.PINKY_TIP_left.value),

                              # RIGHT HAND
                              (BodyIdentifier.WRIST_right.value, BodyIdentifier.THUMB_CMC_right.value), 
                              (BodyIdentifier.THUMB_CMC_right.value, BodyIdentifier.THUMB_MCP_right.value),
                              (BodyIdentifier.THUMB_MCP_right.value, BodyIdentifier.THUMB_IP_right.value),
                              (BodyIdentifier.THUMB_IP_right.value, BodyIdentifier.THUMB_TIP_right.value),

                              (BodyIdentifier.WRIST_right.value, BodyIdentifier.INDEX_FINGER_MCP_right.value),
                              (BodyIdentifier.INDEX_FINGER_MCP_right.value, BodyIdentifier.INDEX_FINGER_PIP_right.value),
                              (BodyIdentifier.INDEX_FINGER_PIP_right.value, BodyIdentifier.INDEX_FINGER_DIP_right.value),
                              (BodyIdentifier.INDEX_FINGER_DIP_right.value, BodyIdentifier.INDEX_FINGER_TIP_right.value),

                              (BodyIdentifier.INDEX_FINGER_MCP_right.value, BodyIdentifier.MIDDLE_FINGER_MCP_right.value),
                              (BodyIdentifier.MIDDLE_FINGER_MCP_right.value, BodyIdentifier.MIDDLE_FINGER_PIP_right.value),
                              (BodyIdentifier.MIDDLE_FINGER_PIP_right.value, BodyIdentifier.MIDDLE_FINGER_DIP_right.value),
                              (BodyIdentifier.MIDDLE_FINGER_DIP_right.value, BodyIdentifier.MIDDLE_FINGER_TIP_right.value),

                              (BodyIdentifier.MIDDLE_FINGER_MCP_right.value, BodyIdentifier.RING_FINGER_MCP_right.value),
                              (BodyIdentifier.RING_FINGER_MCP_right.value, BodyIdentifier.RING_FINGER_PIP_right.value),
                              (BodyIdentifier.RING_FINGER_PIP_right.value, BodyIdentifier.RING_FINGER_DIP_right.value),
                              (BodyIdentifier.RING_FINGER_DIP_right.value, BodyIdentifier.RING_FINGER_TIP_right.value),

                              (BodyIdentifier.WRIST_right.value, BodyIdentifier.PINKY_MCP_right.value),
                              (BodyIdentifier.PINKY_MCP_right.value, BodyIdentifier.PINKY_PIP_right.value),
                              (BodyIdentifier.PINKY_PIP_right.value, BodyIdentifier.PINKY_DIP_right.value),
                              (BodyIdentifier.PINKY_DIP_right.value, BodyIdentifier.PINKY_TIP_right.value),

                              # 2 HAND + SHOULDER + ELBOW
                              (BodyIdentifier.RIGHT_SHOULDER.value, BodyIdentifier.RIGHT_ELBOW.value),
                              (BodyIdentifier.RIGHT_ELBOW.value, BodyIdentifier.WRIST_right.value),

                              (BodyIdentifier.RIGHT_SHOULDER.value, BodyIdentifier.LEFT_SHOULDER.value),

                              (BodyIdentifier.LEFT_SHOULDER.value, BodyIdentifier.LEFT_ELBOW.value),
                              (BodyIdentifier.LEFT_ELBOW.value, BodyIdentifier.WRIST_left.value)]
            
            neighbor_link = [(i, j) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = BodyIdentifier.RIGHT_SHOULDER.value
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    print(edge)
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD