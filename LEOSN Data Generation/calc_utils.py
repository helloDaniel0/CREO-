import networkx as nx
import math

f_user = 14 * 10 ** 9  # Hz
B_user = 0.125 * 10 ** 9  # Hz
EIRP_user = 6.02  # dBW
Gr_user = 37.7  # dBi

f_gw = 28.5 * 10 ** 9  # Hz
B_gw = 0.5 * 10 ** 9  # Hz
EIRP_gw = 68.4  # dBW
Gr_gw = 45  # dBi

f_sat = 13.5 * 10 ** 9  # Hz
B_sat = 0.25 * 10 ** 9  # Hz
EIRP_sat = 66.9  # dBW
Gr_sat = 40.9  # dBi

f = [f_user, f_sat, f_gw]
B = [B_user, B_sat, B_gw]
EIRP = [EIRP_user / 4, EIRP_sat / 9, EIRP_gw / 8]
Gr = [Gr_user, Gr_sat, Gr_gw]

T = [535.9, 362.9]  # K
k_0 = 1.38 * 10 ** (-23)
AS = 2.9


def calculate_FSPL(d, type):
    return 20 * math.log10(d) + 20 * math.log10(f[type]) - 147.55 - 2.9  # dB


def calculate_Rx(d, type):
    fspl = calculate_FSPL(d, type)
    return EIRP[type] + Gr[type] - fspl  # dBW


def calculate_SNR(d, type):
    Pr = calculate_Rx(d, type)
    if type == 0 or type == 2:
        Pn = 10 * math.log10(k_0 * T[1] * B[type])
    else:
        Pn = 10 * math.log10(k_0 * T[0] * B[type])
    return Pr - Pn  # dB


def calculate_bw(point_a_cbf, point_b_cbf, type):
    d = calculate_dis(point_a_cbf, point_b_cbf)
    snr = 10 ** (calculate_SNR(d, type) / 10)
    return B[type] * math.log2(1 + snr)


def degToRad(deg):
    pi = math.pi
    return deg * pi / 180


def judge_neighbor_sat_node(i, j, orbits, sats):
    if j == (i // sats) * sats + (i + 1) % sats or j == (i // sats) * sats + (i - 1) % sats or j == (i + sats) % (
            orbits * sats) or j == (i - sats) % (orbits * sats):
        return True
    else:
        return False


"""
    input: (lat, long, alt)
    output: (x, y, z)
    algorithm borrowed from StarPerf's MATLAB codes; conversion of coordinate systems
"""


def lla2cbf(position):
    R = 6371 * (10 ** 3)
    pi = math.pi
    r = R + position[2]
    theta = pi / 2 - position[0] * pi / 180
    phi = 2 * pi + position[1] * pi / 180
    x = (r * math.sin(theta)) * math.cos(phi)
    y = (r * math.sin(theta)) * math.sin(phi)
    z = r * math.cos(theta)
    return (x, y, z)


"""
    get the ground converage limit L of a celestial object
    see README.md for more details
"""


def getCoverageLimitL(elevation, depression, altitude):
    R = 6371 * 10 ** 3
    gamma = 90 - (elevation + depression)
    l = R * math.sin(degToRad(gamma))
    theta = (180 - gamma) / 2
    d = l / (math.tan(degToRad(theta)))
    L = math.sqrt((altitude + d) ** 2 + l ** 2)
    return L


'''
    given coordinates of a satellite, a ground station, and coverage limit L, check 
    if the satellite can cover the ground station
'''


def checkSatCoverGroundStation(sat_position_cbf, gs_position_cbf, L):
    x1, y1, z1 = sat_position_cbf
    x2, y2, z2 = gs_position_cbf
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return dist <= L


'''
    compute (electromagnetic) wave's propogation latency between two points in vacuum
'''


def computeLatency(point_a_cbf, point_b_cbf):
    dist = math.sqrt((point_a_cbf[0] - point_b_cbf[0]) ** 2 + (point_a_cbf[1] - point_b_cbf[1]) ** 2 + (
            point_a_cbf[2] - point_b_cbf[2]) ** 2)
    return dist / 299792458


'''
    given a source, destination, and the graph, check whether an e2e path exists
'''


def pathExists(gw1, gw2, G):
    if gw1 == -1 or gw2 == -1:
        return False
    return nx.has_path(G, gw1, gw2)


def calculate_dis(point_a_cbf, point_b_cbf):
    """
    计算两个节点之间的欧几里得距离。

    参数:
    x1, y1, z1: 第一个节点的CBF坐标
    x2, y2, z2: 第二个节点的CBF坐标

    返回:
    两个节点之间的距离
    """
    dist = math.sqrt((point_a_cbf[0] - point_b_cbf[0]) ** 2 + (point_a_cbf[1] - point_b_cbf[1]) ** 2 + (
            point_a_cbf[2] - point_b_cbf[2]) ** 2)
    return dist
