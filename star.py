import numpy as np
from sgp4.api import Satrec
from sgp4.api import jday
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
import time


def extendibility_centrality(G, node):
    """
    Calculate the extendibility centrality of a node in a graph.
    """
    neighbors = list(G.neighbors(node))
    ec = sum(G.degree(v) for v in neighbors)
    return ec


def find_min_degree_node(G, excluded_nodes):
    """
    Find the minimum degree node in the graph that is not in excluded_nodes,
    breaking ties using extendibility centrality.
    """
    min_degree = min(G.degree(v) for v in G.nodes())
    min_degree_nodes = [
        v for v in G.nodes() if G.degree(v) == min_degree and v not in excluded_nodes
    ]
    if not min_degree_nodes:
        return None  # No suitable node found
    min_ec_node = min(min_degree_nodes, key=lambda v: extendibility_centrality(G, v))
    return min_ec_node


def find_max_distance_node(positions, start_node):
    """
    Find the node with the maximum distance from the start node, breaking ties using extendibility centrality.
    """
    distances = np.linalg.norm(positions - positions[start_node], axis=1)
    distances[start_node] = -np.inf
    max_distance_node = np.argmax(distances)
    return max_distance_node


def mdmd(
    G, positions, fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right
):
    num_satellites = len(positions)
    distance_average = []
    # Initialize existing connections with no connections
    existing_connections = {
        i: {"front": None, "behind": None, "left": None, "right": None}
        for i in range(num_satellites)
    }

    # Track nodes that cannot be connected in the current iteration
    excluded_nodes = set()

    # Iterate over each satellite to find a connection in each direction
    for i in range(num_satellites):
        # Identify the satellite with the minimum degree to start the MDMD process, excluding any that cannot be connected
        min_degree_node = find_min_degree_node(G, excluded_nodes)
        if min_degree_node is None:
            break
        # A variable to track if at least one connection was made for the min_degree_node
        connection_made = False

        # For each direction, find the farthest satellite within the FOV
        for direction, fov_matrix, opposite in zip(
            ["front", "behind", "left", "right"],
            [fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right],
            ["behind", "front", "right", "left"],
        ):

            distances = np.linalg.norm(positions - positions[min_degree_node], axis=1)
            distances[fov_matrix[min_degree_node] == 0] = -1
            while np.max(distances) != -1:
                furthest_idx = np.argmax(distances)
                if (
                    existing_connections[min_degree_node][direction] is None
                    and existing_connections[furthest_idx][opposite] is None
                ):
                    # Mark the potential connection
                    existing_connections[min_degree_node][direction] = furthest_idx
                    existing_connections[furthest_idx][opposite] = min_degree_node
                    G.add_edge(furthest_idx, min_degree_node)
                    distance_average.append(distances[furthest_idx])
                    connection_made = True
                    break
                else:
                    # Mark this satellite as unreachable and look for the next one
                    distances[furthest_idx] = -1
        if not connection_made:
            excluded_nodes.add(min_degree_node)
    return G, distance_average


def read_tle_file(file_path):
    with open(file_path, "r") as file:
        tle_data = file.readlines()
    tle_sets = []
    for i in range(0, len(tle_data), 3):
        satellite_name = tle_data[i].strip()
        line1 = tle_data[i + 1].strip()
        line2 = tle_data[i + 2].strip()
        tle_sets.append((satellite_name, line1, line2))
    return tle_sets


def calculate_cartesian_coordinates(tle_data, epoch):
    num_satellites = len(tle_data)
    positions = np.zeros((num_satellites, 3))
    velocities = np.zeros((num_satellites, 3))

    year = epoch.year
    month = epoch.month
    day = epoch.day
    hour = epoch.hour
    minute = epoch.minute
    second = epoch.second
    jd, fr = jday(year, month, day, hour, minute, second)

    for i in range(num_satellites):
        satellite = Satrec.twoline2rv(tle_data[i][1], tle_data[i][2])
        e, position, velocity = satellite.sgp4(jd, fr)
        positions[i, :] = tuple([1000 * x for x in position])
        velocities[i, :] = tuple([1000 * x for x in velocity])

    return positions, velocities


def calculate_relative_positions_all(positions, headings):
    num_satellites = len(positions)
    relative_positions_all = np.zeros((num_satellites, num_satellites, 2))

    for i in range(num_satellites):
        diffs = positions - positions[i]
        distances = np.linalg.norm(diffs, axis=1)
        angles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
        relative_angles = angles - headings[i]
        relative_angles = (relative_angles + 360) % 360
        relative_positions_all[i, :, 0] = distances
        relative_positions_all[i, :, 1] = relative_angles

    return relative_positions_all


def create_field_of_view_matrices(relative_positions_all, fov_angle, max_distance):
    num_satellites = len(relative_positions_all)
    fov_half_angle = fov_angle / 2
    fov_matrix_front = np.zeros((num_satellites, num_satellites), dtype=int)
    fov_matrix_behind = np.zeros((num_satellites, num_satellites), dtype=int)
    fov_matrix_left = np.zeros((num_satellites, num_satellites), dtype=int)
    fov_matrix_right = np.zeros((num_satellites, num_satellites), dtype=int)

    angles = relative_positions_all[:, :, 1]
    distances = relative_positions_all[:, :, 0]
    front_mask = (360 - fov_half_angle <= angles) | (angles < fov_half_angle)
    right_mask = (90 - fov_half_angle <= angles) & (angles < 90 + fov_half_angle)
    behind_mask = (180 - fov_half_angle <= angles) & (angles < 180 + fov_half_angle)
    left_mask = (270 - fov_half_angle <= angles) & (angles < 270 + fov_half_angle)

    fov_matrix_front[(front_mask) & (distances <= max_distance)] = 1
    fov_matrix_behind[(behind_mask) & (distances <= max_distance)] = 1
    fov_matrix_left[(left_mask) & (distances <= max_distance)] = 1
    fov_matrix_right[(right_mask) & (distances <= max_distance)] = 1

    return fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right


def generate_network(positions):
    num_satellites = len(positions)
    G = nx.Graph()

    # Add nodes (satellites) to the graph
    for i in range(num_satellites):
        G.add_node(i)

    return G


def greedy_search_furthest(
    G, positions, fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right
):
    num_satellites = len(positions)
    distance_average = []
    # Track potential connections without adding them immediately
    existing_connections = {
        i: {"front": None, "behind": None, "left": None, "right": None}
        for i in range(num_satellites)
    }

    # Identify potential connections
    for i in range(num_satellites):
        for direction, fov_matrix, opposite in zip(
            ["front", "behind", "left", "right"],
            [fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right],
            ["behind", "front", "right", "left"],
        ):
            # Set unreachable distances to satellites not in the FOV
            distances = np.linalg.norm(positions - positions[i], axis=1)
            distances[fov_matrix[i] == 0] = -1

            # Changed from np.min(distances) to np.max(distances)
            while np.max(distances) != -1:
                # Changed from np.argmin(distances) to np.argmax(distances)
                furthest_idx = np.argmax(distances)
                if (
                    existing_connections[i][direction] is None
                    and existing_connections[furthest_idx][opposite] is None
                ):
                    # Mark the potential connection
                    existing_connections[i][direction] = furthest_idx
                    existing_connections[furthest_idx][opposite] = i
                    distance_average.append(distances[furthest_idx])
                    break
                else:
                    # Mark this satellite as unreachable and look for the next one
                    distances[furthest_idx] = -1

    # Add edges based on the final existing_connections
    for satellite, connections in existing_connections.items():
        for direction, target in connections.items():
            if target is not None and not G.has_edge(satellite, target):
                G.add_edge(satellite, target)

    return G, distance_average


def greedy_search(
    G, positions, fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right
):
    num_satellites = len(positions)
    max_distance = 999999999999  # Use this for unreachable distances
    distance_average = []
    # Track potential connections without adding them immediately
    existing_connections = {
        i: {"front": None, "behind": None, "left": None, "right": None}
        for i in range(num_satellites)
    }

    # Identify potential connections
    for i in range(num_satellites):
        for direction, fov_matrix, opposite in zip(
            ["front", "behind", "left", "right"],
            [fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right],
            ["behind", "front", "right", "left"],
        ):
            # Set unreachable distances to satellites not in the FOV
            distances = np.linalg.norm(positions - positions[i], axis=1)
            distances[fov_matrix[i] == 0] = max_distance

            while np.min(distances) != max_distance:
                closest_idx = np.argmin(distances)
                if (
                    existing_connections[i][direction] is None
                    and existing_connections[closest_idx][opposite] is None
                ):
                    # Mark the potential connection
                    existing_connections[i][direction] = closest_idx
                    existing_connections[closest_idx][opposite] = i

                    distance_average.append(distances[closest_idx])
                    break
                else:
                    # Mark this satellite as unreachable and look for the next one
                    distances[closest_idx] = max_distance
    for satellite, connections in existing_connections.items():
        for direction, target in connections.items():
            if target is not None and not G.has_edge(satellite, target):
                G.add_edge(satellite, target)
    return G, distance_average


tle_file_path = "starlink.txt"
tle_data = read_tle_file(tle_file_path)
greedy_times = []
mdmd_times = []
greedy_furthest_times = []
greedy_connectivity = []
mdmd_connectivity = []
greedy_furthest_connectivity = []
iterations = []
iteration_number = 0
start_epoch = datetime.utcnow()
end_epoch = start_epoch + timedelta(days=1)
time_step = timedelta(hours=1)
current_epoch = start_epoch
while current_epoch <= end_epoch:
    positions, velocities = calculate_cartesian_coordinates(tle_data, current_epoch)
    headings = np.degrees(np.arctan2(velocities[:, 1], velocities[:, 0]))
    headings = (headings + 360) % 360
    relative_positions_all = calculate_relative_positions_all(positions, headings)

    fov_angle = 60
    max_distance = 5016000
    fov_matrix_front, fov_matrix_behind, fov_matrix_left, fov_matrix_right = (
        create_field_of_view_matrices(relative_positions_all, fov_angle, max_distance)
    )
    np.fill_diagonal(fov_matrix_front, 0)
    np.fill_diagonal(fov_matrix_behind, 0)
    np.fill_diagonal(fov_matrix_left, 0)
    np.fill_diagonal(fov_matrix_right, 0)

    # Generate the initial network graph
    G = generate_network(positions)
    start_time = time.time()
    G_greedy, greedy_close_distance = greedy_search(
        G.copy(),
        positions,
        fov_matrix_front,
        fov_matrix_behind,
        fov_matrix_left,
        fov_matrix_right,
    )
    greedy_time = time.time() - start_time
    greedy_times.append(greedy_time)

    # MDMD
    start_time = time.time()
    G_mdmd, mdmd_distance = mdmd(
        G.copy(),
        positions,
        fov_matrix_front,
        fov_matrix_behind,
        fov_matrix_left,
        fov_matrix_right,
    )
    mdmd_time = time.time() - start_time
    mdmd_times.append(mdmd_time)

    # Greedy furthest
    start_time = time.time()
    G_greedy_fur, greedy_fur_distance = greedy_search_furthest(
        G.copy(),
        positions,
        fov_matrix_front,
        fov_matrix_behind,
        fov_matrix_left,
        fov_matrix_right,
    )
    greedy_furthest_time = time.time() - start_time
    greedy_furthest_times.append(greedy_furthest_time)

    # Calculate algebraic connectivity
    greedy_connectivity.append(nx.algebraic_connectivity(G_greedy, method="lanczos"))
    mdmd_connectivity.append(nx.algebraic_connectivity(G_mdmd, method="lanczos"))
    greedy_furthest_connectivity.append(
        nx.algebraic_connectivity(G_greedy_fur, method="lanczos")
    )
    print(
        sum(greedy_close_distance) / len(greedy_close_distance),
        sum(mdmd_distance) / len(mdmd_distance),
        sum(greedy_fur_distance) / len(greedy_fur_distance),
    )
    # Record the iteration number
    iterations.append(iteration_number)
    iteration_number += 1
    print(current_epoch)
    # Increment the epoch
    current_epoch += time_step

average_greedy_time = sum(greedy_times) / len(greedy_times)
average_mdmd_time = sum(mdmd_times) / len(mdmd_times)
average_greedy_furthest_time = sum(greedy_furthest_times) / len(greedy_furthest_times)
print(average_greedy_time, average_mdmd_time, average_greedy_furthest_time)
print(
    sum(greedy_connectivity) / len(greedy_connectivity),
    sum(mdmd_connectivity) / len(mdmd_connectivity),
    sum(greedy_furthest_connectivity) / len(greedy_furthest_connectivity),
)
# Plot algebraic connectivity for each method
plt.figure(figsize=(10, 5))
plt.plot(iterations, greedy_connectivity, label="Greedy Closest", marker="o")
plt.plot(iterations, mdmd_connectivity, label="MDMD", marker="s")
plt.plot(iterations, greedy_furthest_connectivity, label="Greedy Furthest", marker="^")
plt.xlabel("Iteration Number")
plt.ylabel("Algebraic Connectivity")
plt.title("Algebraic Connectivity vs Iteration Number")
plt.legend()
plt.grid(True)
plt.show()
