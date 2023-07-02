import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cartesian_to_spherical(cartesian_points):
    # Convert Cartesian coordinates to spherical coordinates
    r = np.linalg.norm(cartesian_points, axis=1)  # Radial distance
    theta = np.arccos(cartesian_points[:, 2] / r)  # Inclination angle
    phi = np.arctan2(cartesian_points[:, 1], cartesian_points[:, 0])  # Azimuth angle

    # Set of points in spherical coordinates
    spherical_points = np.column_stack((r, theta, phi))
    return spherical_points

def spherical_to_cartesian(spherical_points):
    r = spherical_points[:, 0]  # Radial distance
    theta = spherical_points[:, 1]  # Inclination angle
    phi = spherical_points[:, 2]  # Azimuth angle

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Set of points in Cartesian coordinates
    cartesian_points = np.column_stack((x, y, z))
    return cartesian_points

# First chain of vectors
points1 = np.array([[0, 0, 0],  # Point 1: Origin
                    [0, -0.18, 0],  # Point 2
                    [0.153, -0.331, -0.125],  # Point 3
                    [0.319, -0.235, -0.286]])  # Point 4

# Second chain of vectors
points3 = np.array([[0, 0, 0],  # Point 1: Origin
                    [0, 0.18, 0],  # Point 2
                    [0.153, 0.331, -0.125],  # Point 3
                    [0.319, 0.235, -0.286]])  # Point 4

robot_arm_dimensions = np.array([0.08, 0.12, 0.18])

vectors1 = np.diff(points1, axis=0)
vectors3 = np.diff(points3, axis=0)
spherical_points_1 = cartesian_to_spherical(vectors1)
spherical_points_3 = cartesian_to_spherical(vectors3)
spherical_points_1[:, 0] = robot_arm_dimensions
spherical_points_3[:, 0] = robot_arm_dimensions
points2 = spherical_to_cartesian(np.vstack((np.array([0, 0, 0]), spherical_points_1)))
points4 = spherical_to_cartesian(np.vstack((np.array([0, 0, 0]), spherical_points_3)))
points2 = cumulative_matrix = np.cumsum(points2, axis=0)
points4 = cumulative_matrix = np.cumsum(points4, axis=0)

points = np.concatenate((points1, points2, points3, points4))

# Extract the coordinates for plotting
X, Y, Z = zip(*points)

# Create a 3D plot
fig = plt.figure(figsize=(12, 5))

# First subplot
ax1 = fig.add_subplot(121, projection='3d')

# Plot the points and vectors for the first chain
for i, (x, y, z) in enumerate(zip(X[:len(points1)], Y[:len(points1)], Z[:len(points1)])):
    ax1.scatter(x, y, z, c='b', marker='o')
    ax1.text(x, y, z, f'({x}, {y}, {z})', color='black', fontsize=8, ha='center')

# Plot the lines connecting the points of the first chain
colors1 = plt.cm.jet(np.linspace(0, 1, len(points1) - 1))
for i in range(len(points1) - 1):
    ax1.plot([X[i], X[i + 1]], [Y[i], Y[i + 1]], [Z[i], Z[i + 1]], c=colors1[i], linewidth=2)
    distance = np.linalg.norm(points1[i + 1] - points1[i])
    mid_point = (points1[i + 1] + points1[i]) / 2
    ax1.text(mid_point[0], mid_point[1], mid_point[2], f'{distance:.2f}', color=colors1[i], fontsize=10, ha='center')

# Plot the second chain of vectors starting from the origin
colors3 = plt.cm.jet(np.linspace(0, 1, len(points3) - 1))
for i in range(len(points3) - 1):
    ax1.plot([points3[i, 0], points3[i + 1, 0]], [points3[i, 1], points3[i + 1, 1]], [points3[i, 2], points3[i + 1, 2]],
             c=colors3[i], linestyle='--', linewidth=2)
    distance = np.linalg.norm(points3[i + 1] - points3[i])
    mid_point = (points3[i + 1] + points3[i]) / 2
    ax1.text(mid_point[0], mid_point[1], mid_point[2],
             f'({points3[i + 1, 0]:.2f}, {points3[i + 1, 1]:.2f}, {points3[i + 1, 2]:.2f})\n{distance:.2f}',
             color=colors3[i], fontsize=10, ha='center')

ax1.set_xlim([-0.5, 0.5])
ax1.set_ylim([-0.5, 0.5])
ax1.set_zlim([-0.5, 0.5])

# Set the axis labels for the first subplot
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Second subplot
ax2 = fig.add_subplot(122, projection='3d')

# Plot the points and vectors for the second chain
for i, (x, y, z) in enumerate(zip(X[len(points1):len(points1) + len(points2)],
                                   Y[len(points1):len(points1) + len(points2)],
                                   Z[len(points1):len(points1) + len(points2)])):
    ax2.scatter(x, y, z, c='b', marker='o')
    ax2.text(x, y, z, f'({x}, {y}, {z})', color='black', fontsize=8, ha='left')

# Plot the lines connecting the points of the second chain
colors2 = plt.cm.jet(np.linspace(0, 1, len(points2) - 1))
for i in range(len(points2) - 1):
    ax2.plot([X[i + len(points1)], X[i + len(points1) + 1]], [Y[i + len(points1)], Y[i + len(points1) + 1]],
             [Z[i + len(points1)], Z[i + len(points1) + 1]], c=colors2[i], linewidth=2)
    distance = np.linalg.norm(points2[i + 1] - points2[i])
    mid_point = (points2[i + 1] + points2[i]) / 2
    ax2.text(mid_point[0], mid_point[1], mid_point[2], f'{distance:.2f}', color=colors2[i], fontsize=10, ha='left')

# Plot the third chain of vectors starting from the origin
colors4 = plt.cm.jet(np.linspace(0, 1, len(points4) - 1))
for i in range(len(points4) - 1):
    ax2.plot([points4[i, 0], points4[i + 1, 0]], [points4[i, 1], points4[i + 1, 1]], [points4[i, 2], points4[i + 1, 2]],
             c=colors4[i], linestyle='--', linewidth=2)
    distance = np.linalg.norm(points4[i + 1] - points4[i])
    mid_point = (points4[i + 1] + points4[i]) / 2
    ax2.text(mid_point[0], mid_point[1], mid_point[2],
             f'({points4[i + 1, 0]:.2f}, {points4[i + 1, 1]:.2f}, {points4[i + 1, 2]:.2f})\n{distance:.2f}',
             color=colors4[i], fontsize=10, ha='left')

ax2.set_xlim([-0.2, 0.2])
ax2.set_ylim([-0.2, 0.2])
ax2.set_zlim([-0.2, 0.2])

# Set the axis labels for the second subplot
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Set the title for both subplots
ax1.set_title('Human')
ax2.set_title('Robot')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.3)

# Show the plot
plt.show()
