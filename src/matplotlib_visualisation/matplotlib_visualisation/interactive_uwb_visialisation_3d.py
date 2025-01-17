import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.widgets import Button
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import euclidean
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Draggable3DPoint:
    point_list = []
    lock = None  # Only allow dragging one point at a time
    gdop_text = None  # Hold the text for GDOP
    uncertainty_volumes = []
    def __init__(self, ax, xs, ys, zs, std_deviation=0.1):
        self.ax = ax
        self.fig = ax.figure
        self.canvas = self.fig.canvas
        self.press = None

        # Initial positions
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.zs = np.array(zs)
        self.std_deviation = std_deviation

        # Scatter plot
        self.points = ax.scatter(xs, ys, zs, color='blue')

        self.cid_press = self.points.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.points.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.points.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidkey = self.points.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

        Draggable3DPoint.point_list.append(self)
        Draggable3DPoint.update_gdop()
        self.update_uncertainty_volume()

    def on_key_press(self, event):
        if event.key == 'ctrl+c':
            plt.close(self.points.figure)  # Close the plot

    def on_press(self, event):
        if event.inaxes != self.ax or Draggable3DPoint.lock is not None:
            return

        # Find the point closest to the click in 2D projection
        closest_point = self.get_closest_point(event)

        if closest_point is not None:
            self.press = closest_point
            # Disable rotation
            self.ax.mouse_init(rotate_btn=None, zoom_btn=None)
            Draggable3DPoint.lock = self

    def on_release(self, event):
        if Draggable3DPoint.lock is not self:
            return

        self.press = None
        Draggable3DPoint.lock = None
        # Re-enable rotation
        self.ax.mouse_init()
        self.points.figure.canvas.draw()
        Draggable3DPoint.update_gdop()
        self.update_uncertainty_volume()

    def on_motion(self, event):
        if self.press is None or Draggable3DPoint.lock is not self or event.inaxes != self.ax:
            return

        ind, x0, y0, z0 = self.press

        # Calculate movement in screen coordinates
        dx = event.x - x0
        dy = event.y - y0

        # Current 3D coordinates of the point
        current_x, current_y, current_z = self.xs[ind], self.ys[ind], self.zs[ind]

        t = np.array([current_x, current_y, current_z])

        # Get the view transformation matrix
        theta_x = np.radians(self.ax.elev + 90)
        theta_z = np.radians(self.ax.azim - 90)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        # Rotation matrix to align the point with the view direction
        R = np.dot(Rz, Rx)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        translation_B = np.array([-dx, +dy, 0])
        translation_B_homogeneous = np.append(translation_B, 1)  # Convert to homogeneous coordinates

        # Apply the transformation T to the point in frame B
        P_prime_homogeneous = np.dot(T, translation_B_homogeneous)

        # Convert back to Cartesian coordinates
        dx, dy, dz = P_prime_homogeneous[:3] - t

        # Calculate the effect of delta movement in the new coordinate system on the original coordinates
        scale_x = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) / self.ax.bbox.width
        scale_y = (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) / self.ax.bbox.height
        scale_z = (self.ax.get_zlim()[1] - self.ax.get_zlim()[0]) / self.ax.bbox.height

        self.xs[ind] += dx * scale_x * 1.5
        self.ys[ind] += dy * scale_y * 1.5
        self.zs[ind] += dz * scale_z * 1.5

        self.update()
        self.update_gdop()
        self.press = (ind, event.x, event.y, z0)

    def get_closest_point(self, event):
        screen_coords = []
        for x, y, z in zip(self.xs, self.ys, self.zs):
            x2, y2, _ = proj_transform(x, y, z, self.ax.get_proj())
            x_screen, y_screen = self.ax.transData.transform((x2, y2))
            screen_coords.append((x_screen, y_screen))
        
        screen_coords = np.array(screen_coords)
        distances = np.sqrt((screen_coords[:, 0] - event.x)**2 + (screen_coords[:, 1] - event.y)**2)
        ind = np.argmin(distances)

        if distances[ind] < 10:  # Sensitivity threshold in pixels
            return ind, event.x, event.y, self.zs[ind]
        return None

    def update(self):
        self.points._offsets3d = (self.xs, self.ys, self.zs)
        self.ax.draw_artist(self.points)
        self.canvas.blit(self.ax.bbox)



    def update_uncertainty_volume(self):
        pass 
    




    @staticmethod
    def update_gdop():
        positions = np.array([[p.xs[0], p.ys[0], p.zs[0]] for p in Draggable3DPoint.point_list])
        if len(positions) < 2:
            return  # Not enough points to calculate GDOP
        A = np.hstack((positions, np.ones((positions.shape[0], 1))))  # Augmenting with ones for affine transformations
        try:
            inv_at_a = np.linalg.inv(A.T @ A)
            gdop = np.sqrt(np.trace(inv_at_a))
            if Draggable3DPoint.gdop_text:
                Draggable3DPoint.gdop_text.set_text(f'GDOP: {gdop:.2f}')
            else:
                Draggable3DPoint.gdop_text = Draggable3DPoint.point_list[0].ax.text(0.5, 0.95, 1,f'GDOP: {gdop:.2f}', transform=Draggable3DPoint.point_list[0].ax.transAxes, ha='center', fontsize=12, color='red')
        except np.linalg.LinAlgError:
            pass  # Matrix is singular, cannot compute GDOP

# Function to add a new point
def add_point(event):
    new_xs = np.random.uniform(-1, 1, 1)
    new_ys = np.random.uniform(-1, 1, 1)
    new_zs = np.random.uniform(-1, 1, 1)
    Draggable3DPoint(ax, new_xs, new_ys, new_zs)
    plt.draw()

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

anchor_pos = (0, 0, 0)
ax.plot(*anchor_pos, 'o', color='red', label='Anchor')

# Add some initial points
initial_xs = np.random.rand(1)
initial_ys = np.random.rand(1)
initial_zs = np.random.rand(1)
Draggable3DPoint(ax, initial_xs, initial_ys, initial_zs)

ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Add Point')
button.on_clicked(add_point)

plt.show()
