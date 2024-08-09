import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Point
import numpy as np
from matplotlib.widgets import Button


class DraggablePoint:
    lock = None  # Only allow dragging one point at a time
    points = []  # Class variable to hold all points for intersection calculation
    intersection_patch = None  # Hold the patch for intersection
    intersection_area_text = None  # Hold the text for intersection area
    gdop_text = None  # Hold the text for GDOP

    def __init__(self, ax, anchor, init_pos, std_dev):
        self.ax = ax
        self.anchor = anchor
        self.point = plt.Circle(init_pos, 0.1, color='blue', alpha=0.7, picker=True)
        self.ax.add_patch(self.point)
        self.std_dev = std_dev
        self.press = None  # Store press information
        self.donut_patch = None

        DraggablePoint.points.append(self)  # Add this point to the list of points
        self.connect()  # Set up event handlers
        self.update_donut(init_pos)
        DraggablePoint.update_intersection()
        
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlim(-10, 10)

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidkey = self.point.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        if event.key == 'ctrl+c':
            plt.close(self.point.figure)  # Close the plot

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.point.axes or DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center, event.xdata, event.ydata)
        DraggablePoint.lock = self

    def on_motion(self, event):
        'on motion we will move the point if the mouse is over us'
        if self.press is None or DraggablePoint.lock is not self: return
        if event.inaxes != self.point.axes: return
        center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        new_center = (center[0] + dx, center[1] + dy)
        self.point.center = new_center
        self.update_donut(new_center)
        self.update_gdop(self.anchor)
        DraggablePoint.update_intersection()  # Update intersection when any point is moved
        self.point.figure.canvas.draw_idle()

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self: return
        self.press = None
        DraggablePoint.lock = None
        self.point.figure.canvas.draw_idle()

    def update_donut(self, center):
        'Update the donut shape for the point'
        if self.donut_patch:
            self.donut_patch.remove()
        outer = Point(center).buffer(np.sqrt((center[0] - self.anchor[0])**2 + (center[1] - self.anchor[1])**2) + 3 * self.std_dev)
        inner = Point(center).buffer(max(np.sqrt((center[0] - self.anchor[0])**2 + (center[1] - self.anchor[1])**2) - 3 * self.std_dev, 0))
        self.donut = outer.difference(inner)
        self.donut_patch = DraggablePoint.plot_polygon_with_hole(self.ax, self.donut, facecolor='blue', alpha=0.5)

    @staticmethod
    def plot_polygon_with_hole(ax, polygon, **kwargs):
        if polygon.geom_type == 'Polygon':
            path = DraggablePoint.path_from_polygon(polygon)
            patch = PathPatch(path, **kwargs)
            ax.add_patch(patch)
            return patch
        
        elif polygon.geom_type == 'MultiPolygon':
            patch = []
            for part in polygon.geoms:
                path = DraggablePoint.path_from_polygon(part)
                patch.append(PathPatch(path, **kwargs))
            
            for part in patch:
                ax.add_patch(part)
            return patch


    @staticmethod
    def path_from_polygon(polygon):
        exterior_coords = polygon.exterior.coords.xy
        codes = [Path.MOVETO] + [Path.LINETO] * (len(exterior_coords[0]) - 1) + [Path.CLOSEPOLY]
        path_data = [(code, (x, y)) for code, x, y in zip(codes, *exterior_coords)]
        for interior in polygon.interiors:
            interior_coords = interior.coords.xy
            codes = [Path.MOVETO] + [Path.LINETO] * (len(interior_coords[0]) - 1) + [Path.CLOSEPOLY]
            path_data += [(code, (x, y)) for code, x, y in zip(codes, *interior_coords)]
        path = Path([data[1] for data in path_data], [data[0] for data in path_data])
        return path

    @staticmethod
    def update_intersection():
        'Update the intersection area of all donuts'
        if isinstance(DraggablePoint.intersection_patch, list):
            for patch in DraggablePoint.intersection_patch:
                patch.remove()
        elif DraggablePoint.intersection_patch:
            DraggablePoint.intersection_patch.remove()

        intersection = DraggablePoint.points[0].donut
        for point in DraggablePoint.points[1:]:
            intersection = intersection.intersection(point.donut)
        if not intersection.is_empty:
            DraggablePoint.intersection_patch = DraggablePoint.plot_polygon_with_hole(DraggablePoint.points[0].ax, intersection, facecolor='green', alpha=0.75)
            # Update intersection area display
            area = intersection.area
            if DraggablePoint.intersection_area_text:
                DraggablePoint.intersection_area_text.set_text(f'Intersection Area: {area:.2f}')
            else:
                DraggablePoint.intersection_area_text = DraggablePoint.points[0].ax.text(0.5, 0.01, f'Intersection Area: {area:.2f}', transform=DraggablePoint.points[0].ax.transAxes, ha='center')

    @staticmethod
    def update_gdop(anchor_pos):

        def calculate_gdop(positions):
            if len(positions) < 2:
                return None  # Not enough points to calculate GDOP
            A = []
            for pos in positions:
                x_i, y_i = pos
                x, y = anchor_pos
                R = np.sqrt((x_i - x)**2 + (y_i - y)**2)
                A.append([(x_i - x)/R, (y_i - y)/R, 1])
            A = np.array(A)
            try:
                inv_at_a = np.linalg.inv(A.T @ A)
                gdop = np.sqrt(np.trace(inv_at_a))
                return gdop
            except np.linalg.LinAlgError:
                return None  # Matrix is singular, cannot compute GDOP

        gdop = calculate_gdop(np.array([point.point.center for point in DraggablePoint.points]))
        if gdop is not None:
            if DraggablePoint.gdop_text:
                DraggablePoint.gdop_text.set_text(f'GDOP: {gdop:.2f}')
            else:
                DraggablePoint.gdop_text = DraggablePoint.points[0].ax.text(0.5, 0.95, f'GDOP: {gdop:.2f}', transform=DraggablePoint.points[0].ax.transAxes, ha='center', fontsize=12, color='red')
        else:
            print('Not enough points to calculate GDOP')
        
        


fig, ax = plt.subplots()

ax.grid(True)
plt.axis('equal')
plt.legend()

anchor_pos = (0, 0)
ax.plot(*anchor_pos, 'o', color='red', label='Anchor')

std_dev = 0.1

ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Add Point')

def add_point(event):
    new_point = DraggablePoint(ax, anchor_pos, ((np.random.random()-0.5)*10 , (np.random.random()-0.5)*10), std_dev)
    points.append(new_point)
    plt.draw()

button.on_clicked(add_point)

points = [
    DraggablePoint(ax, anchor_pos, (1, 1), std_dev),
]

plt.show()
