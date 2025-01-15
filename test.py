import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import pandas as pd


anchor_1_gt = [-2.486, -1.746, 0.252]
anchor_2_gt = [3.045, -1.366, 0.900]
anchor_3_gt = [-2.795, 1.065, 2.005]
anchor_4_gt = [2.859, 2.447, 2.146]


calculated_position_1 = [-2.311557875876655,-1.865119492663806,0.33172133027699147]
calculated_position_2 = [7.207018642291071,-1.5073402457661933,0.566961083157065]
calculated_position_3 = [-2.7301439512596835,0.5357063324674097,2.343093086745037]
calculated_position_4 = [1.6176539611172795,3.75427714950098,2.294142285846655]

error1 = np.linalg.norm(np.array(anchor_1_gt) - np.array(calculated_position_1))
error2 = np.linalg.norm(np.array(anchor_2_gt) - np.array(calculated_position_2))
error3 = np.linalg.norm(np.array(anchor_3_gt) - np.array(calculated_position_3))
error4 = np.linalg.norm(np.array(anchor_4_gt) - np.array(calculated_position_4))

for i in range(1,5):
    print(f"Error {i}: {eval(f'error{i}')}")




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([anchor_1_gt[0], calculated_position_1[0]], [anchor_1_gt[1], calculated_position_1[1]], [anchor_1_gt[2], calculated_position_1[2]], 'r-')
ax.plot([anchor_2_gt[0], calculated_position_2[0]], [anchor_2_gt[1], calculated_position_2[1]], [anchor_2_gt[2], calculated_position_2[2]], 'g-')
ax.plot([anchor_3_gt[0], calculated_position_3[0]], [anchor_3_gt[1], calculated_position_3[1]], [anchor_3_gt[2], calculated_position_3[2]], 'b-')
ax.plot([anchor_4_gt[0], calculated_position_4[0]], [anchor_4_gt[1], calculated_position_4[1]], [anchor_4_gt[2], calculated_position_4[2]], 'y-')

ax.plot([calculated_position_1[0]], [calculated_position_1[1]], [calculated_position_1[2]], 'ro')
ax.plot([calculated_position_2[0]], [calculated_position_2[1]], [calculated_position_2[2]], 'go')
ax.plot([calculated_position_3[0]], [calculated_position_3[1]], [calculated_position_3[2]], 'bo')
ax.plot([calculated_position_4[0]], [calculated_position_4[1]], [calculated_position_4[2]], 'yo')

plt.show()



# Step 1: Load the CSV data
# Make sure to replace 'your_file.csv' with the actual filename or path
csv_file = 'csv_file.csv'
data = pd.read_csv(csv_file)


anchor_to_show = 1
# Step 2: Calculate the Euclidean norm (L2 norm) of the (x, y, z) coordinates
data['norm'] = np.linalg.norm(np.array(data[['x', 'y', 'z']].values) - np.array(anchor_1_gt), axis=1)
data['norm_estimated'] = np.linalg.norm(np.array(data[['x', 'y', 'z']].values) - np.array(calculated_position_1), axis=1)
# Step 3: Create a scatter plot of 'distance' vs 'norm'
plt.figure(figsize=(8, 6))
plt.plot(data['distance'],color='blue', label='Distance')
plt.plot(data['norm'], color='red', label='Euclidean Norm (x, y, z)')
plt.plot(data['norm_estimated'], color='green', label='Euclidean Norm (x, y, z) estimated')

# Step 4: Customize the plot
plt.title('Scatter Plot: Distance vs L2 Norm of (x, y, z)')
plt.xlabel('Distance')
plt.ylabel('Euclidean Norm (x, y, z)')
plt.grid(True)
plt.legend()

# Step 5: Show the plot
plt.show()


