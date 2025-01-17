import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class BiasModel:
    """
    Represents the bias model for UWB measurements.
    Currently, only a constant bias and linear bias are supported.
    
    Attributes:
    -----------
    constant_bias : float
        Fixed offset added to all measurements (in meters).
    linear_bias : float
        Multiplicative factor applied to the true distance.
    """
    constant_bias: float = 0.0
    linear_bias: float = 1.0

    def apply(self, distance: float) -> float:
        """
        Apply the bias model to a distance measurement.
        
        Parameters:
        -----------
        distance : float
            True distance to be biased.
            
        Returns:
        --------
        float
            Biased distance measurement.
        """
        return self.constant_bias + self.linear_bias * distance

@dataclass
class NoiseModel:
    """
    Represents the noise model for UWB measurements.
    Currently, the noise model includes Gaussian noise and outliers which are generated with a certain probability and simply amplify the distance (after bias) by a certain perentage range.

    Attributes:
    -----------
    variance : float
        Variance of the Gaussian noise (in meters squared).
    outlier_probability : float
        Probability of generating an outlier [0,1].
    outlier_range : Tuple[float, float]
        Range for outlier multiplication factor (min, max).
    """

    variance: float = 0.1
    outlier_probability: float = 0.0
    outlier_range: Tuple[float, float] = (0.2, 0.3)

    def __post_init__(self):
        """Validate the noise model parameters after initialization."""
        if self.variance < 0:
            raise ValueError("Variance must be non-negative")
        if not 0 <= self.outlier_probability <= 1:
            raise ValueError("Outlier probability must be between 0 and 1")
        if self.outlier_range[0] > self.outlier_range[1]:
            raise ValueError("Outlier range minimum must be less than maximum")

    def apply(self, distance: float) -> float:
        """
        Apply noise model to a distance measurement.
        
        Parameters:
        -----------
        distance : float
            Distance to be corrupted with noise.
            
        Returns:
        --------
        float
            Noisy distance measurement.
        """
        if np.random.random() < self.outlier_probability:
            # Generate outlier
            factor = np.random.uniform(*self.outlier_range)
            return distance * (1 + factor)
        else:
            # Add Gaussian noise
            return distance + np.random.normal(0, np.sqrt(self.variance))

class Anchor:
    """
    Represents a UWB anchor in 3D space with realistic measurement simulation.
    
    This class simulates an Ultra-Wideband anchor that can measure distances
    to targets, including realistic error sources such as biases, noise,
    and outliers.

    Attributes:
    -----------
    anchor_id : str
        Unique identifier for the anchor.
    position : np.ndarray
        3D coordinates of the anchor [x, y, z].
    bias_model : BiasModel
        Model for systematic measurement biases.
    noise_model : NoiseModel
        Model for random measurement noise and outliers.
    """
    
    def __init__(self, 
                 anchor_id: str,
                 x: float, 
                 y: float, 
                 z: float,
                 bias_model: Optional[BiasModel] = None,
                 noise_model: Optional[NoiseModel] = None):
        """
        Initialize an UWB anchor.
        
        Parameters:
        -----------
        anchor_id : str
            Unique identifier for the anchor.
        x, y, z : float
            3D coordinates of the anchor position.
        bias_model : BiasModel, optional
            Model for systematic measurement biases.
        noise_model : NoiseModel, optional
            Model for random measurement noise and outliers.
        """
        self.anchor_id = str(anchor_id)
        self.position = np.array([float(x), float(y), float(z)])
        self.bias_model = bias_model or BiasModel()
        self.noise_model = noise_model or NoiseModel()

    def measure_distance(self, target_position: np.ndarray, include_errors: bool = True) -> float:
        """
        Measure the distance to a target position with optional error simulation.
        
        Parameters:
        -----------
        target_position : np.ndarray
            3D coordinates of the target [x, y, z].
        include_errors : bool
            If True, apply bias and noise models to the measurement.
            
        Returns:
        --------
        float
            Measured distance (with or without simulated errors).
        """
        if not isinstance(target_position, np.ndarray):
            target_position = np.array(target_position)
            
        if target_position.shape != (3,):
            raise ValueError("Target position must be a 3D point")

        # Calculate true distance
        true_distance = np.linalg.norm(self.position - target_position)
        
        if not include_errors:
            return true_distance
            
        # Apply bias model
        biased_distance = self.bias_model.apply(true_distance)
        
        # Apply noise model
        measured_distance = self.noise_model.apply(biased_distance)
        
        return measured_distance
    
    def update_bias_model(self,
                        constant_bias: float,
                        linear_bias: float) -> None:
        """
        Update the bias model parameters.
        
        Parameters:
        -----------
        constant_bias : float
            New constant bias offset.
        linear_bias : float
            New linear bias factor.
        """
        self.bias_model = BiasModel(
            constant_bias=constant_bias,
            linear_bias=linear_bias
        )

    def update_noise_model(self, 
                         outlier_probability: float,
                         outlier_range: Tuple[float, float]) -> None:
        """
        Update the noise model parameters.
        
        Parameters:
        -----------
        outlier_probability : float
            New probability of generating outliers [0,1].
        outlier_range : Tuple[float, float]
            New range for outlier multiplication factor (min, max).
        """
        self.noise_model = NoiseModel(
            variance=self.noise_model.variance,
            outlier_probability=outlier_probability,
            outlier_range=outlier_range
        )
        
    @property
    def ground_truth(self) -> Tuple[np.ndarray, BiasModel, NoiseModel]:
        """
        Get the ground truth parameters of the anchor.
        
        Returns:
        --------
        Tuple[np.ndarray, BiasModel, NoiseModel]
            Anchor position, bias model and noise model parameters.
        """
        return self.position, self.bias_model, self.noise_model

class UWBNetwork:
    """
    Represents a network of UWB anchors for multi-anchor measurements.
    
    This class manages multiple UWB anchors and provides methods for
    conducting measurements from all anchors to a target position.

    Attributes:
    -----------
    anchors : List[Anchor]
        List of UWB anchors in the network.
    """
    
    def __init__(self, anchors: List[Anchor]):
        """
        Initialize the UWB network.
        
        Parameters:
        -----------
        anchors : List[Anchor]
            List of UWB anchors to include in the network.
        """
        self.anchors = anchors

    def measure_distances(self, 
                        target_position: np.ndarray,
                        include_errors: bool = True) -> dict:
        """
        Measure distances from all anchors to a target position.
        
        Parameters:
        -----------
        target_position : np.ndarray
            3D coordinates of the target [x, y, z].
        include_errors : bool
            If True, apply bias and noise models to the measurements.
            
        Returns:
        --------
        dict
            Dictionary mapping anchor IDs to measured distances.
        """
        return {
            anchor.anchor_id: anchor.measure_distance(target_position, include_errors)
            for anchor in self.anchors
        }