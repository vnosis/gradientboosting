import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
# Use of sklearn.tree for the *Weak Learner* is allowed
from sklearn.tree import DecisionTreeRegressor 
from enum import Enum

class LossType(Enum):
    L2 = 1
    L1 = 2
    HUBER = 3
    
class GBRegressor(BaseEstimator, RegressorMixin):
    """
    Custom Gradient Boosting Regressor.
    Uses L2, L1 and Huber Losses
    Weak learner is constrained to a shallow Regression Tree (Decision Stump).
    """

    # --------------------------------------------------------------------------------
    # Task 1: Weak Learner Initialization. TODO: Change the necessary parameters to 
    # initialize the decision stump
    # --------------------------------------------------------------------------------
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=None, loss_type=LossType.L2, delta=None):
        """
        Initializes the Gradient Boosting Regressor.
        
        Parameters:
        n_estimators (int): The number of boosting stages (trees) to perform.
        learning_rate (float): Shrinks the contribution of each tree.
        max_depth (int): The maximum depth of the base regression estimator (weak learner).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators_ = []  # Stores the trained base learners
        self.initial_prediction_ = None # Stores the initial prediction
        self.loss_type = loss_type
        self.delta = delta #Huber threshold

    # --------------------------------------------------------------------------------
    # TASK 1: Weak Learner Implementation 
    # --------------------------------------------------------------------------------
    def _make_weak_learner(self):
        """
        Creates and returns an instance of the weak learner, i.e., the Decision Stump.
        TODO: Create and return the stump. DOCUMENT THE CODE
        """

        # Your code
        stump = DecisionTreeRegressor(max_depth=self.max_depth if self.max_depth else 1)
        return stump
        


    # --------------------------------------------------------------------------------
    # TASK 3: The negative gradient for L2, L1 and Huber loss
    # --------------------------------------------------------------------------------

    def _get_negative_gradient(self, y, prediction):
        """
        Computes the negative gradient according to the loss_type configuration parameter selected. 
        For LossType.L2, that is the residuals.
        For LossType.L1, ... COMPLETE DOCUMENTATION
        For LossType.Huber, YOU COMPLETE

        Parameters:
        y : True label
        prediction: current prediction

        Returns:
        ngrad : the negative gradient
        """

        #TODO Your code

        if self.loss_type == LossType.L2:
            #y - pred after taking the derivative with respect to the output of the MSE function
            return y - prediction

        elif self.loss_type == LossType.L1:
            #This returns 1 if y > pred, -1 if y < pred, and 0 if equal.
            return np.sign(y - prediction)

        elif self.loss_type == LossType.HUBER:
            diff = y - prediction
            abs_diff = np.abs(diff)
            
            # Correct Huber Gradient logic:
            ngrad = np.where(abs_diff <= self.delta,diff, self.delta * np.sign(diff))
            return ngrad



    # --------------------------------------------------------------------------------
    # TASK 4: The Core Boosting Algorithm
    # --------------------------------------------------------------------------------        
    def fit(self, X, y):
        """
        Builds the additive model in a forward stage-wise fashion.

        Parameters:
        X (np.ndarray): The training input samples.
        y (np.ndarray): The target values.
        """
        # Convert X and y to NumPy arrays for consistent handling
        X, y = np.array(X), np.array(y)
        
        # 1. Initialize the model with a constant value (e.g. mean of y)
        self.initial_prediction_ = np.mean(y)

        self.history_ = []
        
        # Initialize the current ensemble prediction F(x) to F_0(x)
        current_prediction = np.full_like(y, self.initial_prediction_, dtype=float)
        
        for m in range(self.n_estimators):
            
            # 2. Compute the negative gradient (pseudo-residuals)
            residuals = self._get_negative_gradient(y, current_prediction)
            
            # 3. Fit the base learner (h_m) to the pseudo-residuals
            #h_m = None #TODO
            #TODO           
            h_m = self._make_weak_learner()
            h_m.fit(X, residuals)
            
            # Store the weak learner
            self.estimators_.append(h_m)
            
            # 4. Update the ensemble model
            #h_m_prediction = None # TODO Get prediction from the new tree on X
            # Fmx = Fm-1x + learning_rate * h_mx
            h_m_prediction = h_m.predict(X)
            
            #current_prediction = None # TODO Update current_prediction using learning_rate
            current_prediction += self.learning_rate * h_m_prediction

            # saving MSE to history
            mse_at_stage = np.mean((y - current_prediction)**2)
            self.history_.append(mse_at_stage)
            
        return self


    # --------------------------------------------------------------------------------
    # TASK 4: Make Predictions
    # --------------------------------------------------------------------------------
    def predict(self, X):
        """
        Predicts target values for new data points.

        Parameters:
        X (np.ndarray): The input samples.

        Returns:
        ensemble_prediction (np.ndarray): The predicted target values.
        """

        X = np.array(X)
        
        if not self.estimators_:
            return np.full(X.shape[0], self.initial_prediction_)
            
        ensemble_prediction = np.full(X.shape[0], self.initial_prediction_, dtype=float)
        
        for h_m in self.estimators_:
            ensemble_prediction += self.learning_rate * h_m.predict(X)
        
        return ensemble_prediction

    def _get_loss(self, y, prediction):
        """Calculates the actual loss value (Task 2 formulas)."""
        diff = y - prediction
        if self.loss_type == LossType.L2:
            return 0.5 * diff**2
        elif self.loss_type == LossType.L1:
            return np.abs(diff)
        elif self.loss_type == LossType.HUBER:
            return np.where(np.abs(diff) <= self.delta, 
                            0.5 * diff**2, 
                            self.delta * (np.abs(diff) - 0.5 * self.delta)) 

    ###USED AI FOR PLOTTING
    def plot_loss_analysis(self, y_true, y_pred):
            """
            Comprehensive plot containing:
            1. Residuals vs Loss Curve
            2. Residuals vs Gradient Steps
            3. Actual vs Predicted Scatter Plot (Perfect Fit)
            """
            import matplotlib.pyplot as plt
            from sklearn.metrics import mean_squared_error
            
            # 1. Calculate residuals and coordinate data
            actual_residuals = y_true - y_pred
            actual_loss = self._get_loss(y_true, y_pred)
            actual_grad = self._get_negative_gradient(y_true, y_pred) # Reuses Task 3 logic
            
            # Create theoretical curves
            limit = max(np.abs(actual_residuals)) * 1.1
            z_range = np.linspace(-limit, limit, 500)
            curve_loss = self._get_loss(z_range, 0) 
            curve_grad = self._get_negative_gradient(z_range, 0)

            # 2. Setup 3-panel figure
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            
            # --- Subplot 1: Loss Curve ---
            ax1.plot(z_range, curve_loss, color='teal', alpha=0.3, label="Theoretical Loss")
            ax1.scatter(actual_residuals, actual_loss, color='blue', s=15, alpha=0.5)
            ax1.set_title(f"Data on {self.loss_type.name} Loss Curve")
            ax1.set_xlabel("Residual (y - pred)")
            ax1.set_ylabel("Loss Value")

            # --- Subplot 2: Gradient Curve ---
            ax2.plot(z_range, curve_grad, color='crimson', alpha=0.3, label="Theoretical Gradient")
            ax2.scatter(actual_residuals, actual_grad, color='black', s=15, alpha=0.5)
            ax2.set_title("Actual Gradients (Step Directions)")
            ax2.set_xlabel("Residual (y - pred)")
            ax2.set_ylabel("Gradient Value")

            # --- Subplot 3: Actual vs Predicted (Your Original Plot) ---
            ax3.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predictions')
            line_coords = [y_true.min(), y_true.max()]
            ax3.plot(line_coords, line_coords, color='red', linestyle='--', label='Perfect Fit')
            
            mse = mean_squared_error(y_true, y_pred)
            ax3.set_title(f"Actual vs Predicted (MSE: {mse:.2f})")
            ax3.set_xlabel("Actual Values")
            ax3.set_ylabel("Predicted Values")
            ax3.legend()

            for ax in [ax1, ax2, ax3]:
                ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.show()