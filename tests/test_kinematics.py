"""Tests for kinematic models."""
import numpy as np
import pytest

from nhmrs._nhmrs_utils.kinematics import (
    KinematicsModel,
    UnicycleKinematics,
    DifferentialDriveKinematics,
    AckermannKinematics,
)


class TestKinematicsModels:
    """Test suite for all kinematic models."""

    def test_unicycle_kinematics(self):
        """Test unicycle kinematic model."""
        model = UnicycleKinematics(dt=0.1)
        
        # Check dimensions
        assert model.state_dim == 3
        assert model.action_dim == 2
        
        # Check action bounds
        action_low, action_high = model.get_action_bounds()
        assert len(action_low) == 2
        assert len(action_high) == 2
        
        # Test step function
        state = np.array([0.0, 0.0, 0.0])
        action = np.array([1.0, 0.5])
        new_state = model.step(state, action)
        
        assert new_state.shape == (3,)
        assert not np.array_equal(new_state, state)
        
        # Test position and orientation extraction
        pos = model.get_position(new_state)
        assert pos.shape == (2,)
        
        ori = model.get_orientation(new_state)
        assert isinstance(ori, (float, np.floating))

    def test_differential_drive_kinematics(self):
        """Test differential drive kinematic model."""
        model = DifferentialDriveKinematics(dt=0.1)
        
        # Check dimensions
        assert model.state_dim == 3
        assert model.action_dim == 2
        
        # Check action bounds
        action_low, action_high = model.get_action_bounds()
        assert len(action_low) == 2
        assert len(action_high) == 2
        
        # Test step function
        state = np.array([0.0, 0.0, 0.0])
        action = np.array([1.0, 1.0])  # Equal wheel speeds = straight line
        new_state = model.step(state, action)
        
        assert new_state.shape == (3,)
        # With equal wheel speeds, orientation should not change significantly
        assert abs(new_state[2] - state[2]) < 0.1

    def test_ackermann_kinematics(self):
        """Test Ackermann steering kinematic model."""
        model = AckermannKinematics(dt=0.1)
        
        # Check dimensions
        assert model.state_dim == 3
        assert model.action_dim == 2
        
        # Check action bounds
        action_low, action_high = model.get_action_bounds()
        assert len(action_low) == 2
        assert len(action_high) == 2
        
        # Test step function
        state = np.array([0.0, 0.0, 0.0])
        action = np.array([1.0, 0.0])  # Zero steering = straight line
        new_state = model.step(state, action)
        
        assert new_state.shape == (3,)
        # With zero steering, orientation should not change
        assert abs(new_state[2] - state[2]) < 1e-6

    def test_angle_normalization(self):
        """Test that angles are properly normalized to [-π, π]."""
        model = UnicycleKinematics(dt=0.1)
        
        # Start with large angle
        state = np.array([0.0, 0.0, 4 * np.pi])
        action = np.array([0.0, 0.0])
        new_state = model.step(state, action)
        
        # Angle should be normalized
        assert -np.pi <= new_state[2] <= np.pi

    def test_kinematics_consistency(self):
        """Test that all models follow the same interface."""
        models = [
            UnicycleKinematics(dt=0.1),
            DifferentialDriveKinematics(dt=0.1),
            AckermannKinematics(dt=0.1),
        ]
        
        for model in models:
            # All should have same state dimension
            assert model.state_dim == 3
            
            # All should have action dimension = 2
            assert model.action_dim == 2
            
            # All should implement required methods
            assert hasattr(model, 'step')
            assert hasattr(model, 'get_position')
            assert hasattr(model, 'get_orientation')
            assert hasattr(model, 'get_action_bounds')


if __name__ == "__main__":
    import os
    import sys
    
    # Unset ROS environment variables to avoid plugin conflicts
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('ROS_DISTRO', None)
    os.environ.pop('AMENT_PREFIX_PATH', None)
    
    # Re-exec with clean environment
    if 'NHMRS_TEST_REEXEC' not in os.environ:
        os.environ['NHMRS_TEST_REEXEC'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    # Now run pytest
    pytest.main([__file__, "-v"])
