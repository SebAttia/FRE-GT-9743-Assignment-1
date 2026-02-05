import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional


class InterpMethod(Enum):
    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):
    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrpolation_method: ExtrapMethod) -> None:
        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrpolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float):
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float):
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float):
        pass
    
    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_
    
    @property
    def values(self) -> np.ndarray:
        return self.values_
    
    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()
    
    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    def __init__(self, axis1: np.ndarray, values: np.ndarray, extrpolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS, extrpolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

        if len(axis1) != len(values):
            print("Length of x input is different from y")
        else:
            # Sort arrays based on x
            idx = np.argsort(axis1)
            self.axis1_ = np.array(axis1)[idx]
            self.values_ = np.array(values)[idx]
            self.length_ = len(self.axis1_)
                

    def interpolate(self, x_star: float) -> float:
        bins = self.axis1_
        # Get indices of the bin
        bin_idx = np.digitize(x_star, bins, right=False)
        # To take into account values that falls outside of the left bound of the interval
        if bin_idx == 0:
            return self.values_[0]
        # To take into account values that falls outside of the right bound of the interval
        elif bin_idx >= len(bins):
            return self.values_[-1]
        # To take into account what falls inside the intervals
        else:
            return self.values_[bin_idx - 1]

    def gradient_wrt_ordinate(self, x_star: float) -> np.ndarray:
        # Gradient is going to be a vector of zeros with a one at location of gradient of x
        # locate position using the interpolate function
        bins = self.axis1_
        bin_idx = np.digitize(x_star, bins, right=False)
        if bin_idx == 0:
            loc = 0
        elif bin_idx >= len(bins):
            loc = len(self.axis1_) - 1
        else:
            loc = bin_idx - 1
        # generate a vector of zeros
        v = np.zeros(self.length_)
        # At index location, input 1 to get the gradient
        v[loc] = 1
        return v

    def integrate(self, l: float, u: float) -> float:
        # Compute integral using np.trapz did not work
        x_right = np.append(self.axis1_[1:], np.inf)
        lengths = np.maximum(0, np.minimum(u, x_right) - np.maximum(l, self.axis1_))
        return np.sum(self.values_ * lengths)

    def gradient_of_integrated_value_wrt_ordinate(self, l: float, u: float) -> np.ndarray:
        x_r = np.append(self.axis1_[1:], np.inf)
        lengths = np.maximum(0, np.minimum(u, x_r) - np.maximum(l, self.axis1_))
        return lengths


class InterpolatorFactory:
    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrpolation_method: ExtrapMethod):
        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrpolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')


# Required test wrapper functions
def qfCreate1DInterpolator(axis1, values, interp_method, extrap_method):
    return InterpolatorFactory.create_1d_interpolator(
        axis1, values,
        InterpMethod.from_string(interp_method),
        ExtrapMethod.from_string(extrap_method)
    )


def qfInterpolate1D(x, interp_1d):
    return interp_1d.interpolate(x)


def qfInterpolate1DIntegral(start_x, end_x, interp_1d):
    return interp_1d.integrate(start_x, end_x)


def qfInterpolate1DGrad(x, interp_1d):
    return interp_1d.gradient_wrt_ordinate(x)


def qfInterpolate1DIntegralGrad(start_x, end_x, interp_1d):
    return interp_1d.gradient_of_integrated_value_wrt_ordinate(start_x, end_x)
