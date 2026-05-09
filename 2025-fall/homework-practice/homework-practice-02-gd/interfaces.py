import numpy as np 
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC

class LossFunction(ABC):

    @abstractmethod
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение функции потерь на данных X,y для весов w
        """
        ...

    @abstractmethod
    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент функции потерь в точке w
        """
        ...
        
    

class LossFunctionClosedFormMixin(ABC):

    @abstractmethod
    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        ...



class LinearRegressionInterface(ABC):

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        ...
    
    @abstractmethod
    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        ...

    @abstractmethod
    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        ...
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        ...


class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class AbstractOprimizer(ABC):
    def set_model(self, model: LinearRegressionInterface) -> None:
        self.model = model
    
    @abstractmethod
    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм обучения.
        """