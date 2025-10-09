import numpy as np
from descents import BaseDescent, AnalyticSolutionOptimizer
from dataclasses import dataclass
from enum import auto, Enum
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
    
    @abstractmethod
    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        ...


class MSELoss(LossFunction):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        raise NotImplementedError()
        # TODO: implement

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        raise NotImplementedError()
        # TODO: implement

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        #  Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        return self.analytic_solution_func(X, y)
    
    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        raise NotImplementedError()
    
    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        raise NotImplementedError()


class L1Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        self.core_loss = core_loss
        self.rate = rate
    

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:

        core_part = self.core_loss.gradient(X, y, w)

        penalty_part = ...

        raise NotImplementedError()
        # TODO: implement



class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | AnalyticSolutionOptimizer] = None, # fix it, perhaps refactoring of the responsibility is needed
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        # TODO: реализовать функцию предсказания в линейной регрессии
        raise NotImplementedError("predict function is not implemented")

    def compute_gradients(self) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        """
        raise NotImplementedError("Gradient caclucation is not implemented")


    def compute_loss(self) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        """
        raise NotImplementedError("Loss calculation is not implemented")


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        # TODO: реализовать обучение модели
        self.X_train, self.y_train = X, y

        raise NotImplementedError("Linear Regression training is not implemented")
