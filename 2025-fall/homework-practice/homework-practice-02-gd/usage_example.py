from descents import VanillaGradientDescent, AnalyticSolutionOptimizer
from linear_regression import LinearRegression, MSELoss


X = ...

y = ...

# Использование итерационных процесов

vanilla_optimizer = VanillaGradientDescent()

linreg = LinearRegression(vanilla_optimizer, loss_function=MSELoss())

linreg.fit(X, y)

"""
linreg.fit(X, y) ->  linreg.optimizer.optimize()
optimize контролирует процесс спуска и управляет итерационным процессом и остановом.

linreg.optimizer.optimize() -> linreg.optimizer._step()
на каждом шаге итерационного процесса

linreg.optimizer._step() -> linreg.optimizer._update_weights()

linreg.optimizer._update_weights() -> linreg.optimizer.model.compute_gradients()
linreg.optimizer._update_weights() -> linreg.optimizer.lr_schedule.get_lr(self.iteration)
здесь происходит обновление весов в модели и подъем обратно до linreg.optimizer._step

после останова итерационного процесса происходит подъем в fit и выход в __main__
"""


# Использование аналитического решения


closed_form_optimizer = AnalyticSolutionOptimizer()

loss = MSELoss(analytic_solution_func=MSELoss._svd_analytic_solution)

linreg_analyt = LinearRegression(closed_form_optimizer, 
                          loss_function=loss)

linreg_analyt.fit(X, y)


"""
linreg_analyt.fit(X, y) ->  linreg_analyt.optimizer.optimize()
optimize дергает интерфейс аналитического решения, которое принадлежит функции потерь

linreg_analyt.optimizer.optimize() -> loss.analytic_solution(X, y)
диспатч-интерфейс дергает конкретную функцию, определенную на этапе инициализации

loss.analytic_solution(X, y) -> MSELoss._svd_analytic_solution(X, y)

вовзрат имплементирующей функции проходит через интерфейс 
и присваивается весам модели в linreg_analyt.optimizer.optimize()
"""



# Добавление регуляризации


"""
Регуляризация имплеменируется как класс с интерфейсом LossFunction, 
который принимает на вход loss, который мы хотим регуляризовать, 
и для имплементации интерфейса внутри методов дергает методы исходного лосса, 
и прибавляет к ним свой численный кусочек, что возможно благодаря линейности градиента.

Для внешнего мира она ведёт себя в точности как любой другой итеративный оптимизатор.

Аналитическое решение, к сожалению, ей нужно задавать явно, поскольку оно линейностью не обладает  
"""