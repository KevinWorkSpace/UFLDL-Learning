# 向量化（Vectorization）  
##  

对于如房价数据的小数据量任务，通常使用线性回归，因为代码不需要执行地非常快。尽管您在练习 1A 和 1B 里是建议使用 for 循环的，但对于较大规模的问题， for 循环的执行效率就比较低了。这是因为在 MATLAB 里，按顺序执行整个样本的循环是缓慢的。为了避免（使用） for 循环，想要重写（这部分）代码，使其能尽可能地在 MATLAB 里高效地执行向量或矩阵操作（这点同样适用于其他语言，包括 Python，C/C++ —— 要尽可能地重用已经优化过的操作，这里特指使用向量计算库来优化计算效率）。  

下面是一些在 MATLAB 里各种向量化的操作方法。  

## 案例：多矩阵-向量相乘（Example: Many matrix-vector products）  

经常一次计算多个矩阵或矢量的乘积（矩阵乘法）。例如，当对数据集（其中，参数 $\theta$ 可能是一个二维矩阵或矢量）中的每个样本计算 $\theta^{\top}x^{(i)}$。要形成一个包含整个数据集样本的矩阵 $X$ ，可以将每个输入样本的元素或者向量（按照行或列） $x^{(i)}$ 连接起来（更形象地表达是“拼起来”）。这里，每一列是一个样本：  

$$
X = \left[\begin{array}{cccc}
  | & |  &  | & | \\
  x^{(1)} & x^{(2)} & \cdots & x^{(m)}\\
    | & |  &  | & |\end{array}\right]
$$  

因此，对于所有的样本 $$x^{(i)}$$ ，可以一次矩阵运算的形式完成所有样本的 $y^{(i)} = W x^{(i)}$ 计算：  

$$
\left[\begin{array}{cccc}
| & |  &  | & | \\
y^{(1)} & y^{(2)} & \cdots & y^{(m)}\\
| & |  &  | & |\end{array}\right] = Y = W X
$$  

所以，当执行线性回归（Linear Regression）时，可以通过计算 $\theta^{\top}X$ 求得所有的 $y^{(i)} = \theta^{\top}X^{(i)}$ ，以避免 for 循环对所有样本的遍历。  


## 案例：标准化向量（Example: normalizing many vectors）  

假设有前文说到的由众多向量 $x^{(i)}$ 连接形成的矩阵 $X$，同时要对所有的 $x^{(i)}$ 计算 $y^{(i)} = x^{(i)}/||x^{(i)}||_2$， 可以用几个 MATLAB 的矩阵操作来完成。  

```
  X_norm = sqrt( sum(X.^2,1) );
  Y = bsxfun(@rdivide, X, X_norm);
```  


第一行代码，先对 $X$ 中的所有元素做平方操作，所有元素再按列相加得到行向量，最终对行向量中的每个元素做开平方根操作。最后得到的是一个 $1$ 行 $m$ 列，包含了 $||x(i)||_{2}$ 元素的行向量。<font color=red>`bsxfun`</font>函数的作用可以看成是对变量 <font color=red>`Xnorm`</font> 的扩展或者复制，便会得到与矩阵 $X$ 维度相同的矩阵，然后对该矩阵中逐个元素应用二元操作函数（匿名函数 <font color=red>`@rdivide`</font> 对同维矩阵的同位置的所有元素，实现右除操作）。上述例子中，实现了用二元操作函数对每个元素 $X_{ji} = x^{(i)}_{j}$ 除以在向量 $X\text{norm}$ 中与其列位置相同的元素，最后得到 $Y_{ji} = X_{ji} / {X\text{norm}}_i = x_j^{(i)}/||x^{(i)}||_2$。<font color=red>`bsxfun`</font> 可以与几乎所有的二元操作函数使用（例如，@plus，@ge或@eq），更多详情可以查看 <font color=red>`bsxfun`</font> 的 MATLAB 文档。  

## 案例：梯度计算的矩阵乘法（Example: matrix multiplication in gradient computations）  
在线性回归的梯度计算中，其形式可概括为：  

$$
\frac{\partial J(\theta; X,y)}{\partial \theta_j} = \sum_i x_j^{(i)} (\hat{y}^{(i)} - y^{(i)}).
$$  

当有通过单个索引（公式中的 $i$ ）与其它几个固定索引（公式中的 $j$ ）的求和操作时，经常将这个计算改写成矩阵乘法 $[A B]_{jk} = \sum_i A_{ji} B_{ik}$ 的形式。即，如果 $y$ 和 $\hat{y}$ 是列向量（有 $y_i \equiv y^{(i)}$），那么可将上面这样的求和模式重新写成下面这样：  

$$
\frac{\partial J(\theta; X,y)}{\partial \theta_j} = \sum_i X_{ji} (\hat{y}_i - y_i) = [X (\hat{y} - y)]_j.
$$  

因此，由于矩阵的整体计算思想，不需要逐个 $j$ 索引依次计算，实际只需计算 $X (\hat{y} - y)$ 就可以了。在 MATLAB 中的实现如下：  

```
% X(j,i) = j'th coordinate of i'th example.
% y(i) = i'th value to be predicted;  y is a column vector.
% theta = vector of parameters

y_hat = theta'*X; % so y_hat(i) = theta' * X(:,i).  Note that y_hat is a *row-vector*.
g = X*(y_hat' - y);
```  

##  进一步优化练习 1A 和 1B（Exercise 1A and 1B Redux）  

返回您练习的 1A 和 1B 代码中，在 <font color=red>`ex1a_linreg.m`</font> 和 <font color=red>`ex1b_logreg.m`</font> 文件中，您将发现调用 <font color=red>`minFunc`</font> 时分别使用的是文件 <font color=red>`linear_regression_vec.m`</font> 和 <font color=red>`logistic_regression_vec.m`</font> ，但却是被注释掉的，而不是用 <font color=red>`linear_regression.m`</font> 和 <font color=red>`logistic_regression.m`</font> 文件。在本次练习中，请您将 <font color=red>`linear_regression_vec.m`</font> 和 <font color=red>`logistic_regression_vec.m`</font> 里的代码以（前文所讲过的）向量化的方式实现并补充完整。将 <font color=red>`ex1a_linreg.m`</font> 和 <font color=red>`ex1b_logreg.m`</font> 文件中的注释取消掉，并比较二者代码的运行时间，检验（现在的代码）是否和先前原本的代码得到的结果是一样的。
