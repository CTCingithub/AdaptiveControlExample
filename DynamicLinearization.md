<!--
 * @Author: CTC 2801320287@qq.com
 * @Date: 2024-05-20 15:14:31
 * @LastEditors: CTC 2801320287@qq.com
 * @LastEditTime: 2024-11-10 19:46:41
 * @Description: 
 * 
 * Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
-->
# Linearization of Dynamics

## 1. Linearization of Arbitrary Link's Energy

For $i$th link, its kinematic energy follows:

$$
\begin{aligned}
    T_{i} &=&& \frac{1}{2} \int \vec{v}^{\top} \vec{v} dm \\
    &=&& \frac{1}{2} \int \left( \vec{v}_{i} + [\vec{\omega}_{i}] \vec{r} \right)^{\top} \left( \vec{v}_{i} + [\vec{\omega}_{i}] \vec{r} \right) dm \\
    &=&& \frac{1}{2} \int \vec{v}_{i}^{\top} \vec{v}_{i} dm + \int \vec{v}_{i}^{\top} [\vec{\omega}_{i}] \vec{r} dm + \frac{1}{2} \int \vec{r}^{\top} [\vec{\omega}_{i}]^{\top} [\vec{\omega}_{i}] \vec{r} dm \\
    &=&& \frac{1}{2} \vec{v}_{i}^\top \vec{v}_{i} \int dm + \vec{v}_{i}^{\top} [\vec{\omega}_{i}] \int dm \cdot \vec{r}_{C_{i}} \\
    &&& + \frac{1}{2} \int \left( y^{2} + z^{2} \right) dm \cdot \left( \omega_{x}^{i} \right)^{2} + \frac{1}{2} \int \left( x^{2} + z^{2} \right) dm \cdot \left( \omega_{y}^{i} \right)^{2} \\
    &&& + \frac{1}{2} \int \left( x^{2} + y^{2} \right) dm \cdot \left( \omega_{z}^{i} \right)^{2} - \int xy dm \left( \omega_{x}^{i} \omega_{y}^{i} \right) \\
    &&& - \int xz dm \left( \omega_{x}^{i} \omega_{z}^{i} \right) - \int yz dm \left( \omega_{y}^{i} \omega_{z}^{i} \right) \\
    &=&& \frac{1}{2} \vec{v}_{i}^\top \vec{v}_{i} m_{i} + \vec{v}_{i}^{\top} [\vec{\omega}_{i}] m_{i} \vec{r}_{C_{i}} + \frac{1}{2} \vec{\omega}_{i}^{\top} \begin{bmatrix}
        xx_{i} & xy_{i} & xz_{i} \\
        xy_{i} & yy_{i} & yz_{i} \\
        xz_{i} & yz_{i} & zz_{i}
    \end{bmatrix} \vec{\omega}_{i} \\
    &=&& \frac{1}{2} \vec{\omega}_{i}^{\top} J_{C_{i}}^{\text{Former Joint}} \vec{\omega}_{i} + \vec{v}_{i}^{\top} [\vec{\omega}_{i}] m_{i} \vec{r}_{C_{i}} + \frac{1}{2} \vec{v}_{i}^\top \vec{v}_{i} m_{i},
\end{aligned}
$$

where $
m_{i} \vec{r}_{ci} = R_{i}^{0} \begin{pmatrix}
    m x_{i} \\ m y_{i} \\ m z_{i}
\end{pmatrix}$, elements in $J_{C_{i}}^{\text{Former Joint}} = \begin{bmatrix}
    xx_{i} & xy_{i} & xz_{i} \\
    xy_{i} & yy_{i} & yz_{i} \\
    xz_{i} & yz_{i} & zz_{i}
\end{bmatrix}$ are moment of inertia intergrated from former joint along the link, which are different from those in $J_{C_{i}}^{\text{CoM}}$ matrixes.

We have

$$
\begin{aligned}
    xx_{i} &=&& \iiint \left( y^{2} + z^{2} \right) \rho dx dy dz \\
    &=&& \iiint \left[ (y - y_{C_{i}})^{2} + (z - z_{C_{i}})^{2} \right] \rho dx dy dz \\
    &&& + 2 \iiint \left[ y_{C_{i}} y + z_{C_{i}} z \right] \rho dx dy dz \\
    &&& - \iiint \left[ \left( y_{C_{i}} \right)^{2} + \left( z_{C_{i}} \right)^{2} \right] \rho dx dy dz \\
    &=&& J_{xx_{i}} + m_{i} \left[ \left( y_{C_{i}} \right)^{2} + \left( z_{C_{i}} \right)^{2} \right], \\
    xy_{i} &=&& - \iiint xy \rho dx dy dz \\
    &=&& - \iiint \left[ \left( x - x_{C_{i}} \right) \left( y - y_{C_{i}} \right) \right] \rho dx dy dz \\
    &&& - \iiint \left( x_{C_{i}} y + y_{C_{i}} x \right) \rho dx dy dz \\
    &&& + \iiint x_{C_{i}} y_{C_{i}} \rho dx dy dz \\
    &=&& - \left( J_{xy_{i}} + m_{i} x_{C_{i}} y_{C_{i}} \right).
\end{aligned}
$$

Similarly, it's easy to verify that

$$
\begin{gathered}
    yy_{i} = J_{yy_{i}} + m_{i} \left[ \left( x_{C_{i}} \right)^{2} + \left( z_{C_{i}} \right)^{2} \right], \qquad
    zz_{i} = J_{zz_{i}} + m_{i} \left[ \left( x_{C_{i}} \right)^{2} + \left( y_{C_{i}} \right)^{2} \right], \\
    xz_{i} = - \left( J_{xz_{i}} + m_{i} x_{C_{i}} z_{C_{i}} \right), \qquad
    yz_{i} = - \left( J_{yz_{i}} + m_{i} y_{C_{i}} z_{C_{i}} \right).
\end{gathered}
$$

Denote a vector

$$
\begin{aligned}
    \vec{p}^{i} & \triangleq \left[ xx_{i}, xy_{i}, xz_{i}, yy_{i}, yz_{i}, zz_{i}, mx_{i}, my_{i}, mz_{i}, m_{i} \right]^{\top} \\
    &\triangleq \left[ p_{1}^{i}, p_{2}^{i}, \cdots, p_{10}^{i} \right]^{\top},
\end{aligned}
$$

to collect inertial parameters. It's easy to verify that

$$
\begin{aligned}
    J_{C_{i}}^{\text{Former Joint}} \vec{\omega}_{i}
    &= \begin{bmatrix}
        xx_{i} & xy_{i} & xz_{i} \\
        xy_{i} & yy_{i} & yz_{i} \\
        xz_{i} & yz_{i} & zz_{i}
    \end{bmatrix} \vec{\omega}_{i} \\
    &= \begin{bmatrix}
        \omega_{x}^{i} & \omega_{y}^{i} & \omega_{z}^{i} & 0 & 0 & 0 \\
        0 & \omega_{x}^{i} & 0 & \omega_{y}^{i} & \omega_{z}^{i} & 0 \\
        0 & 0 & \omega_{x}^{i} & 0 & \omega_{y}^{i} & \omega_{z}^{i}
    \end{bmatrix} \begin{pmatrix}
        xx_{i} \\ xy_{i} \\ xz_{i} \\ yy_{i} \\ yz_{i} \\ zz_{i}
    \end{pmatrix} \\
    & \triangleq K (\vec{\omega}_{i}) \begin{pmatrix}
        xx_{i} \\ xy_{i} \\ xz_{i} \\ yy_{i} \\ yz_{i} \\ zz_{i}
    \end{pmatrix},
\end{aligned}
$$

Then the kinematic energy can be rewritten as

$$
\begin{aligned}
    T_{i} &= \frac{1}{2} \vec{\omega}_{i}^{\top} J_{C_{i}}^{\text{Former Joint}} \vec{\omega}_{i} + \vec{v}_{i}^{\top} [\vec{\omega}_{i}] m_{i} \vec{r}_{C_{i}} + \frac{1}{2} \vec{v}_{i}^\top \vec{v}_{i} m_{i} \\
    &= \begin{bmatrix}
        \frac{1}{2} \vec{\omega}_{i}^{\top} K (\vec{\omega}_{i}), \vec{v}_{i}^{\top} [\vec{\omega}_{i}] R_{i}^{0}, \frac{1}{2} \vec{v}_{i}^{\top} \vec{v}_{i}
    \end{bmatrix} \vec{p}^{i} \\
    & \triangleq \left( \tilde{T}^{i} \right)^{\top} \vec{p}^{i},
\end{aligned}
$$

The potential energy follows

$$
\begin{aligned}
    V_{i} &= - m_{i} \vec{g} \cdot \left( \vec{r}_{i} + \vec{r}_{Ci} \right) \\
    &= - \vec{g}^{\top} \left( \vec{r}_{i} m_{i} + R_{i}^{0} m_{i} \vec{r}_{Ci} \right) \\
    &= \left[ \vec{0}_{1 \times 6}, - \vec{g}^{\top} R_{i}^{0}, - \vec{g}^{\top} \vec{r}_{i} \right] \vec{p}^{i} \\
    & \triangleq \left( \tilde{V}^{i} \right)^{\top} \vec{p}^{i}.
\end{aligned}
$$

## 2. Linearized Dynamics

Then total kinematic energy $T$ and potential energy $V$ follow

$$
\begin{aligned}
    T &= \sum_{i = 1}^{n} \left( \tilde{T}^{i} \right)^{\top} \vec{p}^{i} \\
    &= \left[ \left( \tilde{T}^{1} \right)^{\top}, \cdots, \left( \tilde{T}^{n} \right)^{\top} \right] \vec{p} \\
    & \triangleq \tilde{T}^{\top} \vec{p}, \\
    V &= \sum_{i = 1}^{n} \left( \tilde{V}^{i} \right)^{\top} \vec{p}^{i} \\
    &= \left[ \left( \tilde{V}^{1} \right)^{\top}, \cdots, \left( \tilde{V}^{n} \right)^{\top} \right] \vec{p} \\
    & \triangleq \tilde{V}^{\top} \vec{p},
\end{aligned}
$$

where $\vec{p} \triangleq \left[ \left( \vec{p}^{i} \right)^{\top}, \cdots, \left( \vec{p}^{n} \right)^{\top} \right]^{\top} $ contains all dynamical parameters. Noting that it's highly possible that $\vec{p}$ has zero-valued or linearly correlated elements.

With the 2nd Lagrangian equation, it can be obtained that

$$
\begin{aligned}
    \vec{\tau} &= \frac{d}{d t} \frac{\partial T}{\partial \dot{\vec{q}}} - \frac{\partial T}{\partial \vec{q}} + \frac{\partial V}{\partial \vec{q}} \\
    &= \left[ \frac{d}{d t} \frac{\partial \tilde{T}^{\top}}{\partial \dot{\vec{q}}} \right] \vec{p} - \frac{\partial \tilde{T}^{\top}}{\partial \vec{q}} \vec{p} + \frac{\partial \tilde{V}^{\top}}{\partial \vec{q}} \vec{p} \\
    &= \left[ \frac{d}{d t} \frac{\partial \tilde{T}^{\top}}{\partial \dot{\vec{q}}} - \frac{\partial \tilde{T}^{\top}}{\partial \vec{q}} + \frac{\partial \tilde{V}^{\top}}{\partial \vec{q}} \right] \vec{p} \\
    & \triangleq R_{n \times 10 n} \vec{p}_{10 n \times 1}
\end{aligned}
$$

The above equation gives a linearized form of manipulator's dynamics. $R = \frac{d}{d t} \frac{\partial \tilde{T}^{\top}}{\partial \dot{\vec{q}}} - \frac{\partial \tilde{T}^{\top}}{\partial \vec{q}} + \frac{\partial \tilde{V}^{\top}}{\partial \vec{q}}$ is the parameter regression matrix and $\vec{p}$ is dynamical parameter vector.

## 3. Simplification of Linearized Dynamics

Although we've already obtained

$$
\begin{gathered}
    \vec{\tau} = R \vec{p}, \\
    R = \frac{d}{d t} \frac{\partial \tilde{T}^{\top}}{\partial \dot{\vec{q}}} - \frac{\partial \tilde{T}^{\top}}{\partial \vec{q}} + \frac{\partial \tilde{V}^{\top}}{\partial \vec{q}} \\
    \vec{p} = \left[ \left( \vec{p}^{i} \right)^{\top}, \cdots, \left( \vec{p}^{n} \right)^{\top} \right]^{\top},
\end{gathered}
$$

zero-valued columns and linearly correlated columns in $R$ may lower correctness when performing online dynamical parameter estimation. Therefore, further simplification should be done.

1. If the $i$th column of $R$, denoted as $R_{i}$, follows

   $$
   R_{i} \equiv \vec{0}
   $$

Then the corresponding $i$th element in $\vec{p}$ has no influence on dynamics. Therefore, the $i$th column of $R$ and $i$th element of $\vec{p}$ can be eliminated when $R_{i} = \vec{0}$ (or $p_{i} = 0$).

2. If a certain column of $R$ is a linear correlation of other columns:

$$
R_{i} \equiv \beta_{i_{1}} R_{i_{1}} + \cdots + \beta_{i_{m}} R_{i_{m}},
$$

where $\beta_{i_{1}}, \cdots, \beta_{i_{m}} $ are constants. Then it's obvious that

$$
R_{i} p_{i} \equiv R_{i_{1}} \left( \beta_{i_{1}} p_{i_{1}} \right) + \cdots + R_{i_{m}} \left( \beta_{i_{m}} p_{i_{m}} \right)
$$

Then we may set

$$
p_{i_{j}} \rightarrow p_{i_{j}} + \beta_{i_{j}} p_{i},
$$

and $p_{i} \rightarrow 0 $.

## 4. Substitution with $\vec{\psi}$ and $\vec{\phi}$

In former 3 sections, we managed to obtain a linearized form of serial manipulator's dynamic:

$$
M \left( \vec{q} \right) \ddot{\vec{q}} + C \left( \vec{q}, \vec{\dot{q}} \right) \dot{\vec{q}} + G \left( \vec{q} \right) = R \left( \vec{q}, \dot{\vec{q}}, \ddot{\vec{q}} \right) \vec{p},
$$

where

$$
\begin{aligned}
    M \left( \vec{q} \right) \ddot{\vec{q}} &= \frac{d}{d t} \frac{\partial T}{\partial \vec{\dot{q}}}, \\
    \sum_{j =1}^{n} C_{ij} \left( \vec{q}, \dot{\vec{q}} \right) \dot{q}_{j} &= \sum_{j =1}^{n} \left[ \sum_{k = 1}^{n} \frac{\dot{q}_{k}}{2} \left( \frac{\partial M_{ik}}{\partial q_{j}} + \frac{\partial M_{ij}}{\partial q_{k}} - \frac{\partial M_{jk}}{\partial q_{i}} \right) \right] \dot{q}_{j}.
\end{aligned}
$$

We have the following conclusions:

1. Acceleration $\ddot{q}_{i}$s appear individually. We may simplily substitute $\ddot{q}_{i}$ with $\psi_{i}$.
2. Velocity $\dot{q}_{i}$s appear in the form of $\dot{q}_{i} \dot{q}_{j}$, which indicates that we should substitute $\dot{q}_{i} \dot{q}_{j}$ with $\frac{1}{2} \left( \dot{q}_{i} \phi_{j} + \dot{q}_{j} \phi_{i} \right)$, in order to guarantee symmetry of inertial and Coriolis force related terms.
