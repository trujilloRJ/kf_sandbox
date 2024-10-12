# EKF vs. UKF for non-linear state estimation

## Motivation

The Kalman Filter (KF) is used for estimating the state of a dynamic system from a series of noisy measurements. KF guarantees an optimal estimation if these two conditions are met:

- **Linear state-space models**: The system state transition (the $F$ matrix) and state-to-measurement function (the $H$ matrix) are linear.

- **Gaussian errors**: The process and measurement noise ($Q$ and $R$ matrices) can be modelled as Gaussian distributions.

In real systems, these conditions are rarely met. In particular, for non-linear systems, the community had proposed variations of the original KF that are **sub-optimal** but still provide reasonably good preformance. Among the proposed solutions, two approaches have been widely adopted, the Extended-KF (EKF) and the Unscented-KF (UKF).

The EKF seek to linearize the non-linear functions around the current estimate. In practice, due to complexity and computational cost, only the first-order Taylor series expansion term is used for the linearization. As such, the EKF requires the computation of Jacobians which for some applications cannot be analytically derived. On the other hand, the UKF uses a different approach. Rather than approximating the non-linear function, the UKF seek to estimate the resulting distribution mean and covariance. To do so, it choses a minimal set points (sigma points) from the current state distribution and propagate them through the true non-linear system. The propagated sigma points are then used to estimate the posterior mean and covariance.

As an engineer...

## Table of Contents

1. [EKF and UKF algorithm](#theory)
2. [Implementation](#example2)
3. [Validation set-up](#validation)
4. [Results](#Results)
4. [Conclusions](#Conclusions)

