"""Covariance estimators: sample, linear and nonlinear shrinkage, DCC-GARCH.

Each estimator satisfies the :class:`~portfolio_risk.estimators.base.CovarianceEstimator`
protocol: ``fit(returns)`` returns ``self`` and populates ``covariance_``, a
labelled :class:`pandas.DataFrame`.

Normalisation convention
------------------------
All estimators use the maximum-likelihood divisor ``n`` (not the
Bessel-corrected ``n - 1``). This matches scikit-learn's covariance module,
which the sample and Ledoit-Wolf estimators wrap, so comparisons involve no
convention conversions. The choice cannot affect the study's conclusions:

* portfolio weights from minimum-variance (or any budget-constrained)
  optimisation are invariant to rescaling the covariance matrix by a constant;
* for a 504-day window the ``n / (n - 1)`` factor is about 1.002, a 0.2%
  variance bias, roughly 30x smaller than the ~6% sampling error of a daily
  variance estimate over the same window.

To recover the unbiased estimator, multiply ``covariance_`` by ``n / (n - 1)``.
"""
