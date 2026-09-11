# Scoring and implementation logic

The project uses **root mean squared error (RMSE) on original
`Item_Outlet_Sales`**, consistent with Analytics Vidhya's
[BigMart solution tutorial](https://www.analyticsvidhya.com/blog/2016/02/bigmart-sales-solution-top-20/).
For evaluated test rows S, the objective is

```
sqrt(sum((actual_sales[i] - predicted_sales[i])**2 for i in S) / len(S))
```

Lower is better. A 2,000-unit error contributes four times the squared loss of
a 1,000-unit error. There is no classification accuracy, percentage-error
normalization, or benefit from rounding predictions to observed sales values.
Under squared error, the ideal point prediction is the conditional mean.
`src/cv.py:rmse` and every experiment score use this original sales scale.

If we train on units `u = sales / price`, weighting errors by `price**2`
reproduces the sales squared-error objective (up to a common normalization).
Equal-weight units can still generalize better when its assumptions reduce
estimation variance; we compare both using sales RMSE. Log-target training
must also be judged after converting predictions back to sales. Since
training sales are nonnegative, clipping a negative sales prediction to zero
cannot increase its squared error; the pipeline saves both raw and clipped
OOF results to measure its effect.

On **2026-09-11**, the unauthenticated browser view of the
[competition leaderboard](https://www.analyticsvidhya.com/datahack/leaderboard/practice-problem-big-mart-sales-iii/)
showed rank #1 at **1126.0294892930**, #2 at 1127.4410360237. Beating the
current leading score requires a lower score on the website's evaluated
hidden rows. This is a moving reference, not a target that a training CV
number can certify. A local RMSE of 1071 cannot be compared directly with
1126 on a different sample.

The current public contest landing page only exposed About/Discuss in this
browser, with Login/Register controls. Its authenticated evaluation details,
actual evaluated row count, public/private allocation, and hidden sales were
not available. The leaderboard UI exposes a private-leaderboard switch, but
that does not reveal the split proportions. No undocumented evaluator,
hidden labels, or other participants' predictions were accessed. Exact
evaluation details remain subject to the authenticated contest rules.

The implemented strategy is to reduce generalization error: preserve the
strong structural baseline, learn preprocessing inside folds, test regularized
boosting and price decoding, evaluate on identical saved folds, use grouped
products for robustness, and measure blend selection separately. We do not
shift predictions using a CV-to-leaderboard offset or select hundreds of
files against public feedback. The filename and code complexity do not
improve the score; accurate row-aligned predictions do.

A correctly formed CSV contains exactly `Item_Identifier`,
`Outlet_Identifier`, and `Item_Outlet_Sales`, in test-file order. The final
writer checks all 5,681 rows, identifiers, numeric finiteness and nonnegative
predictions, with no index column. The new CSV has no leaderboard score
until the user approves an upload and the platform evaluates it.
