---
layout: default
title: Score and Win Predictor
---

## Score and Win Predictor
After earning my cricket analytics certificate, I started applying some of the techniques I learned to analyze data from both the [South Africa vs. India](https://github.com/hmshreyas7/sa-vs-ind-2021) Test series and the [Ashes](https://github.com/hmshreyas7/ashes-2021) in 2021/22. Through this experience, I was able to take a closer look at very intriguing aspects of the game such as a team's reliance on a single batter, batters that have stepped up when their team is under pressure, effectiveness of teams with the ball based on how old it is, and so on.

Once I got a taste of data analytics in cricket thanks to the above experiments, I wanted to build something bigger. Drawing inspiration from popular prediction tools used in the industry such as [ESPNcricinfo's Forecaster](https://www.espncricinfo.com/video/a-look-at-espncricinfo-s-forecaster-1178477) and [CricViz's WinViz](https://www.cricviz.com/), I felt it would be cool to apply the AI & ML knowledge that I gained at university level to solve this problem on my own.

<p style="margin-bottom: 0;"><img src="/assets/ind-vs-sl-2nd-t20i-2022-score-pred.png" width="100%" /></p>
<p style="margin-top: 0;" align = "center"><em>Sri Lanka batting first vs. India - 2nd T20I - 2022</em></p>
<!--more-->
The above plot shows Sri Lanka's projected scores using a [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) model at different stages. It considers the current run rate, number of balls left, and number of wickets in hand to predict the run rate for the remainder of the innings which is then used to calculate the expected final score.

Ball-by-ball data from 800+ T20 international matches provided by [Cricsheet](https://cricsheet.org/) is used for this experiment. 80% of the data is used for training and the remaining 20% is used for testing but before training, a [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) is constructed through which the features are first [standardized](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

After making predictions on the test set using the trained model, 3 things are used to evaluate it: adjusted R-squared, RMSE/MAE, and a residual plot. The adjusted R-squared value observed for this experiment is 0.1742 which appears to be really low and might give the impression that the chosen model is not good. That, however, isn't necessarily the [case](https://blog.minitab.com/en/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit).

[RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) and [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) are 2 error measures that are often used in statistics which give us the average difference between the observed and the predicted values. My regression model had an RMSE of 2.9126 and an MAE of 2.0598 which suggests that, on average, there is a difference of just 2-3 runs per over between what the model predicts and what run rate the batting team actually achieves.

Lastly, the predicted values are plotted against the [standardized residuals](https://stackoverflow.com/questions/65159648/how-to-find-the-standardized-residuals-with-sklearn):

<p style="margin-bottom: 0;"><img src="/assets/score-prediction-linear-regression-residual-plot.png" width="100%" /></p>
<p style="margin-top: 0;" align = "center"><em>Residual plot for the linear regression model</em></p>

This seems to meet most of the [requirements](https://www.qualtrics.com/support/stats-iq/analyses/regression-guides/interpreting-residual-plots-improve-regression/) but there probably is some room for improvement. More importantly, it shows that the prediction error is consistent for a significantly large number of points even though the adjusted R-squared value is low. This tells us that the model might be adequate for the given problem.

Predicting the final score for the team batting first is only one half of the experiment though. The next task is to use ball-by-ball data from T20I chases to build a win prediction model using [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html).

<p style="margin-bottom: 0;"><img src="/assets/ind-vs-sl-2nd-t20i-2022.png" width="100%" /></p>
<p style="margin-top: 0;" align = "center"><em>India chasing vs. Sri Lanka - 2nd T20I - 2022</em></p>

Note that in the above plot, the **value** that the line corresponds to only represents the chasing team's win probability but the **color** indicates which team is ahead in the game at any given point in time.

A lot of the steps involved in this are very similar to the score predictor but additionally other key features such as the target to be chased, runs remaining, and required run rate are also included. The evaluation procedure is different as well - [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html), [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), and [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) are used and the last 2 of those are as shown below:

<p style="margin-bottom: 0; display: table;">
	<img src="/assets/win-prediction-logistic-regression-roc-auc.png" width="50%" />
	<img src="/assets/win-prediction-logistic-regression-confusion-matrix.png" width="50%" />
</p>
<p style="margin-top: 0;" align = "center"><em>ROC AUC and confusion matrix for the logistic regression model</em></p>

Both the prediction models seem to perform well not only based on the evaluation metrics but also based on how well the graphs capture the different events that happen during live games, i.e., there is a clear connection between the direction of the line and an event in the game such as a six, four, dot ball, or wicket.

Outputs produced by non-linear models such as[ decision trees](https://scikit-learn.org/stable/modules/tree.html) and [random forests](https://scikit-learn.org/stable/modules/ensemble.html#forest) appear to lack this connection despite producing similar performance levels. For example, a wicket falling might hardly lead to any change in the line's direction and then an over later, it might drop sharply for no apparent reason. This isn't very intuitive and may seem confusing from a viewer's perspective. Tuning the hyperparameters of these models could potentially help mitigate this issue.

Considering other features such as which batsmen are at the crease, the quality of opposition bowlers, how good specific players are in certain phases of play, the ground/pitch/weather conditions, etc. may also help in building superior models. Acquiring all this information for past matches and then using it for model training, however, is a big challenge.