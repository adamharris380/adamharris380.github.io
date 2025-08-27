
"""ECON 7510 (F24)
Lecture 10
Author: Adam Harris
"""

import numpy as np
import xgboost as xgb
import pandas as pd
from scipy.special import expit

np.random.seed(123)

# Generate data:
N = 2000
df = pd.DataFrame(np.random.randn(N, 10), columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
ϵ = 2 * np.random.rand(N)
df['y'] = np.random.rand(N) > expit(10 * df['a'] * np.exp(df['b']) - 5 * df['c'] + df['c'] * df['d'] + ϵ)
# convert y to int
df['y'] = df['y'].astype(int)

dtrain = xgb.DMatrix(df[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']], label=df['y'])

# can accept tabular data, will keep feature names
model = xgb.XGBClassifier(objective="binary:logistic",
                          max_depth=20,
                          eta=0.01,
                          n_estimators=1000)
model.fit(df[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']], df['y'])

# display importance statistics retaining feature names
importance = model.get_booster().get_score(importance_type='weight')

# Sort by importance
importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
print(importance)


# can accept tabular data, will keep feature names
bst = xgboost(dtrain,
			objective = "binary:logistic",
			max_depth = 3,
			η = 0.01,
			num_round = 10000)

# display importance statistics retaining feature names
\


# 
N_test = 500;
using DataFrames
df_test = DataFrame(randn(N_test,10), [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j])
ϵ_test = 2 * rand(N_test);
df_test.y = rand() .> logistic.(10 .* df_test.a .* exp.(df_test.b) - 5 .* df_test.c + df_test.c .* df_test.d + ϵ_test);
dtest = DMatrix((df_test[:, [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]], df_test.y));

watchlist = OrderedDict(["train" => dtrain, "valid" => dtest])
bst = xgboost(dtrain,
			objective = "binary:logistic",
			max_depth = 3,
			η = 0.01,
			num_round = 10000,
			gamma=1,
			watchlist=watchlist,
			early_stopping_rounds = 10)

function CV_train(dtrain,dcv,max_depth,eta,gamma,alpha)
	watchlist = OrderedDict(["train" => dtrain, "valid" => dcv])
	bst = xgboost(dtrain,
				objective = "binary:logistic",
				max_depth = max_depth,
				η = eta,
				gamma=gamma,
				alpha=alpha,
				num_round = 10000,
				watchlist=watchlist,
				early_stopping_rounds = 10,
				verbosity=0)
	return bst.best_iteration, bst.best_score
end

function CV_inner(k,df,max_depth,eta,gamma,alpha)
	dtrain = DMatrix((df[df.sample_split .!= k, [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]], df.y[df.sample_split .!= k]));
	dcv    = DMatrix((df[df.sample_split .== k, [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j]], df.y[df.sample_split .== k]));
	n, loss = CV_train(dtrain,dcv,max_depth,eta,gamma,alpha);
	return n, loss
end

function CV_outer(df,K,hyperparameter_candidates)
	Random.seed!(1234)
	df.sample_split = Int64.(floor.(rand(N) * 5) .+ 1);

	LOSS  = zeros(size(hyperparameter_candidates,1),K) .+ Inf;
	n_mat = Int64.(zeros(size(hyperparameter_candidates,1),K));

	for i = 1:size(hyperparameter_candidates,1)
		for k = 1:K
			n, loss = CV_inner(k,df,
					hyperparameter_candidates.max_depth[i],
					hyperparameter_candidates.eta[i],
					hyperparameter_candidates.gamma[i],
					hyperparameter_candidates.alpha[i]);
			if !ismissing(n)
				n_mat[i,k] = n;
				LOSS[i,k] = loss;
			end
		end
	end

	return n_mat, LOSS
end


combinations = IterTools.product(Int64.(collect(2:3:8)), 10.0 .^ (-2:1:-1), zeros(1), zeros(1));
hyperparameter_candidates = DataFrame([ (a, b, c, d) for (a, b, c, d) in combinations ], [:max_depth, :eta, :gamma, :alpha])

n_mat, LOSS = CV_outer(df,5,hyperparameter_candidates);
CV_avg_loss = mean(LOSS, dims=2)[:,1];
n = median(n_mat, dims=2)[:,1];
hyperparameter_candidates[findall(CV_avg_loss .== minimum(CV_avg_loss)),:]
Int64(n[findall(CV_avg_loss .== minimum(CV_avg_loss)),:][1,1])
