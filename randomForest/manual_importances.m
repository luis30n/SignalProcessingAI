%funcion de cáculo de las importancias de las variables a partir de la
%matriz T (indica los regresores incluidos en cada modelo aleatorio y el
%NMSE conseguido)
function [importances] = manual_importances(X, y)
[Xtrain, idtrain]  = datasample(X, round(0.66*n));
idtest       = 1:n;
idtest(idtrain) = [];
X_test         = X(idtest);
for train_idx, test_idx in range(10):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = manual_randomforest.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)
print ("Features sorted by their score:")
print (sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True))
end
    end
end
end
