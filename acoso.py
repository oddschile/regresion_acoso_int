import pandas as pd
import seaborn as sns; sns.set()

acoso = pd.read_csv("C:/path/tofile/acosoint.csv")

acoso["acoso_int"].value_counts()

acoso["edad"].describe()

y=['acoso_int']
X=[i for i in acoso if i not in y]



from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, 18)

rfe = rfe.fit(acoso[X], acoso[y])
print(rfe.support_)
print(rfe.ranking_)


cols=["sexo", "edad", "llorar", "escapar","solo", "triste"] 
X=acoso[cols]
y=acoso['acoso_int']


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print('Precisión del en el conjunto de prueba: {:.2f}'.format(logreg.score(X_test, y_test)))

X_new = [0, 15, 1, 1, 1, 1]


new_prediction = logreg.predict(X_new)

print("Prediccion: {}".format(new_prediction))

X_new2 = [1, 18, 0, 0, 0, 0]


new_prediction2 = logreg.predict(X_new2)

print("Prediccion: {}".format(new_prediction2))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("Exactitud de la validación cruzada: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Regresión Logística (área = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Característica de funcionamiento del receptor (Curva ROC)')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()




