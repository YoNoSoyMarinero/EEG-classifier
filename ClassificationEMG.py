from EMGDataFrame import EMGDataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

class ClassificationEMG:

    @classmethod
    def classification_model_train(self, df: EMGDataFrame, discrimination_type: str):

        x_train = df.train_df.drop(['type'], axis=1)
        y_train = df.train_df['type']
        MODELS = ['linear', 'quadratic', 'knn', 'random_forest']

        if MODELS.index(discrimination_type) ==  0 :
            lda = LinearDiscriminantAnalysis()
            lda.fit(x_train, y_train)
            return lda
        elif MODELS.index(discrimination_type) == 1:
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(x_train, y_train)
            return qda
        elif MODELS.index(discrimination_type) == 2:
            knn = KNeighborsClassifier(5)
            knn.fit(x_train, y_train)
            return knn
        elif MODELS.index(discrimination_type) == 3:
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(x_train, y_train)
            return rf
        
    @classmethod
    def classification_model_test(self, df: EMGDataFrame, model: QuadraticDiscriminantAnalysis):
        x_test = df.test_df.drop(['type'], axis=1)
        y_test = df.test_df['type']

        y_predicted = model.predict(x_test)
        return confusion_matrix(y_test, y_predicted), accuracy_score(y_test, y_predicted)