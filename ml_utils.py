# 모듈 로딩
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error # 성능평가 관련 모듈 - MSE, MAE, RMSE 평가
# --------------------------------------------------------------
# 함수 기능: 평가결과 반환함수
# 함수 이름: checkModel
# 매개변수: 학습용데이터셋, 테스트용데이터셋
# 함수결과 : 결과값 DataFrame, 결과문자열
# -------------------------------------------------------------
def checkModel(X_train, X_test, y_train, y_test, model):
    result=[] # 성능평가 결과 저장

    for data, label in [[X_train, y_train],[X_test, y_test]]:
        # 모델 즉, 수식에 데이터 적용해서 예측값
        pre_label = model.predict(data)

        # 모델 성능 평가 -> score(2D_피쳐, 1D_타겟): 모델 적합도
        score = model.score(data, label)

        # 손실 계산 평가 -> rmse, mse, mae...(1D_타겟(정답), 1D_예측값)
        mse = mean_squared_error(label, pre_label) # (정답,예측값)
        mae = mean_absolute_error(label,pre_label)
        rmse = root_mean_squared_error(label,pre_label)


        # 데이터셋별 성능결과 저장
        result.append([score, rmse, mae, mse])

    # 성능평가 결과 DataFrame
    resultDF = pd.DataFrame(result,columns=['score','rmse','mae','mse'], index=['train','test'])
    # 훈련용과 테스트용 점수 차이
    resultDF.loc['diff']=(resultDF.loc['train']-resultDF.loc['test']).abs()
    
    return resultDF


# --------------------------------------------------------------
# 함수 기능: 예측값과 실제값 비교 시각화 함수
# 함수 이름: plot_prediction
# 매개변수: 예측값, 실제값
# 함수결과 : 그래프
# -------------------------------------------------------------
def plot_prediction(expected, predicted):
    plt.figure(figsize=(8, 4))
    plt.scatter(expected, predicted, alpha=0.7, edgecolors='k')

    # 최소/최대값 자동 설정 (데이터 범위 기반)
    min_val = min(min(expected), min(predicted))
    max_val = max(max(expected), max(predicted))

    # 여유 있는 범위 설정 (이상치 방지)
    buffer = (max_val - min_val) * 0.05  # 값 범위의 5% 추가
    min_val -= buffer
    max_val += buffer

    plt.plot([min_val, max_val], [min_val, max_val], '--r', label="Perfect Prediction")  # 기준선
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs True Values")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

#====================================================================
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print("오차행렬")
    print(confusion)
    print(f"정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")  

# ====================================================================
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train,tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]
    get_clf_eval(tgt_test, pred, pred_proba)


from sklearn.preprocessing import StandardScaler
# 정규화를 할건지 로그변환을 할건지에 따라 변경 가능
def get_preprocessed_df(df=None): 
    df_copy = df.copy()
    scaler = StandardScaler()
    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1,1))

    # 변환된 Amount를 Amount_Scaled로 피쳐면 변경 후 DataFrame 맨 앞 컬럼으로 입력
    df_copy.insert(0,'Amount_Scaled', amount_n)
    # 기존 Amount, Time 피쳐 삭제
    df_copy.drop(['Time','Amount'],axis=1,inplace=True)
    return df_copy

import numpy as np
def get_outlier(df=None, column=None, weight=1.5):
    # fraud에 해당하는 column 데이터만 추출, 1/4분위와 3/4분위 지점을 np.percentile로 구함
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values,25)
    quantile_75 = np.percentile(fraud.values,75)

    # IQR을 구하고, IQR에 1.5를 곱해 최댓값과 최솟값 지점 구함
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight

    # 최댓값보다 크거나, 최솟값보다 작은 값을 이상치 데이터로 설정하고 DataFrame index 반환
    outlier_index = fraud[(fraud<lowest_val) | (fraud> highest_val)].index
    return outlier_index


from sklearn.metrics import precision_recall_curve
def precision_recall_curve_plot(y_test, pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    # X축을 threshold값으로, y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    thresholds_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:thresholds_boundary], linestyle='--',label='precision')
    plt.plot(thresholds, recalls[0:thresholds_boundary],label='recall')

    # threshold 값 X축의 Sclae을 0.1단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start,end,0.1),2))
    plt.legend()
    plt.grid()
    plt.show()


#===================================================================

from sklearn.model_selection import cross_val_score
def get_avg_rmse_cv(models):
    for model in models:
        # 분할하지 않고 전체 데이터로 cross_val_score() 수행. 모델별 CV RMSE 값과 평균 RMSE 출력
        # cross_val_score은 mse값 반환, 부호변경, 루트
        rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target,
                                                            scoring="neg_mean_squared_error",cv=5)) # neg_mean_squared_error는 음수값
        rmse_avg = np.mean(rmse_list)
        print(f"\n{model.__class__.__name__} CV RMSE 값 리스트: {np.round(rmse_list,3)}")
        print(f"\n{model.__class__.__name__} CV 평균 RMSE 값: {np.round(rmse_avg,3)}")
    
from sklearn.model_selection import GridSearchCV    
def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, 
                              scoring='neg_mean_squared_error', cv=5)
    grid_model.fit(X_features, y_target)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha:{2}'.format(model.__class__.__name__,
                                        np.round(rmse, 4), grid_model.best_params_))
    return grid_model.best_estimator_



