import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import load_model
from sklearn.model_selection import KFold

data_fp = pd.read_csv('./data/fp(256).txt')
y = pd.read_csv('./data/labels(2248).csv')
data_att = np.load('./data/drugs_structure_attentivefp_ToxCast.npy')
masking = np.load('./data/gin_supervised_masking.npy')
infomax = np.load('./data/gin_supervised_infomax.npy')
edge = np.load('./data/gin_supervised_edgepred.npy')
con = np.load('./data/gin_supervised_contextpred.npy')


#定义回调函数
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau#回调函数
# # 定义回调函数：保存最优模型
checkpoint = ModelCheckpoint("./model/model.hdf5",
                             monitor="val_accuracy",
                             mode="max",
                             save_best_only = True,
                             save_weights_only=False,
                             verbose=1)
# 定义回调函数：提前终止训练
earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0.0001,
                          patience = 40,
                          verbose = 3,
                          mode = 'min',
                          restore_best_weights = True)
# 定义回调函数：学习率衰减
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 10,
                              verbose = 3,
                              min_delta = 0.0001)

# 将回调函数组织为回调列表
callbacks = [earlystop,reduce_lr]





#定义模型
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy',threshold=0.54),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='AUC',multi_label=True,num_labels=27,num_thresholds=498),
      keras.metrics.AUC(name='AUPR',curve='PR',multi_label=True,num_labels=27,num_thresholds=498), # precision-recall curve
]

def create_model(metrics=METRICS):
  model = Sequential([
    layers.Conv1D(32, 2, activation='relu', input_shape=(300,1)),
    layers.MaxPooling1D((2)),
    # layers.Conv1D(32, 2, activation='relu'),
    # layers.MaxPooling1D((2)),
    # layers.Flatten(name="my_intermediate_layer"),
    # layers.Dropout(0.4),
    layers.Dense(1024,name="my_feature"),
    layers.ReLU(),
    layers.Dropout(0.4),
    layers.Dense(526),
    layers.ReLU(),
    layers.Dropout(0.4),
    layers.Dense(27, activation='sigmoid', name="my_last_layer"),
    ])
  return model



#训练、10折叠、在masking特征上进行训练


results = []
# 运行计时部分
time_start = time.time()

kfolder = KFold(n_splits=10, shuffle=True, random_state=2021)
for i, (tra_id, val_id) in enumerate(kfolder.split(masking, y)):
    print(f"***********fold-{i + 1}***********")

    c_x_train = masking[tra_id]
    c_y_train = y.iloc[tra_id]

    c_x_valid = masking[val_id]
    c_y_valid = y.iloc[val_id]

    # scaler = preprocessing.StandardScaler().fit(c_x_train)
    # c_x_train = scaler.transform(c_x_train)

    # scaler = preprocessing.StandardScaler().fit(c_x_valid)
    # c_x_valid = scaler.transform(c_x_valid)

    # reshape for cnn training
    # c_x_test = np.array(c_x_test).reshape(c_x_test.shape[0], c_x_test.shape[1],1)
    c_x_valid = np.array(c_x_valid).reshape(c_x_valid.shape[0], c_x_valid.shape[1], 1)
    c_x_train = np.array(c_x_train).reshape(c_x_train.shape[0], c_x_train.shape[1], 1)

    model = create_model()

    opt = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=METRICS)

    # training
    history = model.fit(c_x_train, c_y_train.values, batch_size=128, epochs=300,
                        validation_data=(c_x_valid, c_y_valid.values), callbacks=callbacks)

    baseline_results = model.evaluate(c_x_valid, c_y_valid.values, verbose=0)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()
    # y_pre = model(c_x_valid)
    # np.save('./ADR.v2/data/y_pre_FP2'+str(i+1)+'.npy',y_pre)

    # model.save('./model/model_FP_masking_cross'+str(i+1)+'.h5')

    result = np.array(baseline_results)
    results.append(result)

print("******************************************************************")
# print(np.array(results))

print(f"-Accuracy score_mean:{np.mean(np.array(results)[:, [1]])}")
# print(f"-Accuracy score_mean:{np.array(results)[:,[1]]}")
print(f"-Accuracy score_std:{np.std(np.array(results)[:, [1]])}")

print(f"-Precision score_mean:{np.mean(np.array(results)[:, [2]])}")
print(f"-Precision score_std:{np.std(np.array(results)[:, [2]])}")

print(f"-Recall score_mean:{np.mean(np.array(results)[:, [3]])}")
print(f"-Recall score_std:{np.std(np.array(results)[:, [3]])}")

print(f"-AUC score_mean:{np.mean(np.array(results)[:, [4]])}")
print(f"-AUC score_std:{np.std(np.array(results)[:, [4]])}")

print(f"-AUPR score_mean:{np.mean(np.array(results)[:, [5]])}")
print(f"-AUPR score_std:{np.std(np.array(results)[:, [5]])}")

# 运行计时部分
time_end = time.time()
print(f"total running time: {(time_end - time_start) / 60} minites")