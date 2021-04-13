# 深度學習:理論及應用 HW3 - EMNIST CNN Classification
##### 學號: 109971014
## Code
#### Using Tensorflow
* /HW3.ipynb
## Report

### \#Model Baseline
#### Baseline Model
![Baseline Model](image/baseline_model.png)
##### Parameters
```
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(62, activation = "softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_label, epochs=20, batch_size=512, verbose=2, shuffle=True, validation_split=0.1)
```
##### Result
> Epoch: 20, Loss: 0.3160, Accuracy: 0.8811

### \#Model Improvement
#### 提升準確率之思路
> 增加 Layer 的數量以及 Cell 的數量

#### Improvement Model 1
![Improvement Model 1](image/improvement_1_model.png)
##### Parameters
```
model_2.add(Conv2D(6, kernel_size=5, strides=1, padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model_2.add(BatchNormalization())

model_2.add(Conv2D(16, kernel_size=5, strides=1, padding = 'Same', activation ='relu'))
model_2.add(BatchNormalization())

model_2.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
model_2.add(Dropout(0.3))

model_2.add(Conv2D(32, kernel_size=5, strides=1, padding = 'Same', activation ='relu'))
model_2.add(BatchNormalization())

model_2.add(Conv2D(64, kernel_size=5, strides=1, padding = 'Same', activation ='relu'))
model_2.add(BatchNormalization())
model_2.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
model_2.add(Dropout(0.3))

model_2.add(Conv2D(128, kernel_size=5, strides=1, padding = 'Same',  activation ='relu'))
model_2.add(BatchNormalization())
model_2.add(Dropout(0.3))

model_2.add(Flatten())

model_2.add(Dense(256, activation = "relu"))
model_2.add(BatchNormalization())
model_2.add(Dropout(0.25))

model_2.add(Dense(62, activation = "softmax"))

opt = Adam(learning_rate=0.005)
model_2.compile(loss='CategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])

model_2.fit(train_data, train_label, epochs=30, batch_size=512, shuffle=True, validation_split=0.1)
```
##### Result
> Epoch: 30, Loss: 0.3243, Accuracy: 0.8786

##### Note
> 對比 Baseline model，多加一層 & 加大一倍的連結網路數量，以及將**Dropout**從 0.2 調整至 0.3，Batch_size 調整為**128**
> 準確率提升 **0.0013**

### \#Training Procedure
#### Baseline Model
![Baseline Epoch Accuracy](image/baseline_epoch_accuracy.png)
![Baseline Epoch Loss](image/baseline_epoch_loss.png)

#### Improvement_1 Model
![Improvement Epoch Accuracy](image/improvement_1_epoch_accuracy.png)
![Improvement Epoch Loss](image/improvement_1_epoch_loss.png)

### \#Error Analysis
* Using ***Improvement_1 Model***
![Confusion Matrix](image/confusion_matrix.png)

#### CONCLUSION
> * **英文大寫'O'** 辨識成 **數字'0'** 達 **1036** 個樣本<br>
> * **英文大寫'I'** 辨識成 **數字'1'** 達 **272** 個樣本<br>
> * **英文小寫'l'** 辨識成 **數字'1'** 達 **269** 個樣本<br>
> * **英文大寫'S'** 辨識成 **數字'5'** 達 **224** 個樣本<br>
> * **英文小寫'c'** 辨識成 **英文大寫'C'** 達 **123** 個樣本<br>
> * **數字'1'** 辨識成 **英文大寫'I'** 達 **747** 個樣本<br>
> * **英文小寫'l'** 辨識成 **英文大寫'I'** 達 **179** 個樣本<br>
> * **英文小寫'm'** 辨識成 **英文大寫'M'** 達 **105** 個樣本<br>
> * **數字'0'** 辨識成 **英文大寫'O'** 達 **1660** 個樣本<br>
> * **數字'5'** 辨識成 **英文大寫'S'** 達 **152** 個樣本<br>
> * **英文大寫'C'** 辨識成 **英文小寫'c'** 達 **340** 個樣本<br>
> * **英文大寫'F'** 辨識成 **英文小寫'f'** 達 **256** 個樣本<br>
> * **數字'9'** 辨識成 **英文小寫'g'** 達 **133** 個樣本<br>
> * **數字'1'** 辨識成 **英文小寫'i'** 達 **104** 個樣本<br>
> * **英文大寫'K'** 辨識成 **英文小寫'k'** 達 **153** 個樣本<br>
> * **數字'1'** 辨識成 **英文小寫'l'** 達 **1585** 個樣本<br>
> * **英文大寫'I'** 辨識成 **英文小寫'l'** 達 **292** 個樣本<br>
> * **英文大寫'M'** 辨識成 **英文小寫'm'** 達 **327** 個樣本<br>
> * **數字'0'** 辨識成 **英文小寫'o'** 達 **211** 個樣本<br>
> * **英文大寫'O'** 辨識成 **英文小寫'o'** 達 **230** 個樣本<br>
> * **英文大寫'P'** 辨識成 **英文小寫'p'** 達 **219** 個樣本<br>
> * **數字'9'** 辨識成 **英文小寫'q'** 達 **193** 個樣本<br>
> * **英文大寫'S'** 辨識成 **英文小寫's'** 達 **378** 個樣本<br>
> * **英文大寫'U'** 辨識成 **英文小寫'u'** 達 **392** 個樣本<br>
> * **英文大寫'V'** 辨識成 **英文小寫'v'** 達 **242** 個樣本<br>
> * **英文大寫'W'** 辨識成 **英文小寫'w'** 達 **172** 個樣本<br>
> * **英文大寫'X'** 辨識成 **英文小寫'x'** 達 **134** 個樣本<br>
> * **英文大寫'Y'** 辨識成 **英文小寫'y'** 達 **194** 個樣本<br>
