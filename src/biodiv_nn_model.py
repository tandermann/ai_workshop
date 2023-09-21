import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# This is a function that let's you rescale your data array. It rescales all values in each column to a range between 0 and 1.
def min_max_scaler(x, min_array=None, max_array=None, inverse_scale=False):
    if inverse_scale:
        x_new = x*(max_array-min_array)+min_array
        return(x_new)
    else:
        min_array = np.min(x,axis=0)
        max_array = np.max(x,axis=0)
        x_new = (x - min_array)/(max_array - min_array)
        return(x_new,min_array,max_array)


def select_train_val_test(x,val_fraction=0.2,test_fraction=0.2,shuffle=True,seed=None):
    all_indices = np.arange(len(x))
    if shuffle:
        if not seed:
            seed = np.random.randint(0,999999999)
        # shuffle all input data and labels
        np.random.seed(seed)
        print('Shuffling data, using seed', seed)
        shuffled_indices = np.random.choice(all_indices, len(all_indices), replace=False)
    else:
        shuffled_indices = all_indices
    # select train, validation, and test data
    n_test_instances = np.round(len(shuffled_indices) * test_fraction).astype(int)
    n_validation_instances = np.round(len(shuffled_indices) * val_fraction).astype(int)
    test_ids = shuffled_indices[:n_test_instances]
    validation_ids = shuffled_indices[n_test_instances:n_test_instances + n_validation_instances]
    train_ids = shuffled_indices[n_test_instances + n_validation_instances:]
    return train_ids, validation_ids, test_ids

# class MCDropout(tf.keras.layers.Dropout):
#     def call(self, inputs):
#         return super().call(inputs, training=True)

def plot_training_history(history, show_best_epoch=True):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history.history['mae'], label='Training set')
    plt.plot(history.history['val_mae'], label='Validation set')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    if show_best_epoch:
        best_epoch = np.where(history.history['val_mae'] == np.min(history.history['val_mae']))[0][0]
        plt.axvline(best_epoch, c='grey', linestyle='--')
        plt.axhline(history.history['val_mae'][best_epoch], c='grey', linestyle='--')
        plt.gca().axvspan(best_epoch, len(history.history['mae']), color='grey', alpha=0.3, zorder=3)
    plt.grid()
    plt.legend(loc='upper center')
    plt.show()

def plot_true_vs_pred(true_labels, predicted_labels):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(true_labels, predicted_labels, 'o', markersize=3, alpha=1)
    plt.grid()
    plt.xlabel('True diversity')
    plt.ylabel('Predicted diversity')
    plt.show()


# load data
tbl = pd.read_csv("/Users/tobiasandermann/Documents/teaching/ai_course_geosciences/biodiv_exercise/data/div_data_all_features.txt", delimiter="\t")
features = tbl.values[:,1:]
feature_names = tbl.columns[1:]
labels = tbl.values[:,0]
plt.hist(labels)
plt.show()
# Rescale labels and features
rescaled_labels = labels/800
rescaled_features, scale_min, scale_max = min_max_scaler(features)
#min_max_scaler(rescaled_features, scale_min, scale_max,inverse_scale=True)

# Separate instances into train and test set
train_set_ids, validation_set_ids, test_set_ids =  select_train_val_test(rescaled_features)
train_features = rescaled_features[train_set_ids]
train_labels = rescaled_labels[train_set_ids]
validation_features = rescaled_features[validation_set_ids]
validation_labels = rescaled_labels[validation_set_ids]
test_features = rescaled_features[test_set_ids]
test_labels = rescaled_labels[test_set_ids]


architecture = []
# Input layer
architecture.append(tf.keras.layers.Flatten(input_shape=[train_features.shape[1]]))
# First hidden layer
architecture.append(tf.keras.layers.Dense(32, activation='relu'))
architecture.append(tf.keras.layers.Dropout(0.2))
#architecture.append(MCDropout(0.2))
# 2nd hidden layer
architecture.append(tf.keras.layers.Dense(8, activation='relu'))
architecture.append(tf.keras.layers.Dropout(0.1))
#architecture.append(MCDropout(0.1))
# Output layer
architecture.append(tf.keras.layers.Dense(1, activation='softplus'))  # sigmoid or tanh or softplus
# Compile the model
model = tf.keras.Sequential(architecture)
model.compile(loss='mae', optimizer='adam', metrics=['mae','mape','mse','msle'])
# Get overview of model architecture
model.summary()



architecture = []

# Input layer
architecture.append(tf.keras.layers.Flatten(input_shape=[4]))
# 1st hidden layer
architecture.append(tf.keras.layers.Dense(5, activation='relu',use_bias=False))
# Output layer
architecture.append(tf.keras.layers.Dense(5, activation='softmax',use_bias=False))

# Compile the model
model = tf.keras.Sequential(architecture)
model.compile(loss='categorical_crossentropy',
 optimizer='adam', metrics=['accuracy'])

# Get overview of model architecture
model.summary()



# Define early stop of training based on validation set
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae',patience=200,restore_best_weights=True)
# Run model training and store training history
history = model.fit(train_features,
                          train_labels,
                          epochs=1000,
                          validation_data=(validation_features, validation_labels),
                          verbose=1,
                          callbacks=[early_stop],
                          batch_size=40)

plot_training_history(history,show_best_epoch=False)

# You can save the model to a file and later load it
model_file = '/Users/tobiasandermann/Documents/teaching/ai_course_geosciences/biodiv_exercise/data/trained_model'
model.save(model_file)
# This is how you load the model
#model = tf.keras.models.load_model(model_file)

estimated_train_labels = model.predict(train_features)
estimated_validation_labels = model.predict(validation_features)
estimated_test_labels = model.predict(test_features)




plot_true_vs_pred(train_labels*800,estimated_train_labels*800)
plot_true_vs_pred(validation_labels*800,estimated_validation_labels*800)
plot_true_vs_pred(test_labels*800,estimated_test_labels*800)




mc_dropout_pred = np.stack([model(test_features,training=True) for i in np.arange(100)])
mc_dropout_mean = mc_dropout_pred.mean(axis=0)
mc_dropout_std = mc_dropout_pred.std(axis=0)
mc_dropout_min = mc_dropout_pred.min(axis=0)
mc_dropout_max = mc_dropout_pred.max(axis=0)

plot_true_vs_pred(test_labels*800,mc_dropout_mean*800)


instance_id = 4
stepsize = 5
estimates = mc_dropout_pred[:,instance_id,:].flatten()*800
estimates = np.random.normal(2.1,0.2,100)
plt.hist(estimates,20)#np.arange(min(estimates),max(estimates+stepsize),stepsize))
plt.axvline(np.mean(estimates), color='r', linestyle='dashed', linewidth=1)
plt.xlim(0,5.5)
plt.grid()
plt.show()

np.log(10)

fig = plt.figure(figsize=[5,5])
#plt.errorbar(test_labels*800,mc_dropout_mean*800, yerr=[mc_dropout_min.flatten()*800,mc_dropout_max.flatten()*800], fmt='o',ecolor='black',elinewidth=0.2)
plt.errorbar(test_labels*800,mc_dropout_mean*800, yerr=mc_dropout_std.flatten()*800, fmt='.',alpha=1,ecolor='black',elinewidth=0.5)
plt.xlim(0,800)
plt.ylim(0,800)
plt.plot([0,800],[0,800],'r-')
plt.xlabel('True diversity')
plt.ylabel('Predicted diversity')
plt.tight_layout()
fig.savefig('/Users/tobiasandermann/Documents/teaching/ai_course_geosciences/biodiv_exercise/figs/mc_dropout_pred.pdf',bbox_inches='tight', dpi = 500)


import os
indir = '/Users/tobiasandermann/Documents/teaching/ai_course_geosciences/biodiv_exercise/data/iucn_data/'
labels_path = os.path.join(indir,'iucnn_train_labels.txt')
a = pd.read_csv(labels_path,sep='\t')
b = a['labels']
b.to_csv(labels_path,sep='\t',index=False)
