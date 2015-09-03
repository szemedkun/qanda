from datasets import Datasets

ds_false = Datasets(task_index = 1, only_supporting = False)
ds_true = Datasets(task_index = 1, only_supporting = False)
ds_true.fit()
ds_false.fit()

X_t, qX_t, Y_t = ds_true.get_training_data()
X_f, qX_f, Y_f = ds_false.get_training_data()