from datasets import load_dataset

# load cifar10 (only small portion for demonstration purposes)
train_ds, test_ds = load_dataset("cifar10", split=["train", "test"])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

train_ds.save_to_disk("train_ds")
val_ds.save_to_disk("val_ds")
test_ds.save_to_disk("test_ds")

# >>> from datasets import load_from_disk
# >>> reloaded_encoded_dataset = load_from_disk("path/of/my/dataset/directory")
