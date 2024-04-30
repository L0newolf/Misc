# Object Classifier Module

This package is meant to handle all the independent object detection/classification.
The package is meant to provide a simple API to access use any model/method behind the scenes.

### Machine-Learning
Classifier.py provides a simple interface to train using scikit-learn and hand-written learning models such as -

<li> Mean
<li> svm
<li> PCA
<li> LDA

Possible Functionalities -
<li> train & evaluate <li> predict <li> save & load

### Deep-Learning
Currently the only trainable method is based on the Object Detection API provided by tensorflow. Details to set this up can be found here -

Once you've set up the object_detection api from tensorflow/models, you can use the following workflow to train, fine-tune, evaluate & extract features from the models supported by Object Detection API.

#### Data Preparation
This step has to be carried out with caution, as it is very crucial & can significantly alter the outcome of the training.

Ensure that all the data that is to be used for training is labeled with bounding boxes and object category/name. There are helper functions that can help convert from vatic & pascal annotations

Vatic -
<li> extract the obj images from videos into training & validation sets
<li> convert directly to .tfrecord
<li> extract features and store it as a dict in pickle

Pascal_VOC -
<li> convert directly to .tfrecord

At the end of this step, you must have a train.tfrecord & eval.tfrecord for training & validation steps respectively.

Finally, you also need to create label & config files to be fed into the object detection api. This step is fairly straight-forward and follows the original steps from OBJ DET API.

Remember to update all the paths & # classes in the config file.
Labels in tensorflow must start with id 1 and not 0. (I know, it sucks!)

#### Train
To train, simply execute the following command from the parent directory of $tensorflow_object_detection_api$

```
python object_detection\train.py --logtostderr --train_dir=$PATH_TO_TRAIN_DIR$  --pipeline_config_path=$PATH_TO_CONFIG_FILE$
```
$PATH_TO_TRAIN_DIR$ - is where the model & checkpoint files will be stored during the training.

The losses reported by the system during training are not to be taken at face value, because it uses the same set of data to train & test. What truly evaluates the performance of the model is how it performs on new data.

#### Evaluate
For this purpose, execute the following evaluate command WHILE you are training, so you can see how the model performs as the number of epochs increase.
```
python object_detection\eval.py --logtostderr --checkpoint_dir=$PATH_TO_TRAIN_DIR$  --pipeline_config_path=$PATH_TO_CONFIG_FILE$ --eval_dir=$PATH_TO_EVAL_DIR$
```

### Tensorboard
Tensorflow comes with a visualization tool called tensorboard, which can be very useful to visualize how your model is learning as it is learning. Once you have set up the evaluation, you can launch tensorboard by pointing to the eval directory, and it'll take you to the localhost url from where you can see how the model is doing.

```
tensorboard --logdir=D:\Datasets\Medical\model\eval
```

#### Freeze/Extract model
In order to deploy your trained model, you have to now convert the trained .ckpt & .pbtxt files into a .pb file

This step simple freezes the weights & biases stored in the checkpoint file, and also provides a trainable model should you wish to re-train/start over from this point.
There are ways to reduce computational over-head at run-time by removing training-only-layers during this step. Refer to tensorflow documentation for doing that.

```
python object_detection\export_inference_graph.py --input_type image_tensor --pipeline_config_path=$PATH_TO_CONFIG_FILE$ --trained_checkpoint_prefix $PATH_TO_TRAIN_DIR\XXXX.ckpt-YYYY$ --output_directory=$PATH_TO_OUTPUT_DIR$
```
$PATH_TO_TRAIN_DIR\XXXX.ckpt-YYYY$ - is the checkpoint file you want to freeze.

#### Deploy
Once you have the .pb file, you are all set to go. All you need to do now is take your $LABELS$.pbtxt, $MODEL$.pb & extract the results from pc/android/any device supported by tensorflow.

#### Extract features
To extract features, you typically get the values from the last fully connected layer (this is usually referred to as the bottleneck layer). Check the model documentation for the exact layer names. There's also a utility function that can display the model architecture & layers under tf_utils.py called inspect_graph().
