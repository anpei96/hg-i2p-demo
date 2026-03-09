This is an implementation demo of Hg-I2P. If you have trouble, feel free to contact me.

**Step one**: Pre-processing the I2P dataset using the Python scripts in tools_boxes. 
The pre-processing containts the segmentation 2D and 3D data using the pre-trained SAM models.
Also, other priors can be computed, such as surface normals and depths. There are not used in Hg-I2P.

**Step two**: Pre-training the baseline I2P registration model
Hg-I2P is based on the original model Matr(ICCV'23). Thus, we first train this model on the training dataset.
It is achieved by using the train_anyscene.py but the model is selected as model_matr.model

**Step three**: Training the proposed registration model Hg-I2P
Using the pre-trained model in step two, we train the our model Hg-I2P.
It is achieved by using the train_anyscene.py but the model is selected as model_fewshot.model_plus

**Step four**: Evaluating the proposed registration model Hg-I2P
It is achieved by first using test_anyscene.py and then using eval_anyscene.py

**Selection of other dataset**: dataset.py and anyscene.py can select the different dataset loaders.
It can be used for cross-domain and cross-dataset evaluation. 
