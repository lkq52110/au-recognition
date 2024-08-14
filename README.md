Start
=
- Python 3.
- Check the required python packages in `requirements.txt`.
```
pip install -r requirements.txt
```

Data
=
The Datasets we used:
  * [DISFA](http://mohammadmahoor.com/disfa-contact-form/)
  * [EmotioNet](http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/)
    (The application website for the EmotioNet dataset appears to be temporarily under maintenance. If you need processed data samples for training (as I am not authorized to provide the original files), contact 📮lkq52110@gmail.com.)

We provide data preprocessing tools in ```tool/``` ,which you can use to process the raw files after downloading the dataset.

You can obtain the weights required for WeightedAsymmetricLoss through the ```tool/calculate_AU_class_weights```, and use the ```tool/image_label_process``` to process the labels of the raw files and set up three-fold cross-validation.

If you need to perform face alignment, you can use the "MTCNN" or "dlib" tools for preprocessing in ```tool/face_align```.

Make sure that you download the ImageNet pre-trained [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) model to `checkpoints/` 

Train and Test
=

- to train on DISFA Dataset, run:
```
python train.py --dataset DISFA --experiment DISFA_fold1 -b 16 -lr 0.0001 --fold 1
```

- to train on EmotioNet Dataset, run:
```
python train.py --dataset EmotioNet --experiment EmotioNet_fold1 -b 16 -lr 0.0001 --fold 1
```

- to test the performance on DISFA or EmotioNet Dataset, run:
```
python test.py --dataset DISFA --experiment DISFA_test_fold1 --resume results/DISFA_fold1/bs_16_seed_0_lr_0.0001/xxxx_fold1.pth --fold 1
```
