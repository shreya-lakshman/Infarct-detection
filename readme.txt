1. Make sure the  CLEANED_DATA folder, model.py, brightness_augmentation.py,  gaussian_augmentation.py,numpy_array_creation.py,Training_Data_withoutaugmentation  and Inference_Data are in the same folder. 
2. Run numpy_array_creation.py, which will create the X1_train_5_aug.pickle, X_test_12.pickle, Y1_train_5_aug.pickle and Y_test_12 pickle files. 
3. Run model.py on the command prompt to see the accuracy for each epoch. The model is saved and a frozen_model_final_new.pb file is created as well. 
4. Set up the OpenVino environment by running setupvars.bat.
5. Run the following command to generate the .xml,.bin and .mapping files for the frozen model, namely frozen_model_final_new(.xml, .bin, .mapping file)
python mo_tf.py  --input_model <path of .pb file>  --output_dir <destination path for .xml>  --input_shape [1,100,100,3] --data_type FP32
6. Once the IR files are generated, run the following command to test the inference images on the model.
python openvino_inference.py -m <path of frozen_model_final_new.xml> -i <path of image whose inference is to be generated>