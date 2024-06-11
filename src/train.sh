export cuda_device=0

################################
##           AVFAD            ##
################################

echo "avfad - training"
echo "----------"

CUDA_VISIBLE_DEVICES=$cuda_device python finetuning.py \
    --df_train data_avfad/df_train.csv \
    --df_val data_avfad/df_val.csv \
    --feature_extractor facebook/hubert-base-ls960 \
    --model facebook/hubert-base-ls960 \
    --output_dir results_avfad/category/hub_base_sentences/ \
    --label category \
    --steps 2500 \
    --warmup_steps 200 \
    --df_test data_avfad/df_test.csv \
    --augmentation 


echo "----------"
echo "avfad - training - cv"
echo "----------"
 
CUDA_VISIBLE_DEVICES=$cuda_device python finetuning_cv.py \
    --dataset avfad \
    --feature_extractor facebook/hubert-base-ls960 \
    --model facebook/hubert-base-ls960 \
    --output_dir results_avfad_cv/category/hub_base_sentences/ \
    --label category \
    --steps 2500 \
    --warmup_steps 200 \
    --augmentation
