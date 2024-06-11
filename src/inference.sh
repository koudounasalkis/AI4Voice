export cuda_device=0

################################
##           AVFAD            ##
################################

echo "avfad - inference"
echo "----------"

CUDA_VISIBLE_DEVICES=$cuda_device python inference.py \
    --df_train data_avfad/df_train.csv \
    --df_test data_avfad/df_test.csv \
    --model results_avfad/category/hub_base_sentences/checkpoint-2500 \
    --output_dir results_avfad/hub_base_sentences/ \
    --label s/p \
    --save_confidence_scores