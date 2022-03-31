# python predict.py --file_out part1_scores.csv \
# --evaluation_folder Predict/spcup_2022_eval_part1 \
# --checkpoint_path checkpoint/ex3_g_00130000  \
# --config config.json 

# python predict.py --file_out part2_scores.csv \
# --evaluation_folder Predict/spcup_2022_eval_part2 \
# --checkpoint_path_unseen checkpoint/ex8_g_00026000  \
# --checkpoint_path checkpoint/ex9_g_00080000 \
# --config config.json 

python predict.py --file_out part1_scores.csv \
--evaluation_folder Predict/spcup_2022_eval_part1 \
--checkpoint_path Outdir/ex3/g_00100000  \
--config config.json 

python predict.py --file_out part2_scores.csv \
--evaluation_folder Predict/spcup_2022_eval_part2 \
--checkpoint_path_unseen Outdir/ex8/g_00024000  \
--checkpoint_path Outdir/ex12/g_00080000 \
--config config.json 