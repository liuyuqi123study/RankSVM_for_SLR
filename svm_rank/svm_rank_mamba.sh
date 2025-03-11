for i in 0.005 0.02 0.05 0.1 0.5 1
do
    #./svm_rank_learn -c $i /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/train_feature_modified0.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_0_mamba_$i.dat
    #./svm_rank_learn -c $i /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/train_feature_modified1.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_1_mamba_$i.dat
    #./svm_rank_learn -c $i /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/train_feature_modified2.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_2_mamba_$i.dat
    #./svm_rank_learn -c $i /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/train_feature_modified3.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_3_mamba_$i.dat
    #./svm_rank_learn -c $i /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/train_feature_modified4.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_4_mamba_$i.dat
    ./svm_rank_classify /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/test_feature_modified0.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_0_mamba_$i.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/predictions_0_$i.dat
    ./svm_rank_classify /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/test_feature_modified1.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_1_mamba_$i.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/predictions_1_$i.dat
    ./svm_rank_classify /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/test_feature_modified2.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_2_mamba_$i.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/predictions_2_$i.dat
    ./svm_rank_classify /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/test_feature_modified3.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_3_mamba_$i.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/predictions_3_$i.dat
    ./svm_rank_classify /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/test_feature_modified4.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/model_4_mamba_$i.dat /Users/yuqi/Downloads/RankSVM_for_SLR/RankMamba/features/predictions_4_$i.dat
done
