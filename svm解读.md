### libsvm整体框架

```
Training:

parse_command_line
    从命令行读入需要设置的参数，并初始化各参数
read_problem
    读入训练数据
svm_check_parameter
    检查参数和训练数据是否符合规定的格式
do_cross_validation
    svm_cross_validation
        对训练数据进行均匀划分，使得同一个类别的数据均匀分布在各个fold中
        最终得到的数组中，相同fold的数据的索引放在连续内存空间数组中
        Cross Validation（k fold）
            svm_train（使用k-1个fold的数据做训练）
            svm_predict（使用剩下的1个fold的数据做预测）
        得到每个样本在Cross Validation中的预测类别
    得出Cross Validation的准确率
svm_train
    classification
        svm_group_classes
            对训练数据按类别进行分组重排，相同类别的数据的索引放在连续内存空间数组中
            得到类别总数：nr_class，
            每个类别的标识：label，
            每个类别的样本数：count，
            每个类别在重排数组中的起始位置：start，
            重排后的索引数组：perm（每个元素的值代表其在原始数组中的索引）
        train k*(k-1)/2 models
            新建1个训练数据子集sub_prob，并使用两个类别的训练数据进行填充
            svm_train_one //根据svm_type的不同，使用不同的方式进行单次模型训练
                solve_c_svc //针对C-SVC这一类型进行模型训练
                    新建1个Solver类对象s
                    s.Solve //使用SMO算法求解对偶问题（二次优化问题）
                        初始化拉格朗日乘子状态向量（是否在边界上）
                        初始化需要参与迭代优化计算的拉格朗日乘子集合
                        初始化梯度G，以及为了重建梯度时候节省计算开销而维护的中间变量G_bar
                        迭代优化
                            do_shrinking（每间隔若干次迭代进行一次shrinking）
                                找出m(a)和M(a)
                                reconstruct_gradient（如果满足停止条件，进行梯度重建）
                                    因为有时候shrinking策略太过aggressive，所以当对shrinking之后的部分变量的优化问题迭代优化到第一次满足停止条件时，便可以对梯度进行重建
                                    接下来的shrinking过程便可以建立在更精确的梯度值上
                                be_shrunk
                                    判断该alpha是否被shrinking（不再参与后续迭代优化）
                                swap_index
                                    交换两个变量的位置，从而使得参与迭代优化的变量（即没有被shrinking的变量）始终保持在变量数组的最前面的连续位置上
                            select_working_set（选择工作集）
                            对工作集的alpha进行更新
                            更新梯度G，拉格朗日乘子状态向量，及中间变量G_bar
                            计算b
                        填充alpha数组和SolutionInfo对象si并返回
                    返回alpha数组和SolutionInfo对象si
                输出decision_function对象（包含alpha数组和b）
            修改nonzero数组，将alpha大于0的对应位置改为true
        填充svm_model对象model
        （包含nr_class，
        label数组，
        b数组，
        probA&probB数组，
        nSV数组，
        l，
        SV二维数组，
        sv_indices数组,
        sv_coef二维数组）并返回 //sv_coef二维数组内元素的放置方式很特别
svm_save_model
    保存模型到制定文件中


​    
Prediction:

svm_predict
    svm_predict_values
        Kernel.k_function //计算预测样本与Support Vectors的Kernel值
        用k(k-1)/2个分类器对预测样本进行预测，得出k(k-1)/2个预测类别
        使用投票策略，选择预测数最多的那个类别作为最终的预测类别
        返回预测类别
```

参考文档：https://www.cnblogs.com/bentuwuying/p/6574620.html