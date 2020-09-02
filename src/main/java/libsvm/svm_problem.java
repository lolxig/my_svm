package libsvm;

public class svm_problem implements java.io.Serializable {
    public int l;    //数据集大小
    public double[] y;    //标签集
    public svm_node[][] x;    //特征集
}
