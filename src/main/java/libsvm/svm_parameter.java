package libsvm;

public class svm_parameter implements Cloneable, java.io.Serializable {
    /* svm_type */
    public static final int C_SVC = 0;
    public static final int NU_SVC = 1;
    public static final int ONE_CLASS = 2;
    public static final int EPSILON_SVR = 3;
    public static final int NU_SVR = 4;

    /* kernel_type */
    public static final int LINEAR = 0;     //线性核
    public static final int POLY = 1;       //多项式核
    public static final int RBF = 2;        //径向基核(高斯核)
    public static final int SIGMOID = 3;    //sigmoid核，tanh核
    public static final int PRECOMPUTED = 4;//自定义核

    public int svm_type;        //SVM模型类别
    public int kernel_type;     //核函数类别
    public int degree;          //多项式核的d
    public double gamma;        //多项式核、高斯核、sigmoid核的gamma
    public double coef0;        //多项式核、sigmoid核的c

    // these are for training only
    public double cache_size;   //缓存大小，单位MB
    public double eps;          //SVM边界允许误差
    public double C;            //参数C，只有C_SVC、EPSILON_SVR、NU_SVR有用
    public int nr_weight;       //权重数量，C_SVC
    public int[] weight_label;  //权重标签、C_SVC
    public double[] weight;     //权重值、C_SVC
    public double nu;           // for NU_SVC, ONE_CLASS, and NU_SVR
    public double p;            // for EPSILON_SVR
    public int shrinking;       //收缩启发式标志
    public int probability;     // do probability estimates

    public Object clone() {
        try {
            return super.clone();
        } catch (CloneNotSupportedException e) {
            return null;
        }
    }

}
