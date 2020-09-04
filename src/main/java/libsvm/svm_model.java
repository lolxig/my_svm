//
// svm_model
//
package libsvm;

public class svm_model implements java.io.Serializable {

    public svm_parameter param;    //parameters
    public int nr_class;        //number of classes
    public int l;                //total #SV
    public svm_node[][] SV;        //支持向量
    public double[][] sv_coef;    //coefficients for SVs in decision functions
    public double[] rho;        //b of the decision function(s) wx+b
    public double[] probA;      //pairwise probability information; empty if -b 0 or in one-class SVM
    public double[] probB;        //pairwise probability information; empty if -b 0 or in one-class SVM
    public int[] sv_indices;    //指示支持向量在训练集中的位置

    // for classification only

    public int[] label;        //label of each class; empty for regression/one-class SVM
    public int[] nSV;        //number of SVs for each class; empty for regression/one-class SVM
    //nSV[0] + nSV[1] + ... + nSV[k-1] = l
}
