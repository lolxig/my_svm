package libsvm;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache {
    private final int l; //数据集大小
    private long size;  //缓存大小


    //单块申请到的内存用class head_t来记录所申请内存，并记录长度。而且通过双向的指针，形成链表，增加寻址的速度
    private static final class head_t {
        head_t prev, next;    // a cicular list
        float[] data;
        int len;        // data[0,len) is cached in this entry
    }

    private final head_t[] head;    //类似于变量指针，该指针用来记录程序所申请的内存
    private final head_t lru_head;  //LRU缓存节点

    /**
     * @param l_    数据集size
     * @param size_ 缓存大小
     */
    Cache(int l_, long size_) {
        l = l_;
        size = size_;
        head = new head_t[l]; //双向链表缓存
        for (int i = 0; i < l; i++)
            head[i] = new head_t();
        size /= 4; //将byte空间转换为float空间
        size -= l * (16 / 4);    //扣除L个head_t 的内存数目
        size = Math.max(size, 2 * (long) l);  // cache must be large enough for two columns
        lru_head = new head_t();    //使用LRU缓存技术
        lru_head.next = lru_head.prev = lru_head; //LRU初始化为空
    }

    //删除某节点
    private void lru_delete(head_t h) {
        // delete from current location
        h.prev.next = h.next;
        h.next.prev = h.prev;
    }

    //插入一个几点，尾插法
    private void lru_insert(head_t h) {
        // insert to last position
        h.next = lru_head;
        h.prev = lru_head.prev;
        h.prev.next = h;
        h.next.prev = h;
    }

    // request data [0,len)
    // return some position p where [p,len) need to be filled
    // (p >= len if nothing needs to be filled)
    // java: simulate pointer using single-element array
    int get_data(int index, float[][] data, int len) {
        head_t h = head[index];
        //如果head[index]已经被使用，删掉它，释放这个节点的内存
        if (h.len > 0)
            lru_delete(h);
        //计算需要的内存
        int more = len - h.len;

        //如果将head[index]释放之后，仍然需要更多的内存，则进行分配
        if (more > 0) {
            //如果剩余内存不够了，接着释放下一个节点，直到得到足够的内存
            while (size < more) {
                head_t old = lru_head.next;
                lru_delete(old);
                size += old.len;
                old.data = null;
                old.len = 0;
            }

            //重新分配内存
            float[] new_data = new float[len];
            //将原数据拷贝到新的内存块
            if (h.data != null)
                System.arraycopy(h.data, 0, new_data, 0, h.len);
            h.data = new_data;
            //扣除使用掉的内存
            size -= more;
            {
                int tmp = h.len;
                h.len = len;
                len = tmp;
            }
        }
        //将重新分配后的内存插入链表中
        lru_insert(h);
        data[0] = h.data;
        return len;
    }

    void swap_index(int i, int j) {
        if (i == j) return;

        if (head[i].len > 0) lru_delete(head[i]);
        if (head[j].len > 0) lru_delete(head[j]);
        do {
            float[] tmp = head[i].data;
            head[i].data = head[j].data;
            head[j].data = tmp;
        } while (false);
        do {
            int tmp = head[i].len;
            head[i].len = head[j].len;
            head[j].len = tmp;
        } while (false);
        if (head[i].len > 0) lru_insert(head[i]);
        if (head[j].len > 0) lru_insert(head[j]);

        if (i > j) do {
            int tmp = i;
            i = j;
            j = tmp;
        } while (false);
        for (head_t h = lru_head.next; h != lru_head; h = h.next) {
            if (h.len > i) {
                if (h.len > j)
                    do {
                        float tmp = h.data[i];
                        h.data[i] = h.data[j];
                        h.data[j] = tmp;
                    } while (false);
                else {
                    // give up
                    lru_delete(h);
                    size += h.len;
                    h.data = null;
                    h.len = 0;
                }
            }
        }
    }
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
abstract class QMatrix {

    abstract float[] get_Q(int column, int len);

    abstract double[] get_QD();

    abstract void swap_index(int i, int j);
}

/**
 * Kernel类是用来进行计算Kernel evaluation矩阵的.
 */
abstract class Kernel extends QMatrix {
    private svm_node[][] x; //特征集
    private final double[] x_square;

    // svm_parameter
    private final int kernel_type;  //核函数类型
    private final int degree;       //多项式核的d
    private final double gamma;     //高斯核gamma
    private final double coef0;     //系数

    abstract float[] get_Q(int column, int len);

    abstract double[] get_QD();

    //交换两个节点，暂时不知何用
    void swap_index(int i, int j) {
        {
            svm_node[] tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
        if (x_square != null) {
            double tmp = x_square[i];
            x_square[i] = x_square[j];
            x_square[j] = tmp;
        }
    }

    //计算pow(base, times)
    private static double powi(double base, int times) {
        double tmp = base, ret = 1.0;
        for (int t = times; t > 0; t /= 2) {
            if (t % 2 == 1) ret *= tmp;
            tmp = tmp * tmp;
        }
        return ret;
    }

    /**
     * 求核函数向量.
     *
     * @param i 样本i
     * @param j 样本j
     * @return 核函数向量
     */
    double kernel_function(int i, int j) {
        switch (kernel_type) {
            case svm_parameter.LINEAR:
                return dot(x[i], x[j]);
            case svm_parameter.POLY:
                return powi(gamma * dot(x[i], x[j]) + coef0, degree);
            case svm_parameter.RBF:
                return Math.exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
            case svm_parameter.SIGMOID:
                return Math.tanh(gamma * dot(x[i], x[j]) + coef0);
            case svm_parameter.PRECOMPUTED:
                return x[i][(int) (x[j][0].value)].value;
            default:
                return 0;    // Unreachable
        }
    }

    /**
     * Kernel构造函数.
     *
     * @param l     样本大小
     * @param x_    样本特征集
     * @param param 样本入参
     */
    Kernel(int l, svm_node[][] x_, svm_parameter param) {
        this.kernel_type = param.kernel_type;   //核函数类型
        this.degree = param.degree; //多项式核的d
        this.gamma = param.gamma;   //多项式核、高斯核、sigmoid核的gamma
        this.coef0 = param.coef0;   //多项式核、sigmoid核的c

        x = x_.clone(); //克隆特征集

        //如果是高斯核，计算x的内积，并存放起来
        if (kernel_type == svm_parameter.RBF) {
            x_square = new double[l];
            for (int i = 0; i < l; i++)
                x_square[i] = dot(x[i], x[i]);
        } else
            x_square = null;
    }

    //计算两个向量的内积
    static double dot(svm_node[] x, svm_node[] y) {
        double sum = 0;
        int xlen = x.length;
        int ylen = y.length;
        int i = 0;
        int j = 0;
        while (i < xlen && j < ylen) {
            if (x[i].index == y[j].index)
                sum += x[i++].value * y[j++].value;
            else {
                if (x[i].index > y[j].index)
                    ++j;
                else
                    ++i;
            }
        }
        return sum;
    }

    /**
     * 静态方法，对参数传入的任意2个样本求kernel evaluation。主要应用在predict过程中.
     */
    static double k_function(svm_node[] x, svm_node[] y, svm_parameter param) {
        switch (param.kernel_type) {
            case svm_parameter.LINEAR:
                return dot(x, y);
            case svm_parameter.POLY:
                return powi(param.gamma * dot(x, y) + param.coef0, param.degree);
            case svm_parameter.RBF: {
                double sum = 0;
                int xlen = x.length;
                int ylen = y.length;
                int i = 0;
                int j = 0;
                while (i < xlen && j < ylen) {
                    if (x[i].index == y[j].index) {
                        double d = x[i++].value - y[j++].value;
                        sum += d * d;
                    } else if (x[i].index > y[j].index) {
                        sum += y[j].value * y[j].value;
                        ++j;
                    } else {
                        sum += x[i].value * x[i].value;
                        ++i;
                    }
                }
                while (i < xlen) {
                    sum += x[i].value * x[i].value;
                    ++i;
                }
                while (j < ylen) {
                    sum += y[j].value * y[j].value;
                    ++j;
                }
                return Math.exp(-param.gamma * sum);
            }
            case svm_parameter.SIGMOID:
                return Math.tanh(param.gamma * dot(x, y) + param.coef0);
            case svm_parameter.PRECOMPUTED:  //x: test (validation), y: SV
                return x[(int) (y[0].value)].value;
            default:
                return 0;    // Unreachable
        }
    }
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
    int active_size;    //计算时实际参加运算的样本数目，经过shrinking处理后，该数目会小于全部样本总数
    byte[] y;       //样本所属类别，+1/-1
    double[] G;        //梯度，G(a) = Qa + p
    static final byte LOWER_BOUND = 0;  //内部点a[i]<=0
    static final byte UPPER_BOUND = 1;  //错分点a[i]>=c
    static final byte FREE = 2;         //支持向量0<a[i]<c
    byte[] alpha_status;    //拉格朗日乘子的状态，分别是[内部点LOWER_BOUND], [错分点UPPER_BOUND], [支持向量FREE]
    double[] alpha; //拉格朗日乘子
    QMatrix Q;  //核函数矩阵
    double[] QD;    //核函数矩阵中的对角线部分
    double eps;     //误差极限
    double Cp, Cn;  //正负样本各自的惩罚系数C
    double[] p;     //目标函数中的系数
    int[] active_set;   //计算时实际参加运算的样本索引
    double[] G_bar;     //重建梯度时的中间变量，可以降低重建的计算开销
    int l;      //样本大小
    boolean unshrink;    //没有进行收缩启发式计算的标志

    static final double INF = java.lang.Double.POSITIVE_INFINITY;   //正无穷大

    double get_C(int i) {
        return (y[i] > 0) ? Cp : Cn;
    }

    //更新节点i的alpha状态，区分它是什么点
    void update_alpha_status(int i) {
        if (alpha[i] >= get_C(i))
            alpha_status[i] = UPPER_BOUND;  //错分点
        else if (alpha[i] <= 0)
            alpha_status[i] = LOWER_BOUND;  //内部点
        else
            alpha_status[i] = FREE;        //支持向量
    }

    /**
     * 判断点i是否为错分点.
     */
    boolean is_upper_bound(int i) {
        return alpha_status[i] == UPPER_BOUND;
    }

    /**
     * 判断点i是否为内部点.
     */
    boolean is_lower_bound(int i) {
        return alpha_status[i] == LOWER_BOUND;
    }

    /**
     * 判断点i是否为支持向量.
     */
    boolean is_free(int i) {
        return alpha_status[i] == FREE;
    }

    // java: information about solution except alpha,
    // because we cannot return multiple values otherwise...
    static class SolutionInfo {
        double obj; //obj为SVM文件转换为的二次规划求解得到的最小值
        double rho; //rho为判决函数的偏置项b
        double upper_bound_p;   //不等式约束的剪辑框格
        double upper_bound_n;
        double r;    // for Solver_NU
    }

    /**
     * 完全交换样本 i 和样本 j 的内容，包括申请的内存的地址.
     */
    void swap_index(int i, int j) {
        //交换特征
        Q.swap_index(i, j);
        //交换标签
        {
            byte tmp = y[i];
            y[i] = y[j];
            y[j] = tmp;
        }
        //交换梯度
        {
            double tmp = G[i];
            G[i] = G[j];
            G[j] = tmp;
        }
        //交换alpha状态
        {
            byte tmp = alpha_status[i];
            alpha_status[i] = alpha_status[j];
            alpha_status[j] = tmp;
        }
        //交换alpha
        {
            double tmp = alpha[i];
            alpha[i] = alpha[j];
            alpha[j] = tmp;
        }
        //交换目标函数系数
        {
            double tmp = p[i];
            p[i] = p[j];
            p[j] = tmp;
        }
        //交换活跃状态
        {
            int tmp = active_set[i];
            active_set[i] = active_set[j];
            active_set[j] = tmp;
        }
        //交换梯度缓存
        {
            double tmp = G_bar[i];
            G_bar[i] = G_bar[j];
            G_bar[j] = tmp;
        }
    }

    /**
     * 重建梯度.
     */
    void reconstruct_gradient() {
        //通过G_bar和自由变量重建非活跃元素的梯度

        //如果全是活跃元素，则不需要重建
        if (active_size == l) return;

        //自由变量的个数.
        int nr_free = 0;

        //按系数P重建非活跃元素的梯度
        for (int j = active_size; j < l; j++)
            G[j] = G_bar[j] + p[j];

        //统计活跃元素里面有多少自由变量
        for (int j = 0; j < active_size; j++)
            if (is_free(j))
                nr_free++;

        //若自由变量个数小于活跃元素个数的1/2，则可能不重建梯度会加快训练，因为自由变量一直处于活跃状态
        if (2 * nr_free < active_size)
            svm.info("\nWARNING: using -h 0 may be faster\n");

        //根据自由变量的分布情况重建边界点的梯度
        if (nr_free * l > 2 * active_size * (l - active_size)) {
            for (int i = active_size; i < l; i++) {
                float[] Q_i = Q.get_Q(i, active_size);
                for (int j = 0; j < active_size; j++)
                    if (is_free(j))
                        G[i] += alpha[j] * Q_i[j];
            }
        } else {
            for (int i = 0; i < active_size; i++)
                if (is_free(i)) {
                    float[] Q_i = Q.get_Q(i, l);
                    double alpha_i = alpha[i];
                    for (int j = active_size; j < l; j++)
                        G[j] += alpha_i * Q_i[j];
                }
        }
    }

    /**
     * 使用SMO算法求解对偶问题.
     *
     * @param l         待求数据集大小
     * @param Q         保存了数据集，特征集，输入参数，核向量
     * @param p_        目标函数的系数
     * @param y_        样本所属类别，+1/-1
     * @param alpha_    待计算的alpha值
     * @param Cp        类别i的惩罚系数
     * @param Cn        类别j的惩罚系数
     * @param eps       SVM边界允许误差极限
     * @param si        待求的偏移项的值
     * @param shrinking 收缩启发式标志，去除边界值，减少计算量
     */
    void Solve(int l,
               QMatrix Q,
               double[] p_,
               byte[] y_,
               double[] alpha_,
               double Cp,
               double Cn,
               double eps,
               SolutionInfo si,
               int shrinking) {

        this.l = l; //样本大小
        this.Q = Q; //样本核函数矩阵
        QD = Q.get_QD();    //核函数矩阵中的对角线部分
        p = p_.clone(); //目标函数中的系数
        y = y_.clone(); //样本所属类别，+1/-1
        alpha = alpha_.clone(); //拉格朗日乘子
        this.Cp = Cp;   //类别i的惩罚系数
        this.Cn = Cn;   //类别j的惩罚系数
        this.eps = eps; //SVM边界允许误差极限
        this.unshrink = false;  //是否已经进行过收缩

        //初始化拉格朗日乘子状态
        {
            alpha_status = new byte[l];
            for (int i = 0; i < l; i++)
                update_alpha_status(i);
        }

        //初始化计算时实际参加运算的样本索引，刚开始，所有节点都参与计算
        {
            active_set = new int[l];
            for (int i = 0; i < l; i++)
                active_set[i] = i;
            active_size = l;
        }

        //初始化梯度，梯度 G = Q * alpha + p ，其中p为全-1的向量
        {
            G = new double[l];
            G_bar = new double[l];
            for (int i = 0; i < l; i++) {
                G[i] = p[i];    //G = p
                G_bar[i] = 0;
            }
            for (int i = 0; i < l; i++)
                //若点i不是内部点
                if (!is_lower_bound(i)) {
                    float[] Q_i = Q.get_Q(i, l);
                    double alpha_i = alpha[i];
                    for (int j = 0; j < l; j++)
                        G[j] += alpha_i * Q_i[j];   //G = Q * alpha + p
                    if (is_upper_bound(i))
                        for (int j = 0; j < l; j++)
                            G_bar[j] += get_C(i) * Q_i[j];
                }
        }

        //优化步骤
        int iter = 0;   //已迭代次数
        int max_iter = Math.max(10_000_000, l > Integer.MAX_VALUE / 100 ? Integer.MAX_VALUE : 100 * l); //最大迭代次数
        int counter = Math.min(l, 1000) + 1;    //收缩启发式计算阈值
        int[] working_set = new int[2]; //工作集，大小为2，就选择两个变量，最大违反对

        while (iter < max_iter) {

            //每执行counter次，就进行一次收缩启发式计算
            if (--counter == 0) {
                counter = Math.min(l, 1000);
                if (shrinking != 0)
                    do_shrinking();
                svm.info(".");
            }

            //等于1表示当前参数已经达到最优解，等于0表示选择到了最大违反对
            if (select_working_set(working_set) != 0) {
                //重建整个梯度
                reconstruct_gradient();
                //检查整个样本集
                active_size = l;
                svm.info("*");
                //重建梯度之后对整个样本集进行检查，如果仍然得到最优解，则表示是真的最优解
                if (select_working_set(working_set) != 0)
                    break;
                else
                    counter = 1;    // do shrinking next iteration
            }

            //获取第一个和第二个向量
            int i = working_set[0];
            int j = working_set[1];

            ++iter;

            //根据得到的工作集，来求解alpha[i]和alpha[j]，边界情况谨慎处理
            float[] Q_i = Q.get_Q(i, active_size);
            float[] Q_j = Q.get_Q(j, active_size);

            //求得i和j的惩罚系数
            double C_i = get_C(i);
            double C_j = get_C(j);

            //得到old的拉格朗日乘子
            double old_alpha_i = alpha[i];
            double old_alpha_j = alpha[j];

            //分两种情况，一种是y[i] != y[j]，沿着斜率正方向求解；另一种是y[i] = y[j]，沿着斜率负方向求解
            if (y[i] != y[j]) {
                double quad_coef = QD[i] + QD[j] + 2 * Q_i[j];  //求得K[ii] + K[jj] - 2K[ij]
                if (quad_coef <= 0) //如果小于0，取一个很小很小的值
                    quad_coef = 1e-12;
                double delta = (-G[i] - G[j]) / quad_coef;  //求得梯度增量
                double diff = alpha[i] - alpha[j];
                alpha[i] += delta;  //根据梯度增量来更新两个值
                alpha[j] += delta;

                //如果在边界上，则沿着边界进行剪辑
                if (diff > 0) {
                    if (alpha[j] < 0) {
                        alpha[j] = 0;
                        alpha[i] = diff;
                    }
                } else {
                    if (alpha[i] < 0) {
                        alpha[i] = 0;
                        alpha[j] = -diff;
                    }
                }
                if (diff > C_i - C_j) {
                    if (alpha[i] > C_i) {
                        alpha[i] = C_i;
                        alpha[j] = C_i - diff;
                    }
                } else {
                    if (alpha[j] > C_j) {
                        alpha[j] = C_j;
                        alpha[i] = C_j + diff;
                    }
                }
            } else {
                double quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
                if (quad_coef <= 0)
                    quad_coef = 1e-12;
                double delta = (G[i] - G[j]) / quad_coef;
                double sum = alpha[i] + alpha[j];
                alpha[i] -= delta;
                alpha[j] += delta;

                if (sum > C_i) {
                    if (alpha[i] > C_i) {
                        alpha[i] = C_i;
                        alpha[j] = sum - C_i;
                    }
                } else {
                    if (alpha[j] < 0) {
                        alpha[j] = 0;
                        alpha[i] = sum;
                    }
                }
                if (sum > C_j) {
                    if (alpha[j] > C_j) {
                        alpha[j] = C_j;
                        alpha[i] = sum - C_j;
                    }
                } else {
                    if (alpha[i] < 0) {
                        alpha[i] = 0;
                        alpha[j] = sum;
                    }
                }
            }

            //根据更新后的两个值来更新活跃样本的梯度
            double delta_alpha_i = alpha[i] - old_alpha_i;
            double delta_alpha_j = alpha[j] - old_alpha_j;

            for (int k = 0; k < active_size; k++) {
                G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
            }

            //更新节点的状态和G_bar
            {
                boolean ui = is_upper_bound(i); //判断节点之前是否为错分点
                boolean uj = is_upper_bound(j);
                update_alpha_status(i); //更新节点的状态
                update_alpha_status(j);

                if (ui != is_upper_bound(i)) {  //如果错分点情况发生了变化
                    Q_i = Q.get_Q(i, l);
                    if (ui)                     //如果是纠正了错分点，则梯度正确下降
                        for (int k = 0; k < l; k++)
                            G_bar[k] -= C_i * Q_i[k];
                    else                        //如果是分到了错分点，则梯度反而应该上升
                        for (int k = 0; k < l; k++)
                            G_bar[k] += C_i * Q_i[k];
                }

                if (uj != is_upper_bound(j)) {
                    Q_j = Q.get_Q(j, l);
                    if (uj)
                        for (int k = 0; k < l; k++)
                            G_bar[k] -= C_j * Q_j[k];
                    else
                        for (int k = 0; k < l; k++)
                            G_bar[k] += C_j * Q_j[k];
                }
            }

        } //迭代完成

        if (iter >= max_iter) {
            if (active_size < l) {
                //重建梯度来计算目标值
                reconstruct_gradient();
                active_size = l;
                svm.info("*");
            }
            System.err.print("\nWARNING: reaching max number of iterations\n");
        }

        //计算偏移项b
        si.rho = calculate_rho();

        //计算目标值
        {
            double v = 0;
            for (int i = 0; i < l; i++)
                v += alpha[i] * (G[i] + p[i]);

            si.obj = v / 2;
        }

        //更新得到的计算完毕的alpha值
        for (int i = 0; i < l; i++)
            alpha_[active_set[i]] = alpha[i];

        //更新边界
        si.upper_bound_p = Cp;
        si.upper_bound_n = Cn;

        svm.info("\noptimization finished, #iter = " + iter + "\n");
    }

    /**
     * 通过看停止条件来确定是否当前变量已经处在最优解上了，如果是则返回1，如果不是对working_set数组赋值并返回0.
     * 如果m(a) <= M(a)，则表示得到了最优解，返回1.
     * 为了加快运算，该最优解的范围是在活跃样本集上.
     */
    int select_working_set(int[] working_set) {
        // return i,j such that
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        //    (if quadratic coefficeint <= 0, replace it with tau)
        //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        double Gmax = -INF;     //i的最大梯度
        double Gmax2 = -INF;    //j的最大梯度
        int Gmax_idx = -1;      //i的索引
        int Gmin_idx = -1;      //j的索引
        double obj_diff_min = INF;

        //确定第一个变量，I_up里面的最大值
        for (int t = 0; t < active_size; t++)
            if (y[t] == +1) {   // y = 1
                if (!is_upper_bound(t)) // alpha[t] < C
                    if (-G[t] >= Gmax) {    //取得最大梯度的值和最大梯度的索引
                        Gmax = -G[t];
                        Gmax_idx = t;
                    }
            } else {            // y = -1
                if (!is_lower_bound(t)) // alpha[t] > 0
                    if (G[t] >= Gmax) {     //取得最大梯度的值和最大梯度的索引
                        Gmax = G[t];
                        Gmax_idx = t;
                    }
            }

        //如果取到了i，则获取i在活跃样本下的Q矩阵
        int i = Gmax_idx;
        float[] Q_i = null;
        if (i != -1) // null Q_i not accessed: Gmax=-INF if i=-1
            Q_i = Q.get_Q(i, active_size);

        //在活跃样本范围内获取第二个优化参数
        for (int j = 0; j < active_size; j++) {
            //两种解法同时进行
            if (y[j] == +1) {
                if (!is_lower_bound(j)) {   //y[i] = +1 且alpha[j] > 0
                    if (G[j] >= Gmax2)  //先获取最大梯度
                        Gmax2 = G[j];

                    //如果满足 -y_t * G[t] < -y_i * G[i] && I_low
                    double grad_diff = Gmax + G[j]; //b[ij] = -y[i] * G[i] + y[j] * G[j]
                    if (grad_diff > 0) {
                        double obj_diff;
                        double quad_coef = QD[i] + QD[j] - 2.0 * y[i] * Q_i[j]; //a[ij] = K[ii] + K[jj] - 2K[ij]
                        if (quad_coef > 0)
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        else
                            obj_diff = -(grad_diff * grad_diff) / 1e-12;

                        //获取最小的j值
                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            } else {
                if (!is_upper_bound(j)) {   //y[i] = -1 且alpha[j] < C
                    if (-G[j] >= Gmax2)
                        Gmax2 = -G[j];

                    double grad_diff = Gmax - G[j];
                    if (grad_diff > 0) {
                        double obj_diff;
                        double quad_coef = QD[i] + QD[j] + 2.0 * y[i] * Q_i[j];
                        if (quad_coef > 0)
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        else
                            obj_diff = -(grad_diff * grad_diff) / 1e-12;

                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
        }

        //两种停止条件，①最优化参数已经达到极限误差内；②极限条件下，没能取到第二个参数
        if (Gmax + Gmax2 < eps || Gmin_idx == -1)
            return 1;

        //返回得到的工作集
        working_set[0] = Gmax_idx;
        working_set[1] = Gmin_idx;
        return 0;
    }

    /**
     * 判断是否要进行收缩.
     */
    private boolean be_shrunk(int i, double Gmax1, double Gmax2) {
        if (is_upper_bound(i)) {    //如果是错分点
            if (y[i] == +1)
                return (-G[i] > Gmax1); //错分点的梯度是负数
            else
                return (-G[i] > Gmax2);
        } else if (is_lower_bound(i)) { //如果是自由点
            if (y[i] == +1)
                return (G[i] > Gmax2);
            else
                return (G[i] > Gmax1);
        } else
            return false;
    }

    /**
     * 进行收缩启发式计算.
     * 优化过程中，有一些值可能已经成为边界值，之后便不再变化.
     * 为了节省训练时间，使用shrinking方法去除这些个边界值，从而可以进一步解决一个更小的子优化问题.
     */
    void do_shrinking() {
        double Gmax1 = -INF;        // max { -y_i * grad(f)_i | i in I_up(\alpha) }
        double Gmax2 = -INF;        // max { y_i * grad(f)_i | i in I_low(\alpha) }

        //先找到最大为反对
        for (int i = 0; i < active_size; i++) {
            if (y[i] == +1) {
                if (!is_upper_bound(i)) {
                    if (-G[i] >= Gmax1)
                        Gmax1 = -G[i];
                }
                if (!is_lower_bound(i)) {
                    if (G[i] >= Gmax2)
                        Gmax2 = G[i];
                }
            } else {
                if (!is_upper_bound(i)) {
                    if (-G[i] >= Gmax2)
                        Gmax2 = -G[i];
                }
                if (!is_lower_bound(i)) {
                    if (G[i] >= Gmax1)
                        Gmax1 = G[i];
                }
            }
        }

        //如果没有进行收缩启发式计算且没有达到最优解
        if (!unshrink && Gmax1 + Gmax2 <= eps * 10) {
            unshrink = true;
            //重建梯度
            reconstruct_gradient();
            active_size = l;
            svm.info("*");
        }

        for (int i = 0; i < active_size; i++)
            if (be_shrunk(i, Gmax1, Gmax2)) {   //如果i要进行收缩
                active_size--;
                //遍历从i到最后一个非边界值
                while (active_size > i) {
                    //如果最后一个非边界值不需要进行收缩，则与节点i进行互换
                    if (!be_shrunk(active_size, Gmax1, Gmax2)) {
                        swap_index(i, active_size);
                        break;
                    }
                    //如果需要收缩，则将那个节点给收缩掉
                    active_size--;
                }
            }
    }

    /**
     * 得到新的alpha之后，计算偏移项b.
     */
    double calculate_rho() {
        double r;
        int nr_free = 0;
        double ub = INF, lb = -INF, sum_free = 0;
        for (int i = 0; i < active_size; i++) {
            double yG = y[i] * G[i];

            if (is_upper_bound(i)) {    //错分点
                if (y[i] < 0)
                    ub = Math.min(ub, yG);
                else
                    lb = Math.max(lb, yG);
            } else if (is_lower_bound(i)) { //内部点
                if (y[i] > 0)
                    ub = Math.min(ub, yG);
                else
                    lb = Math.max(lb, yG);
            } else {                //自由点（SV）
                ++nr_free;
                sum_free += yG;
            }
        }

        if (nr_free > 0)
            r = sum_free / nr_free;
        else
            r = (ub + lb) / 2;

        return r;
    }
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
final class Solver_NU extends Solver {
    private SolutionInfo si;

    void Solve(int l, QMatrix Q, double[] p, byte[] y,
               double[] alpha, double Cp, double Cn, double eps,
               SolutionInfo si, int shrinking) {
        this.si = si;
        super.Solve(l, Q, p, y, alpha, Cp, Cn, eps, si, shrinking);
    }

    // return 1 if already optimal, return 0 otherwise
    int select_working_set(int[] working_set) {
        // return i,j such that y_i = y_j and
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        //    (if quadratic coefficeint <= 0, replace it with tau)
        //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        double Gmaxp = -INF;
        double Gmaxp2 = -INF;
        int Gmaxp_idx = -1;

        double Gmaxn = -INF;
        double Gmaxn2 = -INF;
        int Gmaxn_idx = -1;

        int Gmin_idx = -1;
        double obj_diff_min = INF;

        for (int t = 0; t < active_size; t++)
            if (y[t] == +1) {
                if (!is_upper_bound(t))
                    if (-G[t] >= Gmaxp) {
                        Gmaxp = -G[t];
                        Gmaxp_idx = t;
                    }
            } else {
                if (!is_lower_bound(t))
                    if (G[t] >= Gmaxn) {
                        Gmaxn = G[t];
                        Gmaxn_idx = t;
                    }
            }

        int ip = Gmaxp_idx;
        int in = Gmaxn_idx;
        float[] Q_ip = null;
        float[] Q_in = null;
        if (ip != -1) // null Q_ip not accessed: Gmaxp=-INF if ip=-1
            Q_ip = Q.get_Q(ip, active_size);
        if (in != -1)
            Q_in = Q.get_Q(in, active_size);

        for (int j = 0; j < active_size; j++) {
            if (y[j] == +1) {
                if (!is_lower_bound(j)) {
                    double grad_diff = Gmaxp + G[j];
                    if (G[j] >= Gmaxp2)
                        Gmaxp2 = G[j];
                    if (grad_diff > 0) {
                        double obj_diff;
                        double quad_coef = QD[ip] + QD[j] - 2 * Q_ip[j];
                        if (quad_coef > 0)
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        else
                            obj_diff = -(grad_diff * grad_diff) / 1e-12;

                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            } else {
                if (!is_upper_bound(j)) {
                    double grad_diff = Gmaxn - G[j];
                    if (-G[j] >= Gmaxn2)
                        Gmaxn2 = -G[j];
                    if (grad_diff > 0) {
                        double obj_diff;
                        double quad_coef = QD[in] + QD[j] - 2 * Q_in[j];
                        if (quad_coef > 0)
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        else
                            obj_diff = -(grad_diff * grad_diff) / 1e-12;

                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
        }

        if (Math.max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < eps || Gmin_idx == -1)
            return 1;

        if (y[Gmin_idx] == +1)
            working_set[0] = Gmaxp_idx;
        else
            working_set[0] = Gmaxn_idx;
        working_set[1] = Gmin_idx;

        return 0;
    }

    private boolean be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4) {
        if (is_upper_bound(i)) {
            if (y[i] == +1)
                return (-G[i] > Gmax1);
            else
                return (-G[i] > Gmax4);
        } else if (is_lower_bound(i)) {
            if (y[i] == +1)
                return (G[i] > Gmax2);
            else
                return (G[i] > Gmax3);
        } else
            return (false);
    }

    void do_shrinking() {
        double Gmax1 = -INF;    // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
        double Gmax2 = -INF;    // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
        double Gmax3 = -INF;    // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
        double Gmax4 = -INF;    // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

        // find maximal violating pair first
        int i;
        for (i = 0; i < active_size; i++) {
            if (!is_upper_bound(i)) {
                if (y[i] == +1) {
                    if (-G[i] > Gmax1) Gmax1 = -G[i];
                } else if (-G[i] > Gmax4) Gmax4 = -G[i];
            }
            if (!is_lower_bound(i)) {
                if (y[i] == +1) {
                    if (G[i] > Gmax2) Gmax2 = G[i];
                } else if (G[i] > Gmax3) Gmax3 = G[i];
            }
        }

        if (unshrink == false && Math.max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= eps * 10) {
            unshrink = true;
            reconstruct_gradient();
            active_size = l;
        }

        for (i = 0; i < active_size; i++)
            if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4)) {
                active_size--;
                while (active_size > i) {
                    if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4)) {
                        swap_index(i, active_size);
                        break;
                    }
                    active_size--;
                }
            }
    }

    double calculate_rho() {
        int nr_free1 = 0, nr_free2 = 0;
        double ub1 = INF, ub2 = INF;
        double lb1 = -INF, lb2 = -INF;
        double sum_free1 = 0, sum_free2 = 0;

        for (int i = 0; i < active_size; i++) {
            if (y[i] == +1) {
                if (is_upper_bound(i))
                    lb1 = Math.max(lb1, G[i]);
                else if (is_lower_bound(i))
                    ub1 = Math.min(ub1, G[i]);
                else {
                    ++nr_free1;
                    sum_free1 += G[i];
                }
            } else {
                if (is_upper_bound(i))
                    lb2 = Math.max(lb2, G[i]);
                else if (is_lower_bound(i))
                    ub2 = Math.min(ub2, G[i]);
                else {
                    ++nr_free2;
                    sum_free2 += G[i];
                }
            }
        }

        double r1, r2;
        if (nr_free1 > 0)
            r1 = sum_free1 / nr_free1;
        else
            r1 = (ub1 + lb1) / 2;

        if (nr_free2 > 0)
            r2 = sum_free2 / nr_free2;
        else
            r2 = (ub2 + lb2) / 2;

        si.r = (r1 + r2) / 2;
        return (r1 - r2) / 2;
    }
}

//
// Q matrices for various formulations
//
class SVC_Q extends Kernel {
    private final byte[] y; //特征集
    private final Cache cache;  //缓存数据
    private final double[] QD;  //核函数向量，即gram矩阵

    /**
     * Q矩阵.
     *
     * @param prob  问题数据集
     * @param param 参数集
     * @param y_    归一化之后的y，只能是+1和-1
     */
    SVC_Q(svm_problem prob, svm_parameter param, byte[] y_) {
        //保存数据集大小，保存数据集，保存输入参数
        super(prob.l, prob.x, param);
        y = y_.clone(); //克隆特征集
        cache = new Cache(prob.l, (long) (param.cache_size * (1 << 20)));   //申请缓存，默认param.cache_size为100，则默认缓存大小为100MB
        //构建核函数向量，即K(xi, xj)
        QD = new double[prob.l];
        for (int i = 0; i < prob.l; i++)
            QD[i] = kernel_function(i, i);
    }

    //在活跃样本上计算Q = sum yi*y*K(xi, xj)，len
    float[] get_Q(int i, int len) {
        float[][] data = new float[1][];
        int start;
        if ((start = cache.get_data(i, data, len)) < len) {
            for (int j = start; j < len; j++)
                data[0][j] = (float) (y[i] * y[j] * kernel_function(i, j));
        }
        return data[0];
    }

    double[] get_QD() {
        return QD;
    }

    void swap_index(int i, int j) {
        cache.swap_index(i, j);
        super.swap_index(i, j);
        {
            byte tmp = y[i];
            y[i] = y[j];
            y[j] = tmp;
        }
        double tmp = QD[i];
        QD[i] = QD[j];
        QD[j] = tmp;
    }
}

/**
 * 一分类模型.
 */
class ONE_CLASS_Q extends Kernel {

    private final Cache cache;  //数据缓存
    private final double[] QD;  //对角核向量

    ONE_CLASS_Q(svm_problem prob, svm_parameter param) {
        super(prob.l, prob.x, param);
        cache = new Cache(prob.l, (long) (param.cache_size * (1 << 20)));   //申请缓存
        QD = new double[prob.l];    //
        for (int i = 0; i < prob.l; i++)
            QD[i] = kernel_function(i, i);
    }

    float[] get_Q(int i, int len) {
        float[][] data = new float[1][];
        int start, j;
        if ((start = cache.get_data(i, data, len)) < len) {
            for (j = start; j < len; j++)
                data[0][j] = (float) kernel_function(i, j);
        }
        return data[0];
    }

    double[] get_QD() {
        return QD;
    }

    void swap_index(int i, int j) {
        cache.swap_index(i, j);
        super.swap_index(i, j);
        double tmp = QD[i];
        QD[i] = QD[j];
        QD[j] = tmp;
    }
}

class SVR_Q extends Kernel {
    private final int l;
    private final Cache cache;
    private final byte[] sign;
    private final int[] index;
    private int next_buffer;
    private float[][] buffer;
    private final double[] QD;

    SVR_Q(svm_problem prob, svm_parameter param) {
        super(prob.l, prob.x, param);
        l = prob.l;
        cache = new Cache(l, (long) (param.cache_size * (1 << 20)));
        QD = new double[2 * l];
        sign = new byte[2 * l];
        index = new int[2 * l];
        for (int k = 0; k < l; k++) {
            sign[k] = 1;
            sign[k + l] = -1;
            index[k] = k;
            index[k + l] = k;
            QD[k] = kernel_function(k, k);
            QD[k + l] = QD[k];
        }
        buffer = new float[2][2 * l];
        next_buffer = 0;
    }

    void swap_index(int i, int j) {
        {
            byte tmp = sign[i];
            sign[i] = sign[j];
            sign[j] = tmp;
        }
        {
            int tmp = index[i];
            index[i] = index[j];
            index[j] = tmp;
        }
        double tmp = QD[i];
        QD[i] = QD[j];
        QD[j] = tmp;
    }

    float[] get_Q(int i, int len) {
        float[][] data = new float[1][];
        int j, real_i = index[i];
        if (cache.get_data(real_i, data, l) < l) {
            for (j = 0; j < l; j++)
                data[0][j] = (float) kernel_function(real_i, j);
        }

        // reorder and copy
        float[] buf = buffer[next_buffer];
        next_buffer = 1 - next_buffer;
        byte si = sign[i];
        for (j = 0; j < len; j++)
            buf[j] = (float) si * sign[j] * data[0][index[j]];
        return buf;
    }

    double[] get_QD() {
        return QD;
    }
}

public class svm {
    //
    // construct and solve various formulations
    //
    public static final int LIBSVM_VERSION = 324;

    public static final Random rand = new Random();

    private static final svm_print_interface svm_print_stdout = System.out::print;

    private static svm_print_interface svm_print_string = svm_print_stdout;

    static void info(String s) {
        svm_print_string.print(s);
    }


    /**
     * 训练单个C_SVC模型.
     *
     * @param prob  子数据集
     * @param param 输入参数
     * @param alpha 需要输出的alpha值
     * @param si    需要输出的偏置项等参数
     * @param Cp    类别i的惩罚值C
     * @param Cn    类别j的惩罚值C
     */
    private static void solve_c_svc(svm_problem prob,
                                    svm_parameter param,
                                    double[] alpha,
                                    Solver.SolutionInfo si,
                                    double Cp,
                                    double Cn) {
        //获取子数据集大小
        int l = prob.l;
        double[] minus_ones = new double[l];    //-1
        byte[] y = new byte[l];

        //填充alpha初始值和-1的初始值，这里存在重复填充的问题
        for (int i = 0; i < l; i++) {
            alpha[i] = 0;
            minus_ones[i] = -1;
            if (prob.y[i] > 0)
                y[i] = +1;
            else
                y[i] = -1;
        }

        //将参数输入，并进行训练
        Solver s = new Solver();
        //使用SMO算法求解对偶问题
        s.Solve(l, new SVC_Q(prob, param, y), minus_ones, y, alpha, Cp, Cn, param.eps, si, param.shrinking);

        //alpha对权重的均值
        double sum_alpha = 0;
        for (int i = 0; i < l; i++)
            sum_alpha += alpha[i];

        if (Cp == Cn)
            svm.info("nu = " + sum_alpha / (Cp * prob.l) + "\n");

        //得到alpha*y
        for (int i = 0; i < l; i++)
            alpha[i] *= y[i];
    }

    private static void solve_nu_svc(svm_problem prob, svm_parameter param,
                                     double[] alpha, Solver.SolutionInfo si) {
        int i;
        int l = prob.l;
        double nu = param.nu;

        byte[] y = new byte[l];

        for (i = 0; i < l; i++)
            if (prob.y[i] > 0)
                y[i] = +1;
            else
                y[i] = -1;

        double sum_pos = nu * l / 2;
        double sum_neg = nu * l / 2;

        for (i = 0; i < l; i++)
            if (y[i] == +1) {
                alpha[i] = Math.min(1.0, sum_pos);
                sum_pos -= alpha[i];
            } else {
                alpha[i] = Math.min(1.0, sum_neg);
                sum_neg -= alpha[i];
            }

        double[] zeros = new double[l];

        for (i = 0; i < l; i++)
            zeros[i] = 0;

        Solver_NU s = new Solver_NU();
        s.Solve(l, new SVC_Q(prob, param, y), zeros, y,
                alpha, 1.0, 1.0, param.eps, si, param.shrinking);
        double r = si.r;

        svm.info("C = " + 1 / r + "\n");

        for (i = 0; i < l; i++)
            alpha[i] *= y[i] / r;

        si.rho /= r;
        si.obj /= (r * r);
        si.upper_bound_p = 1 / r;
        si.upper_bound_n = 1 / r;
    }

    /**
     * 训练一分类模型.
     *
     * @param prob  待解决的问题
     * @param param 模型输入参数
     * @param alpha 待求参数
     * @param si    待求参数
     */
    private static void solve_one_class(svm_problem prob,
                                        svm_parameter param,
                                        double[] alpha,
                                        Solver.SolutionInfo si) {
        int l = prob.l; //样本个数
        double[] zeros = new double[l]; //0向量
        byte[] ones = new byte[l];  //1向量

        int n = (int) (param.nu * prob.l);    // # of alpha's at upper bound

        for (int i = 0; i < n; i++)
            alpha[i] = 1;
        if (n < prob.l) //正例系数
            alpha[n] = param.nu * prob.l - n;
        for (int i = n + 1; i < l; i++)
            alpha[i] = 0;

        for (int i = 0; i < l; i++) {
            zeros[i] = 0;
            ones[i] = 1;
        }

        Solver s = new Solver();
        s.Solve(l, new ONE_CLASS_Q(prob, param), zeros, ones, alpha, 1.0, 1.0, param.eps, si, param.shrinking);
    }

    private static void solve_epsilon_svr(svm_problem prob, svm_parameter param,
                                          double[] alpha, Solver.SolutionInfo si) {
        int l = prob.l;
        double[] alpha2 = new double[2 * l];
        double[] linear_term = new double[2 * l];
        byte[] y = new byte[2 * l];
        int i;

        for (i = 0; i < l; i++) {
            alpha2[i] = 0;
            linear_term[i] = param.p - prob.y[i];
            y[i] = 1;

            alpha2[i + l] = 0;
            linear_term[i + l] = param.p + prob.y[i];
            y[i + l] = -1;
        }

        Solver s = new Solver();
        s.Solve(2 * l, new SVR_Q(prob, param), linear_term, y,
                alpha2, param.C, param.C, param.eps, si, param.shrinking);

        double sum_alpha = 0;
        for (i = 0; i < l; i++) {
            alpha[i] = alpha2[i] - alpha2[i + l];
            sum_alpha += Math.abs(alpha[i]);
        }
        svm.info("nu = " + sum_alpha / (param.C * l) + "\n");
    }

    private static void solve_nu_svr(svm_problem prob, svm_parameter param,
                                     double[] alpha, Solver.SolutionInfo si) {
        int l = prob.l;
        double C = param.C;
        double[] alpha2 = new double[2 * l];
        double[] linear_term = new double[2 * l];
        byte[] y = new byte[2 * l];
        int i;

        double sum = C * param.nu * l / 2;
        for (i = 0; i < l; i++) {
            alpha2[i] = alpha2[i + l] = Math.min(sum, C);
            sum -= alpha2[i];

            linear_term[i] = -prob.y[i];
            y[i] = 1;

            linear_term[i + l] = prob.y[i];
            y[i + l] = -1;
        }

        Solver_NU s = new Solver_NU();
        s.Solve(2 * l, new SVR_Q(prob, param), linear_term, y,
                alpha2, C, C, param.eps, si, param.shrinking);

        svm.info("epsilon = " + (-si.r) + "\n");

        for (i = 0; i < l; i++)
            alpha[i] = alpha2[i] - alpha2[i + l];
    }

    //
    // decision_function
    //
    static class decision_function {
        double[] alpha;
        double rho;
    }

    /**
     * 训练单个决策参数.
     *
     * @param prob  排列好的数据集
     * @param param 模型参数
     * @param Cp    第一个类的权重
     * @param Cn    第二个类的权重
     * @param latch 多线程执行参数
     * @return 训练好的决策参数
     */
    static decision_function svm_train_one(svm_problem prob,
                                           svm_parameter param,
                                           double Cp,
                                           double Cn,
                                           CountDownLatch latch) {
        decision_function f = svm_train_one(prob, param, Cp, Cn);
        latch.countDown();
        return f;
    }

    /**
     * 训练单个决策参数.
     */
    static decision_function svm_train_one(svm_problem prob,
                                           svm_parameter param,
                                           double Cp,
                                           double Cn) {
        //待求的alpha值
        double[] alpha = new double[prob.l];
        //待求的偏移项的值
        Solver.SolutionInfo si = new Solver.SolutionInfo();
        //根据传入的参数选择训练模型
        switch (param.svm_type) {
            case svm_parameter.C_SVC:
                solve_c_svc(prob, param, alpha, si, Cp, Cn);
                break;
            case svm_parameter.NU_SVC:
                solve_nu_svc(prob, param, alpha, si);
                break;
            case svm_parameter.ONE_CLASS:
                solve_one_class(prob, param, alpha, si);
                break;
            case svm_parameter.EPSILON_SVR:
                solve_epsilon_svr(prob, param, alpha, si);
                break;
            case svm_parameter.NU_SVR:
                solve_nu_svr(prob, param, alpha, si);
                break;
        }

        //打印得到的参数
        //obj 将SVM转换为二次规划求得的最小值
        //rho 判决函数的偏置项b
        svm.info("obj = " + si.obj + ", rho = " + si.rho + "\n");

        //输出支持向量
        int nSV = 0;    //标准支持向量个数(0<a[i]<c)
        int nBSV = 0;   //边界上的支持向量个数(a[i]=c)
        for (int i = 0; i < prob.l; i++) {
            if (Math.abs(alpha[i]) > 0) {
                ++nSV;
                if (prob.y[i] > 0) {
                    //如果点超出边界，则将其剪辑到边界上
                    if (Math.abs(alpha[i]) >= si.upper_bound_p)
                        ++nBSV;
                } else {
                    if (Math.abs(alpha[i]) >= si.upper_bound_n)
                        ++nBSV;
                }
            }
        }

        //打印支持向量信息
        svm.info("nSV = " + nSV + ", nBSV = " + nBSV + "\n");

        //保存训练参数
        decision_function f = new decision_function();
        f.alpha = alpha;
        f.rho = si.rho;
        return f;
    }

    // Platt's binary SVM Probablistic Output: an improvement from Lin et al.
    private static void sigmoid_train(int l, double[] dec_values, double[] labels,
                                      double[] probAB) {
        double A, B;
        double prior1 = 0, prior0 = 0;
        int i;

        for (i = 0; i < l; i++)
            if (labels[i] > 0) prior1 += 1;
            else prior0 += 1;

        int max_iter = 100;    // Maximal number of iterations
        double min_step = 1e-10;    // Minimal step taken in line search
        double sigma = 1e-12;    // For numerically strict PD of Hessian
        double eps = 1e-5;
        double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
        double loTarget = 1 / (prior0 + 2.0);
        double[] t = new double[l];
        double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
        double newA, newB, newf, d1, d2;
        int iter;

        // Initial Point and Initial Fun Value
        A = 0.0;
        B = Math.log((prior0 + 1.0) / (prior1 + 1.0));
        double fval = 0.0;

        for (i = 0; i < l; i++) {
            if (labels[i] > 0) t[i] = hiTarget;
            else t[i] = loTarget;
            fApB = dec_values[i] * A + B;
            if (fApB >= 0)
                fval += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
            else
                fval += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
        }
        for (iter = 0; iter < max_iter; iter++) {
            // Update Gradient and Hessian (use H' = H + sigma I)
            h11 = sigma; // numerically ensures strict PD
            h22 = sigma;
            h21 = 0.0;
            g1 = 0.0;
            g2 = 0.0;
            for (i = 0; i < l; i++) {
                fApB = dec_values[i] * A + B;
                if (fApB >= 0) {
                    p = Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
                    q = 1.0 / (1.0 + Math.exp(-fApB));
                } else {
                    p = 1.0 / (1.0 + Math.exp(fApB));
                    q = Math.exp(fApB) / (1.0 + Math.exp(fApB));
                }
                d2 = p * q;
                h11 += dec_values[i] * dec_values[i] * d2;
                h22 += d2;
                h21 += dec_values[i] * d2;
                d1 = t[i] - p;
                g1 += dec_values[i] * d1;
                g2 += d1;
            }

            // Stopping Criteria
            if (Math.abs(g1) < eps && Math.abs(g2) < eps)
                break;

            // Finding Newton direction: -inv(H') * g
            det = h11 * h22 - h21 * h21;
            dA = -(h22 * g1 - h21 * g2) / det;
            dB = -(-h21 * g1 + h11 * g2) / det;
            gd = g1 * dA + g2 * dB;


            stepsize = 1;        // Line Search
            while (stepsize >= min_step) {
                newA = A + stepsize * dA;
                newB = B + stepsize * dB;

                // New function value
                newf = 0.0;
                for (i = 0; i < l; i++) {
                    fApB = dec_values[i] * newA + newB;
                    if (fApB >= 0)
                        newf += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
                    else
                        newf += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
                }
                // Check sufficient decrease
                if (newf < fval + 0.0001 * stepsize * gd) {
                    A = newA;
                    B = newB;
                    fval = newf;
                    break;
                } else
                    stepsize = stepsize / 2.0;
            }

            if (stepsize < min_step) {
                svm.info("Line search fails in two-class probability estimates\n");
                break;
            }
        }

        if (iter >= max_iter)
            svm.info("Reaching maximal iterations in two-class probability estimates\n");
        probAB[0] = A;
        probAB[1] = B;
    }

    private static double sigmoid_predict(double decision_value, double A, double B) {
        double fApB = decision_value * A + B;
        // 1-p used later; avoid catastrophic cancellation
        if (fApB >= 0)
            return Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
        else
            return 1.0 / (1 + Math.exp(fApB));
    }

    // Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
    private static void multiclass_probability(int k, double[][] r, double[] p) {
        int t, j;
        int iter = 0, max_iter = Math.max(100, k);
        double[][] Q = new double[k][k];
        double[] Qp = new double[k];
        double pQp, eps = 0.005 / k;

        for (t = 0; t < k; t++) {
            p[t] = 1.0 / k;  // Valid if k = 1
            Q[t][t] = 0;
            for (j = 0; j < t; j++) {
                Q[t][t] += r[j][t] * r[j][t];
                Q[t][j] = Q[j][t];
            }
            for (j = t + 1; j < k; j++) {
                Q[t][t] += r[j][t] * r[j][t];
                Q[t][j] = -r[j][t] * r[t][j];
            }
        }
        for (iter = 0; iter < max_iter; iter++) {
            // stopping condition, recalculate QP,pQP for numerical accuracy
            pQp = 0;
            for (t = 0; t < k; t++) {
                Qp[t] = 0;
                for (j = 0; j < k; j++)
                    Qp[t] += Q[t][j] * p[j];
                pQp += p[t] * Qp[t];
            }
            double max_error = 0;
            for (t = 0; t < k; t++) {
                double error = Math.abs(Qp[t] - pQp);
                if (error > max_error)
                    max_error = error;
            }
            if (max_error < eps) break;

            for (t = 0; t < k; t++) {
                double diff = (-Qp[t] + pQp) / Q[t][t];
                p[t] += diff;
                pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
                for (j = 0; j < k; j++) {
                    Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
                    p[j] /= (1 + diff);
                }
            }
        }
        if (iter >= max_iter)
            svm.info("Exceeds max_iter in multiclass_prob\n");
    }

    // Cross-validation decision values for probability estimates
    //概率估计的交叉验证决策值
    private static void svm_binary_svc_probability(svm_problem prob, svm_parameter param, double Cp, double Cn, double[] probAB) {
        int i;
        int nr_fold = 5;
        int[] perm = new int[prob.l];
        double[] dec_values = new double[prob.l];

        // random shuffle
        for (i = 0; i < prob.l; i++) perm[i] = i;
        for (i = 0; i < prob.l; i++) {
            int j = i + rand.nextInt(prob.l - i);
            do {
                int tmp = perm[i];
                perm[i] = perm[j];
                perm[j] = tmp;
            } while (false);
        }
        for (i = 0; i < nr_fold; i++) {
            int begin = i * prob.l / nr_fold;
            int end = (i + 1) * prob.l / nr_fold;
            int j, k;
            svm_problem subprob = new svm_problem();

            subprob.l = prob.l - (end - begin);
            subprob.x = new svm_node[subprob.l][];
            subprob.y = new double[subprob.l];

            k = 0;
            for (j = 0; j < begin; j++) {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            for (j = end; j < prob.l; j++) {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            int p_count = 0, n_count = 0;
            for (j = 0; j < k; j++)
                if (subprob.y[j] > 0)
                    p_count++;
                else
                    n_count++;

            if (p_count == 0 && n_count == 0)
                for (j = begin; j < end; j++)
                    dec_values[perm[j]] = 0;
            else if (p_count > 0 && n_count == 0)
                for (j = begin; j < end; j++)
                    dec_values[perm[j]] = 1;
            else if (p_count == 0 && n_count > 0)
                for (j = begin; j < end; j++)
                    dec_values[perm[j]] = -1;
            else {
                svm_parameter subparam = (svm_parameter) param.clone();
                subparam.probability = 0;
                subparam.C = 1.0;
                subparam.nr_weight = 2;
                subparam.weight_label = new int[2];
                subparam.weight = new double[2];
                subparam.weight_label[0] = +1;
                subparam.weight_label[1] = -1;
                subparam.weight[0] = Cp;
                subparam.weight[1] = Cn;
                svm_model submodel = svm_train(subprob, subparam);
                for (j = begin; j < end; j++) {
                    double[] dec_value = new double[1];
                    svm_predict_values(submodel, prob.x[perm[j]], dec_value);
                    dec_values[perm[j]] = dec_value[0];
                    // ensure +1 -1 order; reason not using CV subroutine
                    dec_values[perm[j]] *= submodel.label[0];
                }
            }
        }
        sigmoid_train(prob.l, dec_values, prob.y, probAB);
    }

    // Return parameter of a Laplace distribution
    private static double svm_svr_probability(svm_problem prob, svm_parameter param) {
        int i;
        int nr_fold = 5;
        double[] ymv = new double[prob.l];
        double mae = 0;

        svm_parameter newparam = (svm_parameter) param.clone();
        newparam.probability = 0;
        svm_cross_validation(prob, newparam, nr_fold, ymv);
        for (i = 0; i < prob.l; i++) {
            ymv[i] = prob.y[i] - ymv[i];
            mae += Math.abs(ymv[i]);
        }
        mae /= prob.l;
        double std = Math.sqrt(2 * mae * mae);
        int count = 0;
        mae = 0;
        for (i = 0; i < prob.l; i++)
            if (Math.abs(ymv[i]) > 5 * std)
                count = count + 1;
            else
                mae += Math.abs(ymv[i]);
        mae /= (prob.l - count);
        svm.info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + mae + "\n");
        return mae;
    }

    /**
     * 将训练数据按同一类别分组.
     *
     * @param prob         输入的数据集与特征集
     * @param nr_class_ret 类别的数量
     * @param label_ret    标签
     * @param start_ret    每个标签开始的位置
     * @param count_ret    每个标签的样本数量
     * @param perm         原数据集按标签排序的结果
     */
    private static void svm_group_classes(svm_problem prob,
                                          int[] nr_class_ret,
                                          int[][] label_ret,
                                          int[][] start_ret,
                                          int[][] count_ret,
                                          int[] perm) {
        int l = prob.l;
        int max_nr_class = 16;  //初始最大支持16个类别
        int nr_class = 0;   //类别的个数
        int[] label = new int[max_nr_class];    //每个类的标签
        int[] count = new int[max_nr_class];    //每个类的数量
        int[] data_label = new int[l];  //转化标签，转换成0开头的

        for (int i = 0; i < l; i++) {
            int this_label = (int) (prob.y[i]);
            int j;
            //对类的个数进行统计，最多统计16个类
            for (j = 0; j < nr_class; j++) {
                if (this_label == label[j]) {
                    ++count[j];
                    break;
                }
            }
            //将类标签进行转化，转化为0开头的形式
            data_label[i] = j;
            //新加一个类
            if (j == nr_class) {
                //如果类别个数超出，则进行扩容
                if (nr_class == max_nr_class) {
                    max_nr_class *= 2;
                    int[] new_data = new int[max_nr_class];
                    System.arraycopy(label, 0, new_data, 0, label.length);
                    label = new_data;
                    new_data = new int[max_nr_class];
                    System.arraycopy(count, 0, new_data, 0, count.length);
                    count = new_data;
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }

        //如果类别只有两个，并且它们的标签为1和-1，转换成0和1
        if (nr_class == 2 && label[0] == -1 && label[1] == +1) {
            {
                int tmp = label[0];
                label[0] = label[1];
                label[1] = tmp;
            }
            {
                int tmp = count[0];
                count[0] = count[1];
                count[1] = tmp;
            }
            for (int i = 0; i < l; i++)
                if (data_label[i] == 0)
                    data_label[i] = 1;
                else
                    data_label[i] = 0;

        }

        //记录每个类的开始位置
        int[] start = new int[nr_class];
        start[0] = 0;
        for (int i = 1; i < nr_class; i++)
            start[i] = start[i - 1] + count[i - 1];
        //按照类别排序
        for (int i = 0; i < l; i++) {
            perm[start[data_label[i]]] = i;
            ++start[data_label[i]];
        }
        start[0] = 0;
        for (int i = 1; i < nr_class; i++)
            start[i] = start[i - 1] + count[i - 1];

        nr_class_ret[0] = nr_class;
        label_ret[0] = label;
        start_ret[0] = start;
        count_ret[0] = count;
    }

    static class ExecuteTrain implements Runnable {

        private final decision_function[] f;
        private final int p;
        private final svm_problem sub_prob;
        private final svm_parameter param;
        private final double Cp;
        private final double Cn;
        private final CountDownLatch latch;
        private final int ci;
        private final int si;
        private final int cj;
        private final int sj;
        private final boolean[] nonzero;

        public ExecuteTrain(decision_function[] f, int p, svm_problem sub_prob, svm_parameter param,
                            double cp, double cn, CountDownLatch latch, int ci, int si, int cj, int sj,
                            boolean[] nonzero) {
            this.f = f;
            this.p = p;
            this.sub_prob = sub_prob;
            this.param = param;
            Cp = cp;
            Cn = cn;
            this.latch = latch;
            this.ci = ci;
            this.si = si;
            this.cj = cj;
            this.sj = sj;
            this.nonzero = nonzero;
        }

        @Override
        public void run() {
            System.out.printf("模型训练线程[%d]开始执行.\n", p);
            long start = System.currentTimeMillis();
            f[p] = svm_train_one(sub_prob, param, Cp, Cn, latch);
            for (int k = 0; k < ci; k++)
                if (!nonzero[si + k] && Math.abs(f[p].alpha[k]) > 0)
                    nonzero[si + k] = true;
            for (int k = 0; k < cj; k++)
                if (!nonzero[sj + k] && Math.abs(f[p].alpha[ci + k]) > 0)
                    nonzero[sj + k] = true;

            System.out.printf("模型训练线程[%d]-----执行完成，耗时：%dms\n", p, System.currentTimeMillis() - start);
        }

    }

    /**
     * 训练svm模型.
     *
     * @param prob  输入的数据集与特征集
     * @param param 输入的模型参数
     * @return 训练好的模型
     */
    public static svm_model svm_train(svm_problem prob, svm_parameter param) {
        //新建一个模型
        svm_model model = new svm_model();
        //保存输入的模型参数
        model.param = param;

        //分类训练
        if (param.svm_type == svm_parameter.ONE_CLASS ||
                param.svm_type == svm_parameter.EPSILON_SVR ||
                param.svm_type == svm_parameter.NU_SVR) {
            //回归模型和ONE-CLASS模型
            model.nr_class = 2;
            model.label = null;
            model.nSV = null;
            model.probA = null;
            model.probB = null;
            model.sv_coef = new double[1][];

            //回归概率预测
//            if (param.probability == 1 &&
//                    (param.svm_type == svm_parameter.EPSILON_SVR ||
//                            param.svm_type == svm_parameter.NU_SVR)) {
//                model.probA = new double[1];
//                model.probA[0] = svm_svr_probability(prob, param);
//            }

            //模型训练
            decision_function f = svm_train_one(prob, param, 0, 0);
            model.rho = new double[1];
            model.rho[0] = f.rho;

            int nSV = 0;    //支持向量个数
            for (int i = 0; i < prob.l; i++)
                if (Math.abs(f.alpha[i]) > 0)
                    ++nSV;

            model.l = nSV;
            model.SV = new svm_node[nSV][];
            model.sv_coef[0] = new double[nSV];
            model.sv_indices = new int[nSV];
            int j = 0;
            for (int i = 0; i < prob.l; i++)
                if (Math.abs(f.alpha[i]) > 0) { //支持向量
                    model.SV[j] = prob.x[i];
                    model.sv_coef[0][j] = f.alpha[i];
                    model.sv_indices[j] = i + 1;
                    ++j;
                }
        } else {
            //分类模型，C_SVC、NU_SVC
            int l = prob.l;
            int[] tmp_nr_class = new int[1];
            int[][] tmp_label = new int[1][];
            int[][] tmp_start = new int[1][];
            int[][] tmp_count = new int[1][];
            int[] perm = new int[l];

            //将训练数据按同一类别分组
            svm_group_classes(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm);
            int nr_class = tmp_nr_class[0]; //类别个数
            int[] label = tmp_label[0];     //类别标签
            int[] start = tmp_start[0];     //每个标签开始的位置
            int[] count = tmp_count[0];     //每个标签对应的数据量

            //如果只有一个类，报错
            if (nr_class == 1)
                svm.info("WARNING: training data in only one class. See README for details.\n");

            //将数据集按照上述的类别排列在一起
            svm_node[][] x = new svm_node[l][];
            for (int i = 0; i < l; i++)
                x[i] = prob.x[perm[i]];

            //计算惩罚系数C
            double[] weighted_C = new double[nr_class];
            for (int i = 0; i < nr_class; i++)
                weighted_C[i] = param.C;

            //nr_weight记录了从命令行输入的需要设置惩罚系数C的标签的数量
            //weight_label记录了从命令行输入的需要设置惩罚系数C的标签值
            //weight记录了从命令行输入的需要设置的惩罚系数C，与weight_label一起使用
            for (int i = 0; i < param.nr_weight; i++) {
                int j;
                for (j = 0; j < nr_class; j++)
                    if (param.weight_label[i] == label[j])
                        break;
                if (j == nr_class)
                    System.err.print("WARNING: class label " + param.weight_label[i] + " specified in weight is not found\n");
                else
                    weighted_C[j] *= param.weight[i];
            }

            //对于多分类问题，采用1-V-1的方式构造分类器，最后判断采用竞争方式
            //训练k*(k-1)/2个模型，nonzero[]为true则为支持向量
            boolean[] nonzero = new boolean[l];
            for (int i = 0; i < l; i++)
                nonzero[i] = false;
            //取k*(k-1)/2个决策函数
            decision_function[] f = new decision_function[nr_class * (nr_class - 1) / 2];

            //做概率估计，暂时不知道是干啥的
//            double[] probA = null, probB = null;
//            if (param.probability == 1) {
//                probA = new double[nr_class * (nr_class - 1) / 2];
//                probB = new double[nr_class * (nr_class - 1) / 2];
//            }

            //训练k*(k-1)/2个模型
//            ExecutorService service = Executors.newFixedThreadPool(nr_class * (nr_class - 1) / 2);
            CountDownLatch latch = new CountDownLatch(nr_class * (nr_class - 1) / 2);
            int p = 0;
            for (int i = 0; i < nr_class; i++) {
                for (int j = i + 1; j < nr_class; j++) {
                    //建立一个副数据集，按照类标签排序，并填充l，x，y，即样本大小，特征集，标签集，排列顺序已经按聚类分开
                    svm_problem sub_prob = new svm_problem();
                    //si为类别i的起始点，sj为类别j的起始点
                    int si = start[i], sj = start[j];
                    //ci为类别i的样本数量，cj为类别j的样本数量
                    int ci = count[i], cj = count[j];

                    //子数据集为两个大小为两个类别的数据集的叠加
                    sub_prob.l = ci + cj;
                    sub_prob.x = new svm_node[sub_prob.l][];
                    sub_prob.y = new double[sub_prob.l];

                    //填充类别i和类别j的数据到子数据集，正例填充+1，反例填充-1
                    for (int k = 0; k < ci; k++) {
                        sub_prob.x[k] = x[si + k];
                        sub_prob.y[k] = +1;
                    }
                    for (int k = 0; k < cj; k++) {
                        sub_prob.x[ci + k] = x[sj + k];
                        sub_prob.y[ci + k] = -1;
                    }

                    //做概率估计，暂时不知道是干啥的
//                    if (param.probability == 1) {
//                        double[] probAB = new double[2];
//                        svm_binary_svc_probability(sub_prob, param, weighted_C[i], weighted_C[j], probAB);
//                        probA[p] = probAB[0];
//                        probB[p] = probAB[1];
//                    }

//                    //针对类别i和类别j训练单个决策参数，主要是训练alpha和b
//                    f[p] = svm_train_one(sub_prob, param, weighted_C[i], weighted_C[j], latch);
//
//                    //修改nonzero数组，将alpha大于0的对应位置改为true，nonzero为true的表示是支持向量
//                    for (int k = 0; k < ci; k++)
//                        if (!nonzero[si + k] && Math.abs(f[p].alpha[k]) > 0)
//                            nonzero[si + k] = true;
//                    for (int k = 0; k < cj; k++)
//                        if (!nonzero[sj + k] && Math.abs(f[p].alpha[ci + k]) > 0)
//                            nonzero[sj + k] = true;

                    //根据选择参数，将子集传入线程，开始训练，并填入训练参数
                    new Thread(new ExecuteTrain(f, p, sub_prob, param, weighted_C[i], weighted_C[j], latch, ci, si, cj, sj, nonzero)).start();
                    ++p;
                }
            }

            try {
                latch.await();
            } catch (InterruptedException e) {
                System.err.println("多线程等待执行出错，请检查！");
                System.exit(1);
            }
            System.out.println("\n所有模型训练完毕，开始保存模型.\n");

            //填充svm_model对象model

            //填充类别数量
            model.nr_class = nr_class;

            //填充每个类的标签
            model.label = new int[nr_class];
            System.arraycopy(label, 0, model.label, 0, nr_class);

            //填充判决函数偏置项b
            model.rho = new double[nr_class * (nr_class - 1) / 2];
            for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
                model.rho[i] = f[i].rho;

//            if (param.probability == 1) {
//                model.probA = new double[nr_class * (nr_class - 1) / 2];
//                model.probB = new double[nr_class * (nr_class - 1) / 2];
//                for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++) {
//                    model.probA[i] = probA[i];
//                    model.probB[i] = probB[i];
//                }
//            } else {
//                model.probA = null;
//                model.probB = null;
//            }

            //支持向量总个数(对于两类来说，因为只有一个分类模型Total nSV = nSV，但是对于多类，这个是各个分类模型的nSV之和)
            int total_sv = 0;
            int[] nz_count = new int[nr_class];
            model.nSV = new int[nr_class];
            for (int i = 0; i < nr_class; i++) {
                int nSV = 0;
                for (int j = 0; j < count[i]; j++)
                    if (nonzero[start[i] + j]) {
                        ++nSV;
                        ++total_sv;
                    }
                model.nSV[i] = nSV;
                nz_count[i] = nSV;
            }

            svm.info("Total nSV = " + total_sv + "\n");

            //模型支持向量总数
            model.l = total_sv;
            model.SV = new svm_node[total_sv][];
            model.sv_indices = new int[total_sv];
            p = 0;
            for (int i = 0; i < l; i++)
                //如果是支持向量，则填充
                if (nonzero[i]) {
                    model.SV[p] = x[i];
                    model.sv_indices[p++] = perm[i] + 1;
                }

            int[] nz_start = new int[nr_class];
            nz_start[0] = 0;
            for (int i = 1; i < nr_class; i++)
                nz_start[i] = nz_start[i - 1] + nz_count[i - 1];

            //填充支持向量系数
            model.sv_coef = new double[nr_class - 1][];
            for (int i = 0; i < nr_class - 1; i++)
                model.sv_coef[i] = new double[total_sv];

            p = 0;
            for (int i = 0; i < nr_class; i++)
                for (int j = i + 1; j < nr_class; j++) {
                    // classifier (i,j): coefficients with
                    // i are in sv_coef[j-1][nz_start[i]...],
                    // j are in sv_coef[i][nz_start[j]...]

                    int si = start[i];
                    int sj = start[j];
                    int ci = count[i];
                    int cj = count[j];

                    int q = nz_start[i];
                    for (int k = 0; k < ci; k++)
                        if (nonzero[si + k])
                            model.sv_coef[j - 1][q++] = f[p].alpha[k];
                    q = nz_start[j];
                    for (int k = 0; k < cj; k++)
                        if (nonzero[sj + k])
                            model.sv_coef[i][q++] = f[p].alpha[ci + k];
                    ++p;
                }
        }
        return model;
    }

    /**
     * 分层交叉验证.
     *
     * @param prob    问题，包含数据集之类的数据
     * @param param   训练参数
     * @param nr_fold 传入的训练的叠数
     * @param target  待填入的预测值
     */
    public static void svm_cross_validation(svm_problem prob, svm_parameter param, int nr_fold, double[] target) {
        int[] fold_start = new int[nr_fold + 1];
        int l = prob.l;
        int[] perm = new int[l];

        // stratified cv may not give leave-one-out rate
        // Each class to l folds -> some folds may have zero elements
        if ((param.svm_type == svm_parameter.C_SVC ||
                param.svm_type == svm_parameter.NU_SVC) && nr_fold < l) {
            //数据聚类操作
            int[] tmp_nr_class = new int[1];
            int[][] tmp_label = new int[1][];
            int[][] tmp_start = new int[1][];
            int[][] tmp_count = new int[1][];

            svm_group_classes(prob, tmp_nr_class, tmp_label, tmp_start, tmp_count, perm);

            int nr_class = tmp_nr_class[0];
            int[] start = tmp_start[0];
            int[] count = tmp_count[0];

            //随机洗牌，然后用perm数组将数据分组折叠使
            int[] fold_count = new int[nr_fold];
            int[] index = new int[l];
            System.arraycopy(perm, 0, index, 0, l);
            for (int c = 0; c < nr_class; c++)
                for (int i = 0; i < count[c]; i++) {
                    //取一个随机数，将数组随机洗牌
                    int j = i + rand.nextInt(count[c] - i);
                    {
                        int tmp = index[start[c] + j];
                        index[start[c] + j] = index[start[c] + i];
                        index[start[c] + i] = tmp;
                    }
                }
            for (int i = 0; i < nr_fold; i++) {
                fold_count[i] = 0;
                for (int c = 0; c < nr_class; c++)
                    fold_count[i] += (i + 1) * count[c] / nr_fold - i * count[c] / nr_fold;
            }
            fold_start[0] = 0;
            for (int i = 1; i <= nr_fold; i++)
                fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
            for (int c = 0; c < nr_class; c++)
                for (int i = 0; i < nr_fold; i++) {
                    int begin = start[c] + i * count[c] / nr_fold;
                    int end = start[c] + (i + 1) * count[c] / nr_fold;
                    for (int j = begin; j < end; j++) {
                        perm[fold_start[i]] = index[j];
                        fold_start[i]++;
                    }
                }
            //每一叠开始的索引
            fold_start[0] = 0;
            for (int i = 1; i <= nr_fold; i++)
                fold_start[i] = fold_start[i - 1] + fold_count[i - 1];
        } else {
            for (int i = 0; i < l; i++) perm[i] = i;
            for (int i = 0; i < l; i++) {
                int j = i + rand.nextInt(l - i);
                {
                    int tmp = perm[i];
                    perm[i] = perm[j];
                    perm[j] = tmp;
                }
            }
            for (int i = 0; i <= nr_fold; i++)
                fold_start[i] = i * l / nr_fold;
        }

        //遍历每一叠数据
        for (int i = 0; i < nr_fold; i++) {
            int begin = fold_start[i];
            int end = fold_start[i + 1];

            svm_problem subprob = new svm_problem();
            subprob.l = l - (end - begin);
            subprob.x = new svm_node[subprob.l][];
            subprob.y = new double[subprob.l];

            //将数据复制到子数据集
            int k = 0;
            for (int j = 0; j < begin; j++) {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            for (int j = end; j < l; j++) {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }

            //将子数据集进行训练
            svm_model submodel = svm_train(subprob, param);
            if (param.probability == 1 &&
                    (param.svm_type == svm_parameter.C_SVC || param.svm_type == svm_parameter.NU_SVC)) {
//                double[] prob_estimates = new double[svm_get_nr_class(submodel)];
//                for (int j = begin; j < end; j++)
//                    target[perm[j]] = svm_predict_probability(submodel, prob.x[perm[j]], prob_estimates);
            } else
                for (int j = begin; j < end; j++)
                    //进行预测，填入预测值
                    target[perm[j]] = svm_predict(submodel, prob.x[perm[j]]);
        }
    }

    /**
     * 获取SVM模型的类型.
     */
    public static int svm_get_svm_type(svm_model model) {
        return model.param.svm_type;
    }

    /**
     * 获取SVM模型的类别个数.
     */
    public static int svm_get_nr_class(svm_model model) {
        return model.nr_class;
    }

    /**
     * 将模型标签存储到{@code label}里面去
     */
    public static void svm_get_labels(svm_model model, int[] label) {
        if (model.label != null)
            if (model.nr_class >= 0)
                System.arraycopy(model.label, 0, label, 0, model.nr_class);
    }

    public static void svm_get_sv_indices(svm_model model, int[] indices) {
        if (model.sv_indices != null)
            for (int i = 0; i < model.l; i++)
                indices[i] = model.sv_indices[i];
    }

    public static int svm_get_nr_sv(svm_model model) {
        return model.l;
    }

    public static double svm_get_svr_probability(svm_model model) {
        if ((model.param.svm_type == svm_parameter.EPSILON_SVR || model.param.svm_type == svm_parameter.NU_SVR) &&
                model.probA != null)
            return model.probA[0];
        else {
            System.err.print("Model doesn't contain information for SVR probability inference\n");
            return 0;
        }
    }

    /**
     * 模型预测，决策函数为  f(x)=sign(wx + b).
     *
     * @param model      模型
     * @param x          样本
     * @param dec_values 决策值向量
     * @return 分类结果
     */
    public static double svm_predict_values(svm_model model, svm_node[] x, double[] dec_values) {
        //one_class或者是回归预测
        if (model.param.svm_type == svm_parameter.ONE_CLASS ||
                model.param.svm_type == svm_parameter.EPSILON_SVR ||
                model.param.svm_type == svm_parameter.NU_SVR) {
            double[] sv_coef = model.sv_coef[0];
            double sum = 0;
            for (int i = 0; i < model.l; i++)
                sum += sv_coef[i] * Kernel.k_function(x, model.SV[i], model.param);
            sum -= model.rho[0];
            dec_values[0] = sum;

            if (model.param.svm_type == svm_parameter.ONE_CLASS)
                return (sum > 0) ? 1 : -1;
            else
                return sum;
            //分类预测
        } else {
            int nr_class = model.nr_class;  //类别个数
            int l = model.l;    //支持向量个数

            double[] kvalue = new double[l];
            //求wx.
            for (int i = 0; i < l; i++)
                kvalue[i] = Kernel.k_function(x, model.SV[i], model.param);

            int[] start = new int[nr_class];
            start[0] = 0;
            //记录每个类别的支持向量开始点
            for (int i = 1; i < nr_class; i++)
                start[i] = start[i - 1] + model.nSV[i - 1];

            int[] vote = new int[nr_class];
            for (int i = 0; i < nr_class; i++)
                vote[i] = 0;

            int p = 0;
            //一共有 nr_class * (nr_class - 1)个决策函数
            for (int i = 0; i < nr_class; i++)
                for (int j = i + 1; j < nr_class; j++) {
                    double sum = 0;
                    int si = start[i];  //类别i的支持向量起始点
                    int sj = start[j];  //类别j的支持向量起始点
                    int ci = model.nSV[i];  //类别i的支持向量个数
                    int cj = model.nSV[j];  //类别j的支持向量个数

                    double[] coef1 = model.sv_coef[j - 1];  //决策系数
                    double[] coef2 = model.sv_coef[i];
                    for (int k = 0; k < ci; k++)
                        sum += coef1[si + k] * kvalue[si + k];  //决策系数乘以wx
                    for (int k = 0; k < cj; k++)
                        sum += coef2[sj + k] * kvalue[sj + k];
                    sum -= model.rho[p];    //减去偏移项b，为什么是减呢？
                    dec_values[p] = sum;    //在模型i与模型j中得到的决策值

                    if (dec_values[p] > 0)  //根据决策值选择哪个模型
                        ++vote[i];
                    else
                        ++vote[j];
                    p++;
                }

            //巧妙的求最大值索引办法，得到决策值最大的即为该样本预测的类
            int vote_max_idx = 0;
            for (int i = 1; i < nr_class; i++)
                if (vote[i] > vote[vote_max_idx])
                    vote_max_idx = i;

            return model.label[vote_max_idx];
        }
    }

    /**
     * 用模型对样本进行预测.
     *
     * @param model 模型参数
     * @param x     样本
     * @return 预测值
     */
    public static double svm_predict(svm_model model, svm_node[] x) {
        int nr_class = model.nr_class;  //类别个数
        double[] dec_values;    //决策值
        //回归预测
        if (model.param.svm_type == svm_parameter.ONE_CLASS ||
                model.param.svm_type == svm_parameter.EPSILON_SVR ||
                model.param.svm_type == svm_parameter.NU_SVR)
            dec_values = new double[1];
            //分类预测
        else
            dec_values = new double[nr_class * (nr_class - 1) / 2];
        return svm_predict_values(model, x, dec_values);
    }

    public static double svm_predict_probability(svm_model model, svm_node[] x, double[] prob_estimates) {
        if ((model.param.svm_type == svm_parameter.C_SVC || model.param.svm_type == svm_parameter.NU_SVC) &&
                model.probA != null && model.probB != null) {
            int i;
            int nr_class = model.nr_class;
            double[] dec_values = new double[nr_class * (nr_class - 1) / 2];
            svm_predict_values(model, x, dec_values);

            double min_prob = 1e-7;
            double[][] pairwise_prob = new double[nr_class][nr_class];

            int k = 0;
            for (i = 0; i < nr_class; i++)
                for (int j = i + 1; j < nr_class; j++) {
                    pairwise_prob[i][j] = Math.min(Math.max(sigmoid_predict(dec_values[k], model.probA[k], model.probB[k]), min_prob), 1 - min_prob);
                    pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
                    k++;
                }
            if (nr_class == 2) {
                prob_estimates[0] = pairwise_prob[0][1];
                prob_estimates[1] = pairwise_prob[1][0];
            } else
                multiclass_probability(nr_class, pairwise_prob, prob_estimates);

            int prob_max_idx = 0;
            for (i = 1; i < nr_class; i++)
                if (prob_estimates[i] > prob_estimates[prob_max_idx])
                    prob_max_idx = i;
            return model.label[prob_max_idx];
        } else
            return svm_predict(model, x);
    }

    //SVM模型类型枚举，排序规则与SVM_MOD里的顺序一致
    static final String[] svm_type_table = {"c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr",};

    //核函数的类型枚举，排序规则与SVM_MOD里的顺序一致
    static final String[] kernel_type_table = {"linear", "polynomial", "rbf", "sigmoid", "precomputed"};

    /**
     * 保存模型model到文件model_file_name里面去.
     */
    public static void svm_save_model(String model_file_name, svm_model model) throws IOException {

        //数据输出流
        DataOutputStream fp = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(model_file_name)));

        svm_parameter param = model.param;

        //写入SVM的类型，默认为C_SVC
        fp.writeBytes("svm_type " + svm_type_table[param.svm_type] + "\n");
        //写入核函数的类型，默认为RBF，即高斯核
        fp.writeBytes("kernel_type " + kernel_type_table[param.kernel_type] + "\n");

        //根据不同的核函数选择性写入数据
        if (param.kernel_type == svm_parameter.POLY)
            fp.writeBytes("degree " + param.degree + "\n");

        if (param.kernel_type == svm_parameter.POLY ||
                param.kernel_type == svm_parameter.RBF ||
                param.kernel_type == svm_parameter.SIGMOID)
            fp.writeBytes("gamma " + param.gamma + "\n");

        if (param.kernel_type == svm_parameter.POLY ||
                param.kernel_type == svm_parameter.SIGMOID)
            fp.writeBytes("coef0 " + param.coef0 + "\n");

        //写入【类别数量】和【支持向量数量】
        int nr_class = model.nr_class;
        int l = model.l;
        fp.writeBytes("nr_class " + nr_class + "\n");
        fp.writeBytes("total_sv " + l + "\n");

        //写入w和b
        {
            fp.writeBytes("rho");
            for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
                fp.writeBytes(" " + model.rho[i]);
            fp.writeBytes("\n");
        }

        //写入类别标签
        if (model.label != null) {
            fp.writeBytes("label");
            for (int i = 0; i < nr_class; i++)
                fp.writeBytes(" " + model.label[i]);
            fp.writeBytes("\n");
        }

        if (model.probA != null) // regression has probA only
        {
            fp.writeBytes("probA");
            for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
                fp.writeBytes(" " + model.probA[i]);
            fp.writeBytes("\n");
        }
        if (model.probB != null) {
            fp.writeBytes("probB");
            for (int i = 0; i < nr_class * (nr_class - 1) / 2; i++)
                fp.writeBytes(" " + model.probB[i]);
            fp.writeBytes("\n");
        }

        if (model.nSV != null) {
            fp.writeBytes("nr_sv");
            for (int i = 0; i < nr_class; i++)
                fp.writeBytes(" " + model.nSV[i]);
            fp.writeBytes("\n");
        }

        fp.writeBytes("SV\n");
        double[][] sv_coef = model.sv_coef;
        svm_node[][] SV = model.SV;

        for (int i = 0; i < l; i++) {
            for (int j = 0; j < nr_class - 1; j++)
                fp.writeBytes(sv_coef[j][i] + " ");

            svm_node[] p = SV[i];
            if (param.kernel_type == svm_parameter.PRECOMPUTED)
                fp.writeBytes("0:" + (int) (p[0].value));
            else
                for (int j = 0; j < p.length; j++)
                    fp.writeBytes(p[j].index + ":" + p[j].value + " ");
            fp.writeBytes("\n");
        }

        fp.close();
    }

    private static double atof(String s) {
        return Double.parseDouble(s);
    }

    private static int atoi(String s) {
        return Integer.parseInt(s);
    }

    /**
     * 读取模型文件的文件头.
     *
     * @param fp    文件指针
     * @param model 需要加载的模型
     * @return 如果参数没有正确读取，则返回错误
     */
    private static boolean read_model_header(BufferedReader fp, svm_model model) {
        //svm必要参数
        svm_parameter param = new svm_parameter();
        model.param = param;
        //仅用于训练时候的参数不会被分配，但是为了安全起见，数组被分配为null
        param.nr_weight = 0;
        param.weight_label = null;
        param.weight = null;

        try {
            while (true) {
                //逐条读行并填入参数
                String cmd = fp.readLine();
                //读取空格后面的第一个参数
                String arg = cmd.substring(cmd.indexOf(' ') + 1);

                //读取SVM的类型
                if (cmd.startsWith("svm_type")) {
                    int i;
                    //遍历模型类型，如果找到，就设置
                    for (i = 0; i < svm_type_table.length; i++) {
                        if (arg.contains(svm_type_table[i])) {
                            param.svm_type = i;
                            break;
                        }
                    }
                    if (i == svm_type_table.length) {
                        System.err.print("unknown svm type.\n");
                        return false;
                    }
                    //读取核函数类型
                } else if (cmd.startsWith("kernel_type")) {
                    int i;
                    //遍历核函数类型，如果找到，就设置
                    for (i = 0; i < kernel_type_table.length; i++) {
                        if (arg.contains(kernel_type_table[i])) {
                            param.kernel_type = i;
                            break;
                        }
                    }
                    if (i == kernel_type_table.length) {
                        System.err.print("unknown kernel function.\n");
                        return false;
                    }
                    //读取各种参数
                } else if (cmd.startsWith("degree"))
                    param.degree = atoi(arg);
                else if (cmd.startsWith("gamma"))
                    param.gamma = atof(arg);
                else if (cmd.startsWith("coef0"))
                    param.coef0 = atof(arg);
                    //类别的数量
                else if (cmd.startsWith("nr_class"))
                    model.nr_class = atoi(arg);
                    //读取支持向量的个数
                else if (cmd.startsWith("total_sv"))
                    model.l = atoi(arg);
                    //决策参数偏移项b
                else if (cmd.startsWith("rho")) {
                    //一共有nr_class * (model.nr_class - 1) / 2个模型
                    int n = model.nr_class * (model.nr_class - 1) / 2;
                    model.rho = new double[n];
                    StringTokenizer st = new StringTokenizer(arg);
                    for (int i = 0; i < n; i++)
                        model.rho[i] = atof(st.nextToken());
                    //类别标签
                } else if (cmd.startsWith("label")) {
                    int n = model.nr_class;
                    model.label = new int[n];
                    StringTokenizer st = new StringTokenizer(arg);
                    for (int i = 0; i < n; i++)
                        model.label[i] = atoi(st.nextToken());
                } else if (cmd.startsWith("probA")) {
                    int n = model.nr_class * (model.nr_class - 1) / 2;
                    model.probA = new double[n];
                    StringTokenizer st = new StringTokenizer(arg);
                    for (int i = 0; i < n; i++)
                        model.probA[i] = atof(st.nextToken());
                } else if (cmd.startsWith("probB")) {
                    int n = model.nr_class * (model.nr_class - 1) / 2;
                    model.probB = new double[n];
                    StringTokenizer st = new StringTokenizer(arg);
                    for (int i = 0; i < n; i++)
                        model.probB[i] = atof(st.nextToken());
                    //每个类别的支持向量个数
                } else if (cmd.startsWith("nr_sv")) {
                    int n = model.nr_class;
                    model.nSV = new int[n];
                    StringTokenizer st = new StringTokenizer(arg);
                    for (int i = 0; i < n; i++)
                        model.nSV[i] = atoi(st.nextToken());
                    //具体的支持向量可能单独处理
                } else if (cmd.startsWith("SV")) {
                    break;
                } else {
                    System.err.print("unknown text in model file: [" + cmd + "]\n");
                    return false;
                }
            }
        } catch (Exception e) {
            return false;
        }
        return true;
    }

    /**
     * 加载模型.
     *
     * @param model_file_name 模型文件
     * @return 加载好的模型
     */
    public static svm_model svm_load_model(String model_file_name) throws IOException {
        return svm_load_model(new BufferedReader(new FileReader(model_file_name)));
    }

    public static svm_model svm_load_model(BufferedReader fp) throws IOException {
        /*
         * 从文件中读取各种参数.
         *
         * 主要包括：
         *  - 模型类型
         *  - 核函数类型
         *  - 核函数参数
         *  - 类别数量
         *  - 类别标签
         *  - 支持向量个数
         *  - 决策参数
         *  - 支持向量
         */
        svm_model model = new svm_model();
        model.rho = null;
        model.probA = null;
        model.probB = null;
        model.label = null;
        model.nSV = null;

        //读文件头
        if (!read_model_header(fp, model)) {
            System.err.print("ERROR: failed to read model\n");
            return null;
        }

        //读取支持向量参数
        int m = model.nr_class - 1;
        int l = model.l;
        //支持向量的系数
        model.sv_coef = new double[m][l];
        //支持向量
        model.SV = new svm_node[l][];

        for (int i = 0; i < l; i++) {
            //此时文件指针指向第一行支持向量
            String line = fp.readLine();
            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

            //读取决策系数
            for (int k = 0; k < m; k++)
                model.sv_coef[k][i] = atof(st.nextToken());
            //读取支持向量
            int n = st.countTokens() / 2;
            model.SV[i] = new svm_node[n];
            for (int j = 0; j < n; j++) {
                model.SV[i][j] = new svm_node();
                model.SV[i][j].index = atoi(st.nextToken());
                model.SV[i][j].value = atof(st.nextToken());
            }
        }

        fp.close();
        return model;
    }

    public static String svm_check_parameter(svm_problem prob, svm_parameter param) {

        //检查输入的模型类型
        int svm_type = param.svm_type;
        if (svm_type != svm_parameter.C_SVC &&
                svm_type != svm_parameter.NU_SVC &&
                svm_type != svm_parameter.ONE_CLASS &&
                svm_type != svm_parameter.EPSILON_SVR &&
                svm_type != svm_parameter.NU_SVR)
            return "unknown svm type";

        //检查输入的核函数类型
        int kernel_type = param.kernel_type;
        if (kernel_type != svm_parameter.LINEAR &&
                kernel_type != svm_parameter.POLY &&
                kernel_type != svm_parameter.RBF &&
                kernel_type != svm_parameter.SIGMOID &&
                kernel_type != svm_parameter.PRECOMPUTED)
            return "unknown kernel type";

        //检查输入的gamma值，对多项式核、高斯核、sigmoid和有效
        if ((kernel_type == svm_parameter.POLY ||
                kernel_type == svm_parameter.RBF ||
                kernel_type == svm_parameter.SIGMOID) &&
                param.gamma < 0)
            return "gamma < 0";

        //检查输入的d值，对多项式核有效
        if (kernel_type == svm_parameter.POLY && param.degree < 0)
            return "degree of polynomial kernel < 0";

        //检查缓存大小
        if (param.cache_size <= 0)
            return "cache_size <= 0";

        //检查允许误差
        if (param.eps <= 0)
            return "eps <= 0";

        //检查惩罚参数C，仅对C_SVC、EPSILON_SVR、NU_SVR有效
        if (svm_type == svm_parameter.C_SVC ||
                svm_type == svm_parameter.EPSILON_SVR ||
                svm_type == svm_parameter.NU_SVR)
            if (param.C <= 0)
                return "C <= 0";

        //检查惩罚参数nu，仅对NU_SVC、ONE_CLASS、NU_SVR有效
        if (svm_type == svm_parameter.NU_SVC ||
                svm_type == svm_parameter.ONE_CLASS ||
                svm_type == svm_parameter.NU_SVR)
            if (param.nu <= 0 || param.nu > 1)
                return "nu <= 0 or nu > 1";

        //检查EPSILON_SVR的p值
        if (svm_type == svm_parameter.EPSILON_SVR)
            if (param.p < 0)
                return "p < 0";

        //检查启发式设置值
        if (param.shrinking != 0 &&
                param.shrinking != 1)
            return "shrinking != 0 and shrinking != 1";

        //检查probability
        if (param.probability != 0 &&
                param.probability != 1)
            return "probability != 0 and probability != 1";

        if (param.probability == 1 &&
                svm_type == svm_parameter.ONE_CLASS)
            return "one-class SVM probability output not supported yet";

        //检查NU_SVC是否可行
        if (svm_type == svm_parameter.NU_SVC) {
            int l = prob.l;
            int max_nr_class = 16;
            int nr_class = 0;
            int[] label = new int[max_nr_class];
            int[] count = new int[max_nr_class];

            //统计各个类出现的次数，即各个类所属的样本数
            int i;
            for (i = 0; i < l; i++) {
                int this_label = (int) prob.y[i];
                int j;
                for (j = 0; j < nr_class; j++)
                    if (this_label == label[j]) {
                        ++count[j];
                        break;
                    }
                //对类进行两两检查
                if (j == nr_class) {
                    if (nr_class == max_nr_class) {
                        max_nr_class *= 2;
                        int[] new_data = new int[max_nr_class];
                        System.arraycopy(label, 0, new_data, 0, label.length);
                        label = new_data;

                        new_data = new int[max_nr_class];
                        System.arraycopy(count, 0, new_data, 0, count.length);
                        count = new_data;
                    }
                    label[nr_class] = this_label;
                    count[nr_class] = 1;
                    ++nr_class;
                }
            }

            for (i = 0; i < nr_class; i++) {
                int n1 = count[i];
                for (int j = i + 1; j < nr_class; j++) {
                    int n2 = count[j];
                    if (param.nu * (n1 + n2) / 2 > Math.min(n1, n2))
                        return "specified nu is infeasible";
                }
            }
        }

        return null;
    }

    /**
     * 检查模型是否有概率预测.
     */
    public static boolean svm_check_probability_model(svm_model model) {
        return ((model.param.svm_type == svm_parameter.C_SVC || model.param.svm_type == svm_parameter.NU_SVC) &&
                model.probA != null && model.probB != null)
                ||
                ((model.param.svm_type == svm_parameter.EPSILON_SVR || model.param.svm_type == svm_parameter.NU_SVR) &&
                        model.probA != null);
    }

    /**
     * 设置打印参数.
     */
    public static void svm_set_print_string_function(svm_print_interface print_func) {
        if (print_func == null)
            svm_print_string = svm_print_stdout;
        else
            svm_print_string = print_func;
    }
}
