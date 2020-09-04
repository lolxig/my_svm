import libsvm.*;

import java.io.*;
import java.util.*;

class svm_train {
    private svm_parameter param;    //通过命令行获取的各种参数
    private svm_problem prob;       //读取的特征集和标签集数据转换为问题
    private String input_file_name; //输入文件路径
    private String model_file_name; //模型文件路径
    private int cross_validation;   //交叉验证选择核函数标志位，默认为0，可以通过-v 参数设置为1
    private int nr_fold;            //n重交叉验证

    //
    private static final svm_print_interface svm_print_null = s -> {
    };

    /**
     * 打印帮助信息.
     */
    private static void exit_with_help() {
        System.out.print(
                "Usage: svm_train [options] training_set_file [model_file]\n"

                        + "options:\n"

                        //设置SVM的类型
                        + "-s svm_type : set type of SVM (default 0)\n"
                        + "	0 -- C-SVC		(multi-class classification)\n"
                        + "	1 -- nu-SVC		(multi-class classification)\n"
                        + "	2 -- one-class SVM\n"
                        + "	3 -- epsilon-SVR	(regression)\n"
                        + "	4 -- nu-SVR		(regression)\n"

                        //设置核函数，默认为高斯核
                        + "-t kernel_type : set type of kernel function (default 2)\n"
                        + "	0 -- linear: u'*v\n"
                        + "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
                        + "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
                        + "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
                        + "	4 -- precomputed kernel (kernel values in training_set_file)\n"

                        //设置degree，默认为3
                        + "-d degree : set degree in kernel function (default 3)\n"

                        //设置核函数中γ的值，默认为1/k，k为特征（或者说是属性）数
                        + "-g gamma : set gamma in kernel function (default 1/num_features)\n"

                        //设置核函数中的coef 0，默认值为0
                        + "-r coef0 : set coef0 in kernel function (default 0)\n"

                        //设置C-SVC、ε-SVR、n - SVR中从惩罚系数C，默认值为1
                        + "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"

                        //设置v-SVC、one-class-SVM与n-SVR中参数n，默认值0.5
                        + "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"

                        //设置v-SVR的损失函数中的e，默认值为0.1
                        + "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"

                        //设置cache内存大小，以MB为单位，默认值为100
                        + "-m cachesize : set cache memory size in MB (default 100)\n"

                        //设置终止准则中的可容忍偏差，默认值为0.001
                        + "-e epsilon : set tolerance of termination criterion (default 0.001)\n"

                        //是否使用启发式，可选值为0或1，默认值为1
                        + "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"

                        //是否计算SVC或SVR的概率估计，可选值0或1，默认0
                        + "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"

                        //对各类样本的惩罚系数C加权，默认值为1
                        + "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"

                        //n折交叉验证模式
                        + "-v n : n-fold cross validation mode\n"

                        //设置是否打印输出，默认有打印输出
                        + "-q : quiet mode (no outputs)\n");
        System.exit(1);
    }

    //交叉验证
    private void do_cross_validation() {
        int i;
        int total_correct = 0;  //分类正确的数量
        double total_error = 0; //分类错误的数量
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        double[] target = new double[prob.l];

        svm.svm_cross_validation(prob, param, nr_fold, target);
        if (param.svm_type == svm_parameter.EPSILON_SVR ||
                param.svm_type == svm_parameter.NU_SVR) {
            for (i = 0; i < prob.l; i++) {
                double y = prob.y[i];
                double v = target[i];
                total_error += (v - y) * (v - y);
                sumv += v;
                sumy += y;
                sumvv += v * v;
                sumyy += y * y;
                sumvy += v * y;
            }
            System.out.print("Cross Validation Mean squared error = " + total_error / prob.l + "\n");
            System.out.print("Cross Validation Squared correlation coefficient = " +
                    ((prob.l * sumvy - sumv * sumy) * (prob.l * sumvy - sumv * sumy)) /
                            ((prob.l * sumvv - sumv * sumv) * (prob.l * sumyy - sumy * sumy)) + "\n"
            );
        } else {
            for (i = 0; i < prob.l; i++)
                if (target[i] == prob.y[i])
                    ++total_correct;
            System.out.print("Cross Validation Accuracy = " + 100.0 * total_correct / prob.l + "%\n");
        }
    }

    private void run(String[] argv) throws IOException {
        //解析命令行，将传入的命令行配置存放到SvmParameter param中，并获取input file、model file(如果有)
        //在命令行里没有设置的参数，将赋予一些默认值
        parse_command_line(argv);

        //读取数据文件，并将数据切分成特征集和标签集，存放到SvmProblem prob中
        read_problem();

        //参数检查
        String error_msg = svm.svm_check_parameter(prob, param);

        if (error_msg != null) {
            System.err.print("ERROR: " + error_msg + "\n");
            System.exit(1);
        }

        if (cross_validation != 0) {
            //训练多个模型并进行n-fold交叉验证，并打印相关信息（例如均方误差、精度等信息）
            do_cross_validation();
        } else {
            //根据配置训练模型并进行保存
            svm_model model = svm.svm_train(prob, param);
            svm.svm_save_model(model_file_name, model);
        }
    }

    public static void main(String[] argv) throws IOException {
        FileReader in = new FileReader("local_param_train");
        char[] buff = new char[1024];
        int len = in.read(buff);
        //输入命令行参数
        String inputParams = new String(buff, 0, len);
        //切割命令行参数
        String[] params = inputParams.split(" ");

        svm_train t = new svm_train();
        t.run(params);
    }

    private static double atof(String s) {
        double d = Double.parseDouble(s);
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            System.err.print("NaN or Infinity in input\n");
            System.exit(1);
        }
        return (d);
    }

    private static int atoi(String s) {
        return Integer.parseInt(s);
    }

    private void parse_command_line(String[] argv) {
        int i;
        svm_print_interface print_func = null;    // default printing to stdout

        param = new svm_parameter();
        // default values
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.degree = 3;
        param.gamma = 0;    // 1/num_features
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100; //默认缓存100MB
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;    //默认收缩启发式标志开启
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];
        cross_validation = 0;

        // parse options
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-') break;
            if (++i >= argv.length)
                exit_with_help();
            switch (argv[i - 1].charAt(1)) {
                case 's':
                    param.svm_type = atoi(argv[i]);
                    break;
                case 't':
                    param.kernel_type = atoi(argv[i]);
                    break;
                case 'd':
                    param.degree = atoi(argv[i]);
                    break;
                case 'g':
                    param.gamma = atof(argv[i]);
                    break;
                case 'r':
                    param.coef0 = atof(argv[i]);
                    break;
                case 'n':
                    param.nu = atof(argv[i]);
                    break;
                case 'm':
                    param.cache_size = atof(argv[i]);
                    break;
                case 'c':
                    param.C = atof(argv[i]);
                    break;
                case 'e':
                    param.eps = atof(argv[i]);
                    break;
                case 'p':
                    param.p = atof(argv[i]);
                    break;
                case 'h':
                    param.shrinking = atoi(argv[i]);
                    break;
                case 'b':
                    param.probability = atoi(argv[i]);
                    break;
                case 'q':
                    print_func = svm_print_null;
                    i--;
                    break;
                case 'v':
                    cross_validation = 1;
                    nr_fold = atoi(argv[i]);
                    if (nr_fold < 2) {
                        System.err.print("n-fold cross validation: n must >= 2\n");
                        exit_with_help();
                    }
                    break;
                case 'w':
                    ++param.nr_weight;
                {
                    int[] old = param.weight_label;
                    param.weight_label = new int[param.nr_weight];
                    System.arraycopy(old, 0, param.weight_label, 0, param.nr_weight - 1);
                }

                {
                    double[] old = param.weight;
                    param.weight = new double[param.nr_weight];
                    System.arraycopy(old, 0, param.weight, 0, param.nr_weight - 1);
                }

                param.weight_label[param.nr_weight - 1] = atoi(argv[i - 1].substring(2));
                param.weight[param.nr_weight - 1] = atof(argv[i]);
                break;
                default:
                    System.err.print("Unknown option: " + argv[i - 1] + "\n");
                    exit_with_help();
            }
        }

        //设置打印函数
        svm.svm_set_print_string_function(print_func);

        // determine filenames
        if (i >= argv.length)
            exit_with_help();

        //检测输入文件路径
        input_file_name = argv[i];

        if (i < argv.length - 1)
            //如果输入文件路径之后还有参数的话，就是model文件路径
            model_file_name = argv[i + 1];
        else {
            //在输入文件路径中获取输入文件名
            int p = argv[i].lastIndexOf('/');
            ++p;    // whew...
            model_file_name = argv[i].substring(p) + ".model";
        }
    }


    //将数据读入的内存中，存入到svm_problem问题集里面去
    private void read_problem() throws IOException {
        BufferedReader fp = new BufferedReader(new FileReader(input_file_name));
        Vector<Double> vy = new Vector<>();
        Vector<svm_node[]> vx = new Vector<>();
        int max_index = 0;

        while (true) {
            String line = fp.readLine();
            if (line == null) break;

            //切割行数据
            String[] elemts = line.split("\t", -1);

            //提取标签集
            vy.addElement(atof(elemts[elemts.length - 1]));
            //提取特征集
            svm_node[] x = new svm_node[elemts.length - 1];
            for (int i = 0; i < elemts.length - 1; i++) {
                x[i] = new svm_node();
                x[i].index = i + 1;
                x[i].value = atof(elemts[i]);
            }

//            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

//            vy.addElement(atof(st.nextToken()));
//            int m = st.countTokens() / 2;
//            svm_node[] x = new svm_node[m];
//            for (int j = 0; j < m; j++) {
//                x[j] = new svm_node();
//                x[j].index = atoi(st.nextToken());
//                x[j].value = atof(st.nextToken());
//            }
//            if (m > 0) max_index = Math.max(max_index, x[m - 1].index);

            if (elemts.length - 1 > 0)
                max_index = Math.max(max_index, x[elemts.length - 2].index);

            vx.addElement(x);
        }

        //将取得的数据存入到svm_problem问题集里面去
        prob = new svm_problem();
        prob.l = vy.size();
        prob.x = new svm_node[prob.l][];
        for (int i = 0; i < prob.l; i++)
            prob.x[i] = vx.elementAt(i);
        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy.elementAt(i);

        //设置默认gamma值
        if (param.gamma == 0 && max_index > 0)
            param.gamma = 1.0 / max_index;

        //用户自定义核函数
        if (param.kernel_type == svm_parameter.PRECOMPUTED)
            for (int i = 0; i < prob.l; i++) {
                if (prob.x[i][0].index != 0) {
                    System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
                    System.exit(1);
                }
                if ((int) prob.x[i][0].value <= 0 || (int) prob.x[i][0].value > max_index) {
                    System.err.print("Wrong input format: sample_serial_number out of range\n");
                    System.exit(1);
                }
            }

        fp.close();
    }
}
